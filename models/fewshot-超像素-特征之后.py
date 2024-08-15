# @Time : 2021/10/5
# @Author : Z.chang
# @FileName: fewshot.py
# @Software: PyCharm
# @Description：Few-shot  backbone 50+5shot-28000 时间2021-12-03

from collections import OrderedDict

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import to_tensor, to_pil_image
from skimage.future import graph
from skimage.segmentation import mark_boundaries
from models.vgg import Encoder
from models import resnet_50_101
from models.vit_model import VisionTransformer
from functools import partial
# from pytorch_pretrained_vit import ViT
from util import utils
from util.seed_init import place_seed_points

import numpy as np
from torchvision.utils import make_grid
# from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from skimage import io
import matplotlib.pyplot as plt
from skimage.util import img_as_float
from skimage.segmentation import mark_boundaries
from skimage.measure import regionprops
# from skimage.segmentation import slic as myslic  # 测试快慢，随机调试
from skimage.segmentation import slic_superpixels
from models import trimodal_attention as att_fusion
from util.utils import rgb2gray
import cupy


class FewShotSeg(nn.Module):
    """
       Args:
        in_channels:
            number of input channels
        pretrained_path:
            path of the model for initialization
        cfg:
            model configurations
    """

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, depth=12, act_layer=None, norm_layer=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        # 超像素参数 先定义在这里 后面改到config
        # self.train_iter = 10
        # self.eval_iter = 5

        self.config = cfg or {'align': False}
        # # # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', resnet_50_101.resnet50(pretrained=True)), ]))

        # =============== vit 的 encoder block ==========start========
        # norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        # act_layer = act_layer or F.gelus
        # @CZW VIT
        # self.blocks = nn.Sequential(*[
        #     Block(dim=53, num_heads=4, mlp_ratio=4.0, qkv_bias=True, qk_scale=None,
        #           drop_ratio=0., attn_drop_ratio=0., drop_path_ratio=0.,
        #           norm_layer=norm_layer, act_layer=act_layer)
        #     for i in range(depth)
        # ])
        # =============== vit 的 encoder block ==========end========

        # ===============  vision transfromer ==========start========
        # self.has_logits = False
        # # con2d  降维 qu ery and support -> 统一降维，适配vit的输入和输出
        self.proj = nn.Conv2d(1024, 512, kernel_size=(1, 1), stride=1)
        # # @gl 修改in_c  余弦相似度计算时 通道数必须匹配  所以embed_dim修改为1024  维度1024 head得换成可以整除的16
        self.vit_model = VisionTransformer(img_size=448,
                                           patch_size=32,
                                           in_c=512,
                                           embed_dim=512,
                                           # embed_dim=1024,
                                           depth=12,
                                           num_heads=16,
                                           # distilled=True,
                                           representation_size=None,
                                           num_classes=0)
        # ===============  vision transfromer ==========end========
        # config = dict(hidden_size=512, num_heads=8, num_layers=6)
        #         # self.vit_pre = ViT.from_config('B_16_imagenet1k', config)
        # self.vit_pre = ViT('B_16_imagenet1k', pretrained=True)

        # @czw
        self.avgpool = nn.AdaptiveAvgPool2d(1) # 没有mask直接调用

    def forward(self, supp_imgs, fore_mask, back_mask, qry_imgs):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors
        """
        # print("forward")
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        ###### Extract and map features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts_proj_out = self.encoder(imgs_concat)  # 2 512 56 56
        # @GL 通过 1*1 1的卷积核  降维到1*512 VGG不需要
        # img_fts_proj_out = self.proj(img_fts_resnet_out)  # 2 512 56 56
        fts_size = img_fts_proj_out.shape[-2:]  # 最后输出的维度
        supp_fts_proj_out = img_fts_proj_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # 1 1 1 512 56 56 # support_Way x Shot x B x C x H' x W'
        qry_fts_proj_out = img_fts_proj_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W' 1 1 512 56 56

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        ###### Compute loss ######
        align_loss = 0  # query_mask ->support mask
        outputs = []
        for epi in range(batch_size):

            # 超像素计算query  start
            query_feture_map = qry_fts_proj_out.squeeze().permute(1, 2, 0)  # 56 56 512
            query_feture_map = query_feture_map.cpu().to(torch.float64)  # 56 56 512 (原始是单张image，现在是转换为tensor，加载到cpu中)
            temp_nump = query_feture_map.detach().numpy()
            segments_slic = slic_superpixels(temp_nump, n_segments=512, compactness=40)
            # [776]

            gray_img = rgb2gray(query_feture_map.detach().numpy())
            regions = regionprops(segments_slic, intensity_image=gray_img)  # 拿特征

            for props in regions:  # 每个区域的特征
                # image_region = props.image
                print("区域内像素点个数")
                print(props.area)
                print("区域内像素点坐标")
                print(props.coords)
                minr, minc, maxr, maxc = props.bbox  # 特征坐标
                # crop the segment
                # image_region = segments_slic[minr:maxr, minc:maxc]
                image_region = query_feture_map[minr:maxr, minc:maxc, ]  # 原图得到区域，，取平均特征 [2, 3, 512]
                pool_point = self.avgpool(image_region.permute(2, 0, 1))  # [2, 1, 1]
            # superpixels = slic(query_feture_map.detach().numpy(), multichannel=True, n_segments=512, sigma=5)  # 56 56
            # # query_img = to_pil_image(query_feture_map)
            # # query_img = np.array(query_feture_map)
            # # query_superpixels = mark_boundaries(query_img, superpixels)
            # # 获取质心点
            # # regions = regionprops(superpixels)
            # # for props in regions:
            # #     cx, cy = props.centroid
            # # query_superpixels = torch.Tensor(query_superpixels).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # 查询特征转换为超像素
            # superpixels = torch.Tensor(superpixels).unsqueeze(0).unsqueeze(0).cuda(0)  # 1 1 56 56 类似一个mask
            # query_feture_superpixels = [[self.getFeatures(qry_fts_proj_out[way, [epi]],
            #                                               superpixels[way, [epi]])
            #                              for way in range(n_ways)]]  # [[1 512]] list

            """单独获取超像素的个数（超像素点、区域个数）
            1）利用超像素分割支持图片，获得支持图片超像素的的个数；
            2）利用Support的mask与超像素的个数进行比较，旨在使得超像素尽可能落在mask内
            3）逐步优化，获得超像素的个数，后续用固定参数。
            """
            print(supp_fts_proj_out.shape)
            slic = cv2.ximgproc.createSuperpixelSLIC(supp_fts_proj_out.cpu(), region_size=20, ruler=20.0)
            slic.iterate(10)  # 迭代次数，越大效果越好
            mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
            label_slic = slic.getLabels()  # 获取超像素标签
            number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
            print(mask_slic)
            print(label_slic)
            print(number_slic)
            # 超像素计算end

            # ======mask==============前景+背景+vit；+前景
            # supp_fg_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
            #                                  fore_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]  # [[ 1 512 ]]
            # print(len(supp_fg_fts))
            # supp_fg_fts = [[self.handle_vit(self.vit_model(F.interpolate(
            #     self.getFeatures(supp_fts_proj_out[way, shot, [epi]], fore_mask[way, shot, [epi]])[..., None, None],
            #     size=fore_mask.shape[-2:], mode='bilinear')).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
            #                                 fore_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]
            # 背景送入vit
            # supp_bg_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
            #                                  back_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]
            """在support中引入超像素 @czb 先过滤掉背景，然后在前景中利用超像素聚类算法获取超像素的原型"""  # @czb
            """过滤掉背景的前景特征map"""
            supp_fg_map = [[(F.interpolate(
                self.getFeatures(supp_fts_proj_out[way, shot, [epi]], fore_mask[way, shot, [epi]])[..., None, None],
                size=fore_mask.shape[-2:], mode='bilinear'))
                for shot in range(n_shots)] for way in range(n_ways)]  # list[1 512 448 448]
            support_fg_maps = supp_fg_map[0][0].squeeze().permute(1, 2, 0)  # tensor 448 448 512
            support_fg_maps = support_fg_maps.cpu().to(torch.float64)  # 448 448 512
            sup_fg_superpixels = slic(support_fg_maps.detach().numpy(), multichannel=True, n_segments=512,
                                      sigma=5)  # 448 448
            sup_fg_superpixels = torch.Tensor(sup_fg_superpixels).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda(
                0)  # 1 1 1 448 448 类似一个mask
            """支持图片的前景超像素"""
            supp_pixels_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
                                                 sup_fg_superpixels[way, shot, [epi]])
                                for shot in range(n_shots)] for way in
                               range(n_ways)]  # [[ 1 512 ]] # 仅仅拿到support 前景的超像素
            """ support branch superpixels end """
            """支持图片的背景超像素 start"""
            supp_bg_map = [[(F.interpolate(
                self.getFeatures(supp_fts_proj_out[way, shot, [epi]], back_mask[way, shot, [epi]])[..., None, None],
                size=back_mask.shape[-2:], mode='bilinear'))
                for shot in range(n_shots)] for way in range(n_ways)]  # list[1 512 448 448]
            support_bg_maps = supp_bg_map[0][0].squeeze().permute(1, 2, 0)  # tensor 448 448 512
            support_bg_maps = support_bg_maps.cpu().to(torch.float64)  # 448 448 512
            sup_bg_superpixels = slic(support_bg_maps.detach().numpy(), multichannel=True, n_segments=512,
                                      sigma=5)  # 448 448
            sup_bg_superpixels = torch.Tensor(sup_bg_superpixels).unsqueeze(0).unsqueeze(0).unsqueeze(0).cuda(
                0)  # 1 1 1 448 448 类似一个mask

            supp_bg_fts = [[self.handle_vit(self.vit_model(F.interpolate(
                self.getFeatures(supp_fts_proj_out[way, shot, [epi]], sup_bg_superpixels[way, shot, [epi]])[
                    ..., None, None],
                size=sup_bg_superpixels.shape[-2:], mode='bilinear')).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
                                            sup_bg_superpixels[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]  # 1 512 list
            """支持图片的背景超像素结束end"""
            # @ czb
            # fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            fg_prototypes, bg_prototype = self.getPrototype(supp_pixels_fts, supp_bg_fts)
            # superpixels_prototype, superpixels_prototype1 = self.getPrototype(supp_pixels, supp_bg_fts) # @czb 加入粗糙的support分支的超像素原型

            # fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            ###### Compute the distance ######
            # prototypes = [bg_prototype, ] + fg_prototypes
            prototypes = [bg_prototype, ] + fg_prototypes
            # 模态融合 前景和背景分别采用超像素原型强化
            """查询特征"""
            query_feture_map = qry_fts_proj_out.squeeze().permute(1, 2, 0)  # 56 56 512
            query_feture_map = query_feture_map.cpu().to(torch.float64)  # 56 56 512 转换为list[512*1]


            new_prototypes = []
            for prototype in prototypes:
                fusion_prototype = att_fusion.bi_modal_attention(query_feture_superpixels[0][0].permute(1, 0),
                                                                 prototype.permute(1, 0))
                new_prototypes.append(fusion_prototype.permute(1, 0))
            # 模态融合结束
            # mark_boundaries(query_feture_map, superpixels)
            dist = [self.calDist(qry_fts_proj_out[:, epi], prototype) for prototype in new_prototypes]  # @czb 原始查询特征特征
            # dist = [self.calDist(query_superpixels[:, epi], prototype) for prototype in new_prototypes]
            pred = torch.stack(dist, dim=1)  #
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            ###### Prototype alignment loss ######

            # 测试阶段 self.training为Flase 说明测试阶段没有执行if
            # if self.config['align'] and self.training:
            flag = False  # 定义一个flag  True则执行CG, False 则不执行CG
            if self.config['align'] and flag:
                align_loss_epi = self.alignLoss(qry_fts_proj_out[:, epi], pred, supp_fts_proj_out[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size

    ###################@czb计算Query->Resnet/VGG->Feature与Support->Resnet/VGG->Vit->prototype之间的余弦相似度 #################
    def calDist(self, query_cnn_out, prototype, scaler=20):
        """
        Calculate the distance between features and prototypes

        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        dist = F.cosine_similarity(query_cnn_out, prototype[..., None, None], dim=1) * scaler
        return dist

    # def getFeatures(self, fts, mask, is_vit):
    #     """
    #     Extract foreground and background features via masked average pooling
    #     全卷积网络（FCN）能够保留输入图像的中每个像素相对位置；所以通过将二值 mask 与提取到的特征图相乘就可以完全保留目标的特征信息，
    #     排除掉背景等无关类别的特征
    #     Args:
    #         fts: input features, expect shape: 1 x C x H' x W'
    #         mask: binary mask, expect shape: 1 x H x W
    #     """
    #     fts = F.interpolate(fts, size=mask.shape[-2:],
    #                         mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)
    #     # @czb
    #     if is_vit:
    #         masked_fts = torch.sum(fts, dim=(2, 3)) \
    #                      / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
    #     else:
    #         masked_fts = fts * mask[None, ...]
    #     # masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
    #     #                  / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
    #     result1 = np.array(masked_fts.cpu())
    #     return masked_fts
    #  @GL  常规getFeatures
    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        全卷积网络（FCN）能够保留输入图像的中每个像素相对位置；所以通过将二值 mask 与提取到的特征图相乘就可以完全保留目标的特征信息，
        排除掉背景等无关类别的特征
        Args:
            fts: input features, expect shape: 1 x C x H' x W' 1 512 448 448
            mask: binary mask, expect shape: 1 x H x W 1 448 448
        """
        fts = F.interpolate(fts, size=mask.shape[-2:],
                            mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)

        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts

    #  @ 针对vit前后 进行mask 以及sum
    def handle_vit(self, fts, mask):
        """
            对vit输出求均值
        """
        # fts = F.interpolate(fts, size=mask.shape[-2:],
        #                     mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)
        # if is_fore_vit:  # 送入vit前mask
        #     masked_fts = fts * mask[None, ...]
        # else:  # vit 出来进行sum
        #     masked_fts = torch.sum(fts, dim=(2, 3)) \
        #                  / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        masked_fts = torch.sum(fts, dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
        return masked_fts

    # @czb ################通过平均前景和背景特征获得原型###############
    def getPrototype(self, fg_fts, bg_fts):
        """
        Average the features to obtain the prototype，单一原型无法完全准确表示（类似于聚类，通过聚类不同的类可以达到同样的效果），提升多原型（multi-prototype）

        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C]
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [sum(way) / n_shots for way in fg_fts]
        bg_prototype = sum([sum(way) / n_shots for way in bg_fts]) / n_ways
        # print(bg_prototype.shape)
        return fg_prototypes, bg_prototype

    ############## # @CZB过渡段学习CCG(Query->Support)##################
    #####@
    def alignLoss(self, query_vgg_out, pred, support_resnet_out, support_fore_mask, support_back_mask):
        """
        Compute the loss for the prototype alignment branch

        Args:
            query_resnet_out: embedding features for query images
                expect shape: N x C x H' x W'
            pred: predicted segmentation score
                expect shape: N x (1 + Way) x H x W
            support_resnet_out: embedding features for support images
                expect shape: Way x Shot x C x H' x W'
            support_fore_mask: foreground masks for support images
                expect shape: way x shot x H x W
            support_back_mask: background masks for support images
                expect shape: way x shot x H x W
        """
        n_ways, n_shots = len(support_fore_mask), len(support_fore_mask[0])
        # Mask and get query prototype
        pred_mask = pred.argmax(dim=1, keepdim=True)  # N x 1 x H' x W'
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  # 前景+1个背景
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  # 没懂
        ##########@czb query-mask########
        pred_mask = torch.stack(binary_masks, dim=1).float()  # N x (1 + Way) x 1 x H' x W'
        query_prototypes = torch.sum(query_vgg_out.unsqueeze(1) * pred_mask, dim=(0, 3, 4))
        ###########获取query的原型###########
        query_prototypes = query_prototypes / (pred_mask.sum((0, 3, 4)) + 1e-5)  # (1 + Way) x C
        # Compute the support loss
        loss = 0
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            prototypes = [query_prototypes[[0]], query_prototypes[[way + 1]]]
            for shot in range(n_shots):
                img_fts = support_resnet_out[way, [shot]]
                supp_dist = [self.calDist(img_fts, prototype) for prototype in prototypes]
                supp_pred = torch.stack(supp_dist, dim=1)
                supp_pred = F.interpolate(supp_pred, size=support_fore_mask.shape[-2:],
                                          mode='bilinear')
                # Construct the support Ground-Truth segmentation
                supp_label = torch.full_like(support_fore_mask[way, shot], 255,
                                             device=img_fts.device).long()
                supp_label[support_fore_mask[way, shot] == 1] = 1
                supp_label[support_back_mask[way, shot] == 1] = 0
                # Compute Loss
                loss = loss + F.cross_entropy(
                    supp_pred, supp_label[None, ...], ignore_index=255) / n_shots / n_ways
        return loss
