"""
@Create Time : 2022/7/4 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
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
from models.vgg import Encoder
from models import resnet_50_101
from models.vit_model import VisionTransformer
from models import trimodal_attention as att_fusion
from functools import partial
from util import utils
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
from skimage.measure import regionprops
from skimage.segmentation import slic_superpixels
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


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

    def __init__(self, in_channels=3, pretrained_path=None, cfg=None, depth=12, act_layer=None, norm_layer=None,
                 swin_model=None):
        super().__init__()
        self.pretrained_path = pretrained_path
        self.config = cfg or {'align': False}
        # # # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        # Swin transformer
        self.swin = swin_model
        # self.swin2feat_0 = nn.Linear(48, 512)  # input: [2, 48, 56, 56] output: [2, 512, 56, 56]
        self.swin2feat_0 = nn.Conv2d(48, 512, kernel_size=(1, 1), stride=1)
        # self.swin2feat_1 = nn.Linear(768, 512)
        # self.swin2feat_1 = nn.Conv2d(768, 512, kernel_size=(1,1), stride=1)
        # self.swin2feat_2 = nn.Linear(14 * 14, 56 * 56)
        # self.swin2feat_2 = nn.Conv2d(14 * 14, 56 * 56, kernel_size=(1,1), stride=1)
        self.feat_merge = nn.Linear(2 * 56 * 56, 56 * 56)
        self.proto_merge = nn.Linear(1024, 512)
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
        self.res_conv = nn.Conv2d(512, 512, kernel_size=(32, 32), stride=32)
        self.trans = nn.Conv2d(512, 512, kernel_size=(1, 1), stride=1)
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
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, supp_imgs, supp_slic_imgs, fore_mask, back_mask, qry_imgs, qry_slic_imgs, support_images_id,
                query_images_id):
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

        """
         通过 query_images_id获取query原图
        """
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]

        # czw 超像素support
        # img_cv2 = cv2.imread(image_path+'\\'+support_images_id[0][0][0]+'.jpg')

        # cv2.imshow("Mask", temp_img_cv2)
        # cv2.waitKey(0)

        # list_new_support = []
        # for way_temp in range(len(supp_imgs)):
        #     shot_temp_list = []
        #     for shot_temp in range(len(supp_imgs[way_temp])):
        #         list_temp_batch = []
        #         for batch_temp in range(len(supp_imgs[way_temp][shot_temp])):
        #             suport_id_temp = support_images_id[way_temp][shot_temp][batch_temp]
        #             img_cv2 = cv2.imread(image_path+'\\'+suport_id_temp+'.jpg')
        #             segments_slic = slic_superpixels.slic(img_cv2, n_segments=512, compactness=30, start_label=1)  # [333, 500]
        #             regions = regionprops(segments_slic, intensity_image=self.rgb2gray(img_cv2))
        #             mask = np.zeros(img_cv2.shape, dtype="uint8")
        #             for props in regions:
        #                 cy, cx = props.centroid
        #                 pixel_proto = img_cv2[int(cy)][int(cx)]
        #                 pixel_index = segments_slic == props.label # [333, 500]
        #                 mask[pixel_index, :] = pixel_proto
        #             list_temp_batch.append(mask)
        #         shot_temp_list.append(list_temp_batch)
        #     list_new_support.append(shot_temp_list)

        ###### Extract and map features ######
        # @GL 原图
        # q原始特征  q平均池化特征   s原始特征
        # torch.Size([1, 1, 512, 56, 56]) torch.Size([512, 1]) torch.Size([1, 1, 1, 512, 56, 56])
        qry_fts_proj_out, qry_fts_proj_pool, supp_fts_proj_out = self.get_encode_feature(batch_size, n_queries, n_shots,
                                                                                         n_ways, qry_imgs, supp_imgs)
        # @GL slic

        qry_slic_fts_proj_out, qry_slic_fts_proj_pool, supp_slic_fts_proj_out = self.get_encode_feature(batch_size,
                                                                                                        n_queries,
                                                                                                        n_shots,
                                                                                                        n_ways,
                                                                                                        qry_slic_imgs,
                                                                                                        supp_slic_imgs)

        qry_fts_proj_out_swin, qry_fts_proj_pool_swin, supp_fts_proj_out_swin = self.get_swin_feature(batch_size,
                                                                                                      n_queries,
                                                                                                      n_shots,
                                                                                                      n_ways, qry_imgs,
                                                                                                      supp_imgs)
        # @ local & global
        qry_slic_fts_proj_out_swin, qry_slic_fts_proj_pool_swin, supp_slic_fts_proj_out_swin = self.get_swin_feature(
            batch_size, n_queries, n_shots,
            n_ways, qry_slic_imgs, supp_slic_imgs)
        sfp_1, sfp2, sfp3, sfp4, sfp5, sfp6 = supp_fts_proj_out.shape
        # todo 对应点位置融合，代替cat
        supp_fts_proj_out_merge = torch.cat([supp_fts_proj_out.flatten(4), supp_fts_proj_out_swin.flatten(4)],
                                            dim=-1)  # [1, 1, 1, 512, 2*56*56]
        supp_fts_proj_out_merge = self.feat_merge(supp_fts_proj_out_merge).view(sfp_1, sfp2, sfp3, sfp4, sfp5, sfp6)
        # supp_fts_proj_out_merge = self.trans(supp_fts_proj_out_merge)

        sfp_1, sfp2, sfp3, sfp4, sfp5, sfp6 = supp_slic_fts_proj_out.shape
        supp_slic_fts_proj_out_merge = torch.cat(
            [supp_slic_fts_proj_out.flatten(4), supp_slic_fts_proj_out_swin.flatten(4)],
            dim=-1)  # [1, 1, 1, 512, 2*56*56]
        supp_slic_fts_proj_out_merge = self.feat_merge(supp_slic_fts_proj_out_merge).view(sfp_1, sfp2, sfp3, sfp4, sfp5,
                                                                                          sfp6)

        sfp_1, sfp2, sfp3, sfp4, sfp5 = qry_fts_proj_out.shape
        qry_fts_proj_out_merge = torch.cat([qry_fts_proj_out.flatten(3), qry_fts_proj_out_swin.flatten(3)],
                                           dim=-1)  # [1, 1, 1, 512, 2*56*56]
        qry_fts_proj_out_merge = self.feat_merge(qry_fts_proj_out_merge).view(sfp_1, sfp2, sfp3, sfp4, sfp5)

        # sfp_1, sfp2, sfp3, sfp4, sfp5 = qry_slic_fts_proj_out.shape
        # qry_slic_fts_proj_out_merge = torch.cat([qry_slic_fts_proj_out.flatten(3), qry_slic_fts_proj_out_swin.flatten(3)], dim=-1)  # [1, 1, 1, 512, 2*56*56]
        # qry_slic_fts_proj_out_merge = self.feat_merge(qry_slic_fts_proj_out_merge).view(sfp_1, sfp2, sfp3, sfp4, sfp5)

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        ###### Compute loss ######
        align_loss = 0  # query_mask ->support mask
        outputs = []
        for epi in range(batch_size):
            # 原始图片
            supp_fg_fts = [[self.getFeatures(supp_fts_proj_out[way, shot, [epi]],
                                             fore_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]  # [[ 1 512 ]]  resnet 1 1024

            supp_slic_fg_fts = [[self.getFeatures(supp_slic_fts_proj_out[way, shot, [epi]],
                                                  fore_mask[way, shot, [epi]])
                                 for shot in range(n_shots)] for way in range(n_ways)]  # [[ 1 512 ]]  resnet 1 1024

            supp_slic_bg_fts = [[self.getFeatures(supp_slic_fts_proj_out[way, shot, [epi]],
                                                  back_mask[way, shot, [epi]])
                                 for shot in range(n_shots)] for way in range(n_ways)]

            supp_bg_fts = [[self.handle_vit(self.vit_model(F.interpolate(
                self.getFeatures(supp_fts_proj_out_merge[way, shot, [epi]], back_mask[way, shot, [epi]])[
                    ..., None, None],
                size=back_mask.shape[-2:], mode='bilinear')).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1])
                                            + self.res_conv(F.interpolate(
                self.getFeatures(supp_fts_proj_out_merge[way, shot, [epi]], back_mask[way, shot, [epi]])[
                    ..., None, None],
                size=back_mask.shape[-2:], mode='bilinear')),
                                            back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]
            #  原始特征和Vit特征做残差，得到的最后特征进行原型学习

            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts, supp_bg_fts)
            fg_slic_prototypes, bg_slic_prototype = self.getPrototype(supp_slic_fg_fts, supp_slic_bg_fts)
            ###### Compute the distance ######
            prototypes = [bg_prototype, ] + fg_prototypes
            slic_prototypes = [bg_slic_prototype, ] + fg_slic_prototypes
            """原图的prototypes和超像素的prototypes  end"""

            """query特征"""
            # query_feture_map = qry_fts_proj_pool.squeeze().permute(1, 2, 0)  # 56 56 512
            # query_feture_map = query_feture_map.cpu().to(torch.float64)  # 56 56 512
            # query_slic_feture_map = qry_slic_fts_proj_pool.squeeze().permute(1, 2, 0)  # 56 56 512
            # query_slic_feture_map = query_slic_feture_map.cpu().to(torch.float64)  # 56 56 512
            """原始图片的模态融合————support和query"""
            new_prototypes_orginal_image = []  # 512*1 原始图片的query全局平均和support原始图片的原型
            for prototype in prototypes:
                fusion_prototype_1 = att_fusion.bi_modal_attention(qry_fts_proj_pool,
                                                                   prototype)
                new_prototypes_orginal_image.append(fusion_prototype_1)
            """超像素的模态融合————support和query"""
            new_prototypes_slic = []  # 512*1 slic的query全局平均和support的slic图片原型
            for prototype_slic in slic_prototypes:
                fusion_prototype_2 = att_fusion.bi_modal_attention(qry_slic_fts_proj_pool,
                                                                   prototype_slic)
                new_prototypes_slic.append(fusion_prototype_2)
            """超像素融合原型2和原始图片的融合原型1进行最终的融合，生成最终的prototypes"""
            # todo @czb
            final_prototypes = []  #
            for pre_i in range(len(new_prototypes_orginal_image)):
                # final_prototypes.append(att_fusion.bi_modal_attention(new_prototypes_orginal_image[pre_i].permute(1, 0), new_prototypes_slic[pre_i].permute(1, 0)))
                # final_prototypes.append(torch.mul(new_prototypes_orginal_image[pre_i].permute(1, 0), new_prototypes_slic[pre_i].permute(1, 0)))
                final_prototypes.append(new_prototypes_orginal_image[pre_i] + new_prototypes_slic[pre_i])
                # new_prototypes_merge = torch.cat([new_prototypes_orginal_image[pre_i], new_prototypes_slic[pre_i]],
                #                                  dim=0).transpose(0, 1)
                # final_prototypes.append(self.proto_merge(new_prototypes_merge))

            # print(final_prototypes[0].shape)
            # final_prototypes = new_prototypes_orginal_image + new_prototypes_slic
            # dist = [self.calDist(qry_fts_proj_out[:, epi].squeeze().permute(1, 2, 0), prototype_temp.permute(1, 0)) for prototype_temp in final_prototypes]
            # pred = torch.stack(dist, dim=0)  #
            dist = [self.calDist(qry_fts_proj_out[:, epi], prototype[:, epi]) for prototype in final_prototypes]
            pred = torch.stack(dist, dim=1)
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            ###### Prototype alignment loss ######

            # 测试阶段 self.training为Flase 说明测试阶段没有执行if
            if self.config['align'] and self.training:
                # flag = False  # 定义一个flag  True则执行CG, False 则不执行CG,消融实验，测试交叉的效果
                # if self.config['align'] and flag:
                align_loss_epi = self.alignLoss(qry_fts_proj_out[:, epi], pred, supp_fts_proj_out[:, :, epi],
                                                fore_mask[:, :, epi], back_mask[:, :, epi])
                align_loss += align_loss_epi

        output = torch.stack(outputs, dim=1)  # N x B x (1 + Wa) x H x W
        output = output.view(-1, *output.shape[2:])
        return output, align_loss / batch_size

    def get_swin_feature(self, batch_size, n_queries, n_shots, n_ways, qry_imgs, supp_imgs):
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # [2, 3, 448, 448]
        img_fts_proj_out = self.swin(imgs_concat)  # out: [2, 196, 768]  out: [2, 512, 56, 56]
        c_n, c_l, c_c = img_fts_proj_out.shape
        # img_fts_proj_out = self.swin2feat_2(self.swin2feat_1(img_fts_proj_out).permute(0,2,1)).view(c_n, 512, 56, 56)
        img_fts_proj_out = self.swin2feat_0(img_fts_proj_out.view(c_n, -1, 56, 56))
        # @GL 通过 1*1 1的卷积核  降维到1*512 VGG不需要
        # img_fts_proj_out = self.proj(img_fts_resnet_out)  # 2 512 56 56
        fts_size = img_fts_proj_out.shape[-2:]  # 最后输出的维度
        supp_fts_proj_out = img_fts_proj_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # 1 1 1 512 56 56 # support_Way x Shot x B x C x H' x W'
        qry_fts_proj_out = img_fts_proj_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W'
        # czw: 对query全局平均池化 512*56*56 -> 512*1
        qry_fts_proj_pool = self.avgpool(qry_fts_proj_out).squeeze(-1).squeeze(-1)
        return qry_fts_proj_out, qry_fts_proj_pool, supp_fts_proj_out

    def get_encode_feature(self, batch_size, n_queries, n_shots, n_ways, qry_imgs, supp_imgs):
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)  # [2, 3, 448, 448]
        img_fts_proj_out = self.encoder(imgs_concat)  # input: 2 1024 56 56 out: [2, 512, 56, 56]
        # @GL 通过 1*1 1的卷积核  降维到1*512 VGG不需要
        # img_fts_proj_out = self.proj(img_fts_resnet_out)  # 2 512 56 56
        fts_size = img_fts_proj_out.shape[-2:]  # 最后输出的维度
        supp_fts_proj_out = img_fts_proj_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # 1 1 1 512 56 56 # support_Way x Shot x B x C x H' x W'
        qry_fts_proj_out = img_fts_proj_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W'
        # czw: 对query全局平均池化 512*56*56 -> 512*1
        qry_fts_proj_pool = self.avgpool(qry_fts_proj_out).squeeze(-1).squeeze(-1)
        return qry_fts_proj_out, qry_fts_proj_pool, supp_fts_proj_out

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2126, 0.7152, 0.0722])

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
        # query_cnn_out： [1, 512, 56, 56] prototype：list[4] 4个tensor
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
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        fts = F.interpolate(fts, size=mask.shape[-2:],
                            mode='bilinear')  # 默认nearest, linear(3D-only), bilinear(4D-only), trilinear(5D-only)

        masked_fts = torch.sum(fts * mask[None, ...], dim=(2, 3)) \
                     / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)  # 1 x C
        return masked_fts

    #  @GL 针对vit前后 进行mask 以及sum
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
