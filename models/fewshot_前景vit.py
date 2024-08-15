# -*- coding: utf-8 -*-
# @Time : 2021/10/5
# @Author : Z.chang
# @FileName: fewshot.py
# @Software: PyCharm
# @Description：VGG+前景送入vit

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.vgg import Encoder
from models import resnet_50_101
from models.vit_model import VisionTransformer
from functools import partial
from pytorch_pretrained_vit import ViT
from util import utils
import numpy as np
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
import torchvision.utils as vutils

writer = SummaryWriter('run_ccg/vit_visual')


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
        self.config = cfg or {'align': False}
        # Encoder
        self.encoder = nn.Sequential(OrderedDict([
            ('backbone', Encoder(in_channels, self.pretrained_path)), ]))
        # self.encoder = nn.Sequential(OrderedDict([
        #     ('backbone', resnet_50_101.resnet101(pretrained=False)), ]))

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
        self.proj = nn.Conv2d(512, 512, kernel_size=(4, 4), stride=4)
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
        n_ways = len(supp_imgs)
        n_shots = len(supp_imgs[0])
        n_queries = len(qry_imgs)
        batch_size = supp_imgs[0][0].shape[0]
        img_size = supp_imgs[0][0].shape[-2:]
        ###### Extract and map features ######
        # print(len(supp_imgs))
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs]
                                + [torch.cat(qry_imgs, dim=0), ], dim=0)
        img_fts_vgg_out = self.encoder(imgs_concat)  # 2 1024 56 56

        fts_size = img_fts_vgg_out.shape[-2:]  # 最后输出的维度
        supp_fts_vgg_out = img_fts_vgg_out[:n_ways * n_shots * batch_size].view(
            n_ways, n_shots, batch_size, -1, *fts_size)  # support_Way x Shot x B x C x H' x W'
        qry_fts_vgg_out = img_fts_vgg_out[n_ways * n_shots * batch_size:].view(
            n_queries, batch_size, -1, *fts_size)  # query_way x B x C x H' x W'

        fore_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in fore_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        back_mask = torch.stack([torch.stack(way, dim=0)
                                 for way in back_mask], dim=0)  # Way x Shot x B x H x W [1, 1, 1, 448, 448]
        ###### Compute loss ######
        align_loss = 0  # query_mask ->support mask
        outputs = []
        for epi in range(batch_size):
            # ======mask==============
            # 1*1*1*512*448*448
            # @GL supp送入vit之前进行mask得到前景
            # supp_fg_fts = [
            #     [self.handle_vit(supp_fts_vgg_out[way, shot, [epi]], fore_mask[way, shot, [epi]], True) for shot in
            #      range(n_shots)]
            #     for way in range(n_ways)]

            # supp_fg_fts = [[self.getFeatures(supp_fts_vgg_out[way, shot, [epi]],
            #                                  fore_mask[way, shot, [epi]])
            #                 for shot in range(n_shots)] for way in range(n_ways)]  # [[ 1 512 ]]

            supp_bg_fts = [[self.getFeatures(supp_fts_vgg_out[way, shot, [epi]],
                                             back_mask[way, shot, [epi]])
                            for shot in range(n_shots)] for way in range(n_ways)]  # [[ 1 512 ]]
            # ======vit==============

            # supp_vit_output = [
            #     [torch.sum(
            #         (
            #         self.vit_model(supp_fg_fts[way][shot][[epi]]).reshape((1, 14, 14, 768)).permute([0, 3, 2, 1])) *
            #         fore_mask[way, shot, [epi]][None, ...], dim=(2, 3)) \
            #      / (fore_mask[way, shot, [epi]][None, ...].sum(dim=(2, 3)) + 1e-5) for shot
            #      in range(n_shots)] for way in range(n_ways)]
            # @GL vit出来之后进行sum  所以是False
            # supp_fg_fts_vit_output = [
            #     [self.handle_vit(
            #         self.vit_model(supp_fg_fts[way][shot][[epi]]).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
            #         fore_mask[way, shot, [epi]], False) for shot
            #         in range(n_shots)] for way in range(n_ways)]
            # F.interpolate(self.getFeatures(supp_fts_vgg_out[way, shot, [epi]],
            #                                fore_mask[way, shot, [epi]])[..., None, None],
            #               size=fore_mask.shape[-2:], mode='bilinear')
            supp_fg_fts_vit_output = [[self.handle_vit(self.vit_model(F.interpolate(
                self.getFeatures(supp_fts_vgg_out[way, shot, [epi]], fore_mask[way, shot, [epi]])[..., None, None],
                size=fore_mask.shape[-2:], mode='bilinear')).reshape((1, 14, 14, 512)).permute([0, 3, 2, 1]),
                                                       fore_mask[way, shot, [epi]])
                                       for shot in range(n_shots)] for way in range(n_ways)]

            # print('supp_vit_output',supp_vit_output[0][0].shape)

            # pred_f = torch.stack(supp_fg_fts, dim=1)
            # pred_b = torch.stack(supp_bg_fts, dim=1)
            # support_resnet_output = img_fts[0]  # [ 1 1024 56 56]
            # query_resnet_output = img_fts[1]  # 1 1024 56 56
            # query and support -> 统一降维，适配vit的输入和输出
            # qry_fts_vgg_out = self.proj(qry_fts_vgg_out.unsqueeze(0))  # 1 768 14 14 经过con2d后的维度
            # query_resnet_output_reshape = query_resnet_output.reshape((1, 768, 196)).permute([0, 2, 1]) # 1 196 768
            # ==========@CZB在经过resnet后和送入VIT之前把support的mask引进去，把所有背景也当作一个类===========
            # =====================调用VIT================================
            # @gl 修改support_resnet_output 为 img_fts  先匹配维度调通 后续再改
            # @CZB support的前景和背景
            # fore_mask = torch.stack([torch.stack(way, dim=0)
            #                          for way in fore_mask], dim=0)
            # back_mask = torch.stack([torch.stack(way, dim=0)
            #                          for way in back_mask], dim=0)
            # support_resnet_output = F.interpolate(support_resnet_output, size=mask.shape[-2:], mode='bilinear')
            # masked_fts = torch.sum(support_resnet_output * mask[None, ...], dim=(2, 3)) \
            #              / (mask[None, ...].sum(dim=(2, 3)) + 1e-5)
            # ************ 为防止GPU，暂时注释 ***********
            # ++ support_vit_output = self.vit_model(support_resnet_output.unsqueeze(0))  # 196 768
            # ++ support_vit_output_reshape = support_vit_output.permute([2, 0, 1]).reshape((-1, 3, 14, 14)) # 划分开,维度与query_resnet_out适配
            # VIT聚类原型的可视化
            # ++ img_grid = vutils.make_grid(support_vit_output_reshape, normalize=True, scale_each=True, nrow=14)
            # ++ writer.add_image(f'feature_maps', img_grid, global_step=0)
            # print("INFO --VIT END.................\n", s_vit_output.shape)
            # ==============计算相似度=================
            # utils.cosine_similarity()
            # @GL squeeze 删掉1 保留197 768
            # @GL  先将三维改成二维  然后将通道维度调整到1维位置
            # ++ cos_distance = F.cosine_similarity(support_vit_output, query_resnet_output_reshape, dim=1)

            # ==============调用与训练模型==============
            # weights_dict = torch.load(args.weights, map_location=device)
            # # 删除不需要的权重
            # del_keys = ['head.weight', 'head.bias'] if model.has_logits \
            #     else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
            # for k in del_keys:
            #     del weights_dict[k]
            # ============================

            # print('fewshot-69: ', img_fts.shape)   # [2, 512, 49, 49]
            # outputs = self.blocks(img_fts)
            # outputs = self.norm(outputs)  #  是否执行 自行决定

            # print('fewshot-69: ', img_fts.shape)  # [2, 1024, 49, 49]

            # for epi in range(batch_size):
            #     ###### 考虑到support image前景和背景 Extract prototype ######
            #     # Extract foreground and background features via masked average pooling
            #     supp_fg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
            #                                      fore_mask[way, shot, [epi]])
            #                     for shot in range(n_shots)] for way in range(n_ways)]
            #
            #     supp_bg_fts = [[self.getFeatures(supp_fts[way, shot, [epi]],
            #                                  back_mask[way, shot, [epi]])
            #                     for shot in range(n_shots)] for way in range(n_ways)]

            ###### Obtain the prototypes，考虑背景原型——后期通过mine背景信息中的novel class提升novel classes的预测效果######
            # 1*768  1*512*448*448
            fg_prototypes, bg_prototype = self.getPrototype(supp_fg_fts_vit_output, supp_bg_fts)
            ###### Compute the distance ######
            prototypes = [bg_prototype, ] + fg_prototypes
            dist = [self.calDist(qry_fts_vgg_out[:, epi], prototype) for prototype in prototypes]
            pred = torch.stack(dist, dim=1)  #
            outputs.append(F.interpolate(pred, size=img_size, mode='bilinear'))
            ###### Prototype alignment loss ######
            if self.config['align'] and self.training:
                align_loss_epi = self.alignLoss(qry_fts_vgg_out[:, epi], pred, supp_fts_vgg_out[:, :, epi],
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
