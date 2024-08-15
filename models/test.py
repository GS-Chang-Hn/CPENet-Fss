"""
@Create Time : 2022/10/18 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
import torch
from swin_transformer_pytorch import SwinTransformer

net = SwinTransformer(
    hidden_dim=96,
    layers=(2, 2, 6, 2),
    heads=(3, 6, 12, 24),
    channels=3,
    num_classes=3,
    head_dim=32,
    window_size=7,
    downscaling_factors=(4, 2, 2, 2),
    relative_pos_embedding=True
)
print(net)
dummy_x = torch.randn(1, 3, 224, 224)
logits = net(dummy_x)  # (1,3)
print(net)
print(logits)