"""
@Create Time : 2022/4/29 
@Authors     : Allen_Chang
@Description : 请在这里添加功能描述
@Modif. List : 请在这里添加修改记录
"""
import os

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from torchvision.transforms import Compose


trans = Compose([transforms.ToTensor()])
url = "D:\\changzb\\sapace\\LZU_WORK_SPACE\\Cycle_Cross_Guided\\data\\Pascal\\VOCdevkit\\VOC2012\\SegmentationClassAug"
label = Image.open(os.path.join(url, f'2007_000515.png')).convert("RGB")
print(label)
label = trans(label)
torch.set_printoptions(profile="full")
print(type(label))
print(label)
