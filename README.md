# CPENet-fss
This is the implementation of our paper "Multi-Prototype Collaborative Perception Enhancement Network for Few-shot Semantic Segmentation" that has been submitted to The Visual Computer.
# Description
Few-shot semantic segmentation (FSS) aims at learning to segment an unseen object from the query image with only a few densely annotated support images. Most existing methods rely on capturing the similarity between support and query features to facilitate segmentation for the unseen object. However, the segmentation performance of these methods remains an open problem because of the diversity between objects in the support and query images. To alleviate this problem, we propose a multi-prototype collaborative perception enhancement network. More specifically, we develop the feature recombination module to recombine the support foreground features. Second, the superpixel-guided multi-prototype generation strategy is employed to generate multiple prototypes in the support foreground and the whole query feature by aggregating similar semantic information. Meanwhile, the Vision Transformer (ViT) is used to generate background prototypes from the support background features. Third, we devise a task-aware attention module to establish the interaction by exploring correspondence relations between support and query prototypes. Finally, a non-parametric metric is used to match the features and prototypes. Extensive experiments on two benchmarks, PASCAL-5$^{i}$ and COCO-20$^{i}$, demonstrate that the proposed model has superior segmentation performance compared to baseline methods and is competitive with previous FSS methods. 
## Environment setup
```
Python 3.7 +
torch 1.7.0
torchvision 0.8.0
scipy 1.7.3
tqdm 4.64.0
```
# Please download the following datasets: 
```
PASCAL-5i is based on the PASCAL VOC 2012 and SBD where the val images should be excluded from the list of training samples.
Images are available at: http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
annotations: https://drive.google.com/file/d/1ikrDlsai5QSf2GiSUR3f8PZUzyTubcuF/view?usp=sharing
This work is built on:
OSLSM: https://github.com/lzzcd001/OSLSM
PANet: https://github.com/kaixin96/PANet
Many thanks to their greak work!
```
