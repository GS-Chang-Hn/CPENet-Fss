"""Training Script"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.backends.cudnn as cudnn
from torchvision.transforms import Compose
from models.fewshot import FewShotSeg
from dataloaders.customized import voc_fewshot, coco_fewshot
from dataloaders.transforms import RandomMirror, Resize, ToTensorNormalize
from util.utils import set_seed, CLASS_LABELS
from config import ex
from tensorboardX import SummaryWriter
from util import utils
from models.swin_transformer import swin_tiny_patch4_window7_224 as create_model

writer = SummaryWriter('./Results/scalar_example')

@ex.automain
def main(_run, _config, _log):
    if _run.observers:
        os.makedirs(f'{_run.observers[0].dir}/snapshots', exist_ok=True)
        for source_file, _ in _run.experiment_info['sources']:
            os.makedirs(os.path.dirname(f'{_run.observers[0].dir}/source/{source_file}'),
                        exist_ok=True)
            _run.observers[0].save_file(source_file, f'source/{source_file}')
        shutil.rmtree(f'{_run.observers[0].basedir}/_sources')

    set_seed(_config['seed'])
    cudnn.enabled = True
    cudnn.benchmark = True
    torch.cuda.set_device(device=_config['gpu_id'])
    torch.set_num_threads(1)



    _log.info('###### Create model ######')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    swin_transformer = create_model(num_classes=0).to(device)
    swin_weight_path = "pretrained_model/swin_tiny_patch4_window7_224.pth"
    swin_weights_dict = torch.load(swin_weight_path, map_location=device)["model"]
    # 删除有关分类类别的权重
    for k in list(swin_weights_dict.keys()):
        if "head" in k:
            del swin_weights_dict[k]
    swin_transformer.load_state_dict(swin_weights_dict, strict=False)

    model = FewShotSeg(pretrained_path=_config['path']['init_path'], cfg=_config['model'],swin_model=swin_transformer)
    model = nn.DataParallel(model.cuda(), device_ids=[_config['gpu_id'], ])
    model.train()


    _log.info('###### Load data ######')
    data_name = _config['dataset']
    if data_name == 'VOC':
        make_data = voc_fewshot
    elif data_name == 'COCO':
        make_data = coco_fewshot
    else:
        raise ValueError('Wrong config for dataset!')
    labels = CLASS_LABELS[data_name][_config['label_sets']]
    transforms = Compose([Resize(size=_config['input_size']),
                          RandomMirror()])
    dataset = make_data(
        base_dir=_config['path'][data_name]['data_dir'],  # ./data/Pascal/VOCdevkit/VOC2012/
        split=_config['path'][data_name]['data_split'],  # trainaug
        transforms=transforms,
        to_tensor=ToTensorNormalize(),
        labels=labels,  # {6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20}
        max_iters=_config['n_steps'] * _config['batch_size'],  # 50000 * 1
        n_ways=_config['task']['n_ways'],  # 1
        n_shots=_config['task']['n_shots'],  # 1
        n_queries=_config['task']['n_queries']  # 1
    )

    trainloader = DataLoader(
        dataset,
        batch_size=_config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=True,

        drop_last=True
    )

    _log.info('###### Set optimizer ######')
    optimizer = torch.optim.SGD(model.parameters(), **_config['optim'])
    # optimizer = torch.optim.Adam(model.parameters(), **_config['optim'])
    scheduler = MultiStepLR(optimizer, milestones=_config['lr_milestones'], gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=_config['ignore_label'])
    i_iter = 0
    log_loss = {'loss': 0, 'align_loss': 0}
    _log.info('###### Training ######')
    # 双向指导，超像素和
    for i_iter, sample_batched in enumerate(trainloader):

        # slic_shape_flag = sample_batched['slic_shape_flag'][0][0].item()
        # if slic_shape_flag:
        #     file_name = sample_batched['file_name'][0][0][0]
        #     print(file_name+"出错已跳过训练！")
        #     continue
        # Prepare input
        support_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_images']]
        # print(sample_batched['support_images'][0][0].shape)  # [1, 3, 417, 417]  1 3 390 390
        support_slic_images = [[shot.cuda() for shot in way]
                          for way in sample_batched['support_slic_images']]

        support_fg_mask = [[shot[f'fg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]
        support_bg_mask = [[shot[f'bg_mask'].float().cuda() for shot in way]
                           for way in sample_batched['support_mask']]

        query_images = [query_image.cuda()
                        for query_image in sample_batched['query_images']]
        query_slic_images = [query_slic_image.cuda()
                        for query_slic_image in sample_batched['query_slic_images']]
        query_labels = torch.cat(
            [query_label.long().cuda() for query_label in sample_batched['query_labels']], dim=0)

        # Forward and Backward
        optimizer.zero_grad()
        support_images_id = sample_batched['support_images_id']
        query_images_id = sample_batched['query_images_id']

        # 原图
        query_pred, align_loss = model(support_images, support_slic_images, support_fg_mask, support_bg_mask,
                                       query_images, query_slic_images, support_images_id, query_images_id)
        # metric module
        query_loss = criterion(query_pred, query_labels)
        loss = query_loss + align_loss * _config['align_loss_scaler']
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Log loss
        query_loss = query_loss.detach().data.cpu().numpy()
        align_loss = align_loss.detach().data.cpu().numpy() if align_loss != 0 else 0
        _run.log_scalar('loss', query_loss)
        _run.log_scalar('align_loss', align_loss)
        log_loss['loss'] += query_loss
        log_loss['align_loss'] += align_loss
        # 保存loss到log 可视化
        writer.add_scalar('Support-Query', query_loss, global_step=i_iter)
        writer.add_scalar('Query-Support', align_loss, global_step=i_iter)
        # print loss and take snapshots
        if (i_iter + 1) % _config['print_interval'] == 0:
            loss = log_loss['loss'] / (i_iter + 1)
            align_loss = log_loss['align_loss'] / (i_iter + 1)
            print(f'step {i_iter+1}: loss: {loss}, align_loss: {align_loss}')

        if (i_iter + 1) % _config['save_pred_every'] == 0:
            _log.info('###### Taking snapshot ######')
            torch.save(model.state_dict(),
                       os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))

    _log.info('###### Saving final model ######')
    torch.save(model.state_dict(),
               os.path.join(f'{_run.observers[0].dir}/snapshots', f'{i_iter + 1}.pth'))
