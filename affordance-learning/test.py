import os
import sys
import logging
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm
from models.GAT import Net as model

from utils.viz import viz_pred_os
from utils.util import set_seed
from utils.evaluation import scores, process_seg


parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='ag_dataset')
parser.add_argument('--model_file', type=str, default=None)
parser.add_argument('--save_root', type=str, default='save_models')
##  image
parser.add_argument('--base', action='store_true', default=False)
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=476)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0') 
parser.add_argument('--viz', action='store_true', default=False)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)

args.save_path = os.path.dirname(args.model_file)
os.makedirs(os.path.join(args.save_path, 'viz_seg'), exist_ok=True)
logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))

PALETTE = [[128, 128, 128], [129, 127, 38], [120, 69, 125], [53, 125, 34], 
           [0, 11, 123], [118, 20, 12], [122, 81, 25], [241, 134, 51]]

def colorize_seg(sim, td=0.7):
    num_aff, w, h = sim.shape
    sim = sim.flatten(1)
    max_idx = torch.argmax(sim, dim=0)
    bg_idx = torch.all((sim < td), dim=0)
    out = torch.zeros(sim.shape[-1], 3).long()
    for i in range(num_aff):
        out[max_idx == i] = torch.LongTensor(PALETTE[i])
    out[bg_idx] = torch.LongTensor([0, 0 ,0])
    out = out.view(w, h, 3)
    out = out.cpu().numpy().astype(np.uint8)
    return out


if __name__ == '__main__':
    set_seed(seed=0)

    from data.ego_video_data import TestData, AFF_LIST

    args.class_names = AFF_LIST

    args.test_dir = 'Affordance_Evaluation_Dataset'
    testset = TestData(data_root=args.data_root, crop_size=args.crop_size, test_dir=args.test_dir)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model().cuda()
    model.eval()
    assert os.path.exists(args.model_file), "Please provide the correct model file for testing"
    state_dict = torch.load(args.model_file)['model_state_dict']
    model.load_state_dict(state_dict, strict=False)

    print('Evaluation begining!')

    preds, gts = [], []
    if not args.base:
        iou_class = np.zeros(len(AFF_LIST))
        acc_class = np.zeros(len(AFF_LIST))
        avg_count = np.zeros(len(AFF_LIST))
    else:
        b_iou_class = np.zeros(len(AFF_LIST))
        n_iou_class = np.zeros(len(AFF_LIST))
        b_acc_class = np.zeros(len(AFF_LIST))
        n_acc_class = np.zeros(len(AFF_LIST))
        b_avg_count = np.zeros(len(AFF_LIST))
        n_avg_count = np.zeros(len(AFF_LIST))


    for step, (image, dep, ann_test, obj_name, ori_size, img_name) in enumerate(tqdm(TestLoader)):
        ann_test = ann_test[:, 1:].cuda().float()

        with torch.no_grad():
            pred_test = model(image.cuda(), dep.cuda())
        pred_min, pred_max = pred_test.min(), pred_test.max()
        pred_norm = (pred_test - pred_min) / (pred_max - pred_min + 1e-10)
        pred_, ann_ = process_seg(pred_norm, ann_test, td=0.8)
        
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        img = image[0] * std + mean
        img = img.detach().cpu().numpy() * 255
        img = img.transpose(1, 2, 0).astype(np.uint8)
        pred_bi = colorize_seg(pred_norm.squeeze())
        gt_bi = colorize_seg(ann_test.squeeze())
        
        ori_size = [int(ori_size[0]), int(ori_size[1])]
        img = cv2.resize(img, ori_size)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pred_bi = cv2.resize(pred_bi, ori_size, interpolation=cv2.INTER_NEAREST)
        pred_bi = cv2.cvtColor(pred_bi, cv2.COLOR_BGR2RGB)
        gt_bi = cv2.resize(gt_bi, ori_size, interpolation=cv2.INTER_NEAREST)
        gt_bi = cv2.cvtColor(gt_bi, cv2.COLOR_BGR2RGB)
        
        preds.append(pred_)
        gts.append(ann_)

        if args.viz:
            viz_pred_os(args, image, pred_norm, ann_test, AFF_LIST, step, 'test')

    metrics = scores(gts, preds, n_class=len(AFF_LIST) + 1, ignore_zero=True)
    mIoU = metrics['Mean IoU'] * 100
    mAcc = metrics['Mean Accuracy'] * 100
    iou_class = metrics['Class IoU']
    f1_score = metrics['F1'] * 100

    iou_class_out = ''
    for i, aff in enumerate(['bg'] + args.class_names):
        iou_class_out += '{}: {:.2f} | '.format(aff, iou_class[i])
    iou_class_out = iou_class_out[:-3]
    logger.info(iou_class_out)
    logger.info('mIoU: {:.2f} | F1: {:.2f} | mAcc: {:.2f}'.format( 
        mIoU, f1_score, mAcc))



    
    
