import os
import sys
import time
import logging
import argparse

import torch
import numpy as np
from tqdm import tqdm
from models.GAT import Net as model

from utils.viz import viz_pred_os
from utils.util import set_seed, get_optimizer
from utils.evaluation import scores, process_seg


parser = argparse.ArgumentParser()
##  path
parser.add_argument('--data_root', type=str, default='ag_dataset')
parser.add_argument('--save_root', type=str, default='save_models')
##  image
parser.add_argument('--crop_size', type=int, default=448)
parser.add_argument('--resize_size', type=int, default=476)
##  dataloader
parser.add_argument('--num_workers', type=int, default=8)
##  train
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--warm_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--show_step', type=int, default=30)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--viz', action='store_true', default=False)

#### test
parser.add_argument("--test_batch_size", type=int, default=1)
parser.add_argument('--test_num_workers', type=int, default=8)

args = parser.parse_args()
torch.cuda.set_device('cuda:' + args.gpu)

time_str = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
args.save_path = os.path.join(args.save_root, time_str)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path, exist_ok=True)
dict_args = vars(args)

str_1 = ""
for key, value in dict_args.items():
    str_1 += key + "=" + str(value) + "\n"

logging.basicConfig(filename='%s/run.log' % args.save_path, level=logging.INFO, format='%(message)s')
logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.info(str_1)

if __name__ == '__main__':
    set_seed(seed=0)

    from data.ego_video_data import TrainData, TestData, AFF_LIST

    args.class_names = AFF_LIST

    trainset = TrainData(data_root=args.data_root, resize_size=args.resize_size,
                         crop_size=args.crop_size)

    TrainLoader = torch.utils.data.DataLoader(dataset=trainset,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=args.num_workers,
                                              pin_memory=True)


    args.test_dir = 'Affordance_Evaluation_Dataset'
    testset = TestData(data_root=args.data_root, crop_size=args.crop_size, test_dir=args.test_dir)
    TestLoader = torch.utils.data.DataLoader(dataset=testset,
                                             batch_size=args.test_batch_size,
                                             shuffle=False,
                                             num_workers=args.test_num_workers,
                                             pin_memory=True)

    model = model().cuda()

    model.train()
    optimizer, scheduler = get_optimizer(model, logger, args)

    best_iou, best_epoch = 0, 0
    curr_biou, curr_niou = 0, 0
    print('Train begining!')

    for epoch in range(args.epochs):
        model.train()
        
        for step, (img, dep, ann) in enumerate(TrainLoader):
            img, dep, ann = img.cuda(), dep.cuda(), ann.cuda().float()

            pred, loss_dict = model(img, dep, label=ann)

            loss = sum(loss_dict.values())

            if loss > 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            if (step + 1) % args.show_step == 0:
                log_str = 'epoch: %d/%d + %d/%d | loss: %.3f | ' % (epoch + 1, args.epochs, step + 1, len(TrainLoader), loss)
                log_str += ' | '.join(['%s: %.3f' % (k, v) for k, v in loss_dict.items()])
                log_str += ' | '
                log_str += 'lr {:.6f}'.format(scheduler.get_last_lr()[0])
                logger.info(log_str)

        scheduler.step()
        model.eval()

        preds, gts = [], []
        iou_class = np.zeros(len(AFF_LIST))
        acc_class = np.zeros(len(AFF_LIST))
        avg_count = np.zeros(len(AFF_LIST))

        for step, (image, dep, ann_test, obj_name, _, _) in enumerate(tqdm(TestLoader)):
            ann_test = ann_test[:, 1:].cuda().float()

            with torch.no_grad():
                pred_test = model(image.cuda(), dep.cuda())
            pred_min, pred_max = pred_test.min(), pred_test.max()
            pred_norm = (pred_test - pred_min) / (pred_max - pred_min + 1e-10)
            pred_, ann_ = process_seg(pred_norm, ann_test, td=0.7)
            
            preds.append(pred_)
            gts.append(ann_)

            if args.viz:
                viz_pred_os(args, image, pred_norm, ann_test, AFF_LIST, step, 'test' + str(epoch + 1))

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

        logger.info(
            'mIoU: {:.2f} | F1: {:.2f} | mAcc: {:.2f} | best_iou: {}_{:.2f}'.format(
                mIoU, f1_score, mAcc, best_epoch, best_iou))

        if mIoU > best_iou:
            best_iou = mIoU
            best_epoch = (epoch + 1)
            
            model_name = 'best_' + str(best_epoch) + \
                            '_iou_' + str(round(best_iou, 2)) + '.pth'

            torch.save({'epoch': best_epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(args.save_path, model_name))