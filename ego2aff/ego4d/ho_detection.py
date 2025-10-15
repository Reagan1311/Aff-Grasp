import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from PIL import Image
import numpy as np
import cv2

import sys
sys.path.append('../ho_detector')
from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms


def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)
  
  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


model_dir = '/home/gen/video_affordance/ego2aff/ho_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth'
cfg.CUDA = True 
cfg.class_agnostic = False
cfg_file = '/home/gen/video_affordance/ego2aff/ho_detector/cfgs/res101.yml'
cfg_from_file(cfg_file)
thresh_hand = 0.1
thresh_obj = 0.01

pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=False)
fasterRCNN.create_architecture()
checkpoint = torch.load(model_dir)
fasterRCNN.load_state_dict(checkpoint['model'])
if 'pooling_mode' in checkpoint.keys():
    cfg.POOLING_MODE = checkpoint['pooling_mode']

im_data = torch.FloatTensor(1).cuda()
im_info = torch.FloatTensor(1).cuda()
num_boxes = torch.LongTensor(1).cuda()
gt_boxes = torch.FloatTensor(1).cuda()
box_info = torch.FloatTensor(1).cuda()

fasterRCNN.cuda().eval()

def ho_detect(img):
    w, h = img.shape[:2]
    blobs, im_scales = _get_image_blob(img)
    im_blob = blobs
    im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
    im_data_pt = torch.from_numpy(im_blob)
    im_data_pt = im_data_pt.permute(0, 3, 1, 2)
    im_info_pt = torch.from_numpy(im_info_np)
    with torch.no_grad():
        im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
        im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
        gt_boxes.resize_(1, 1, 5).zero_()
        num_boxes.resize_(1).zero_()
        box_info.resize_(1, 1, 5).zero_() \
        
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label, loss_list = fasterRCNN(im_data, im_info, gt_boxes, num_boxes, box_info)

        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]

        # extact predicted params
        contact_vector = loss_list[0][0] # hand contact state info
        offset_vector = loss_list[1][0].detach() # offset vector (factored into a unit vector and a magnitude)
        lr_vector = loss_list[2][0].detach() # hand side info (left/right)

        # get hand contact 
        _, contact_indices = torch.max(contact_vector, 2)
        contact_indices = contact_indices.squeeze(0).unsqueeze(-1).float()

        # get hand side 
        lr = torch.sigmoid(lr_vector) > 0.5
        lr = lr.squeeze(0).float()

        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
                if cfg.class_agnostic:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4)
                else:
                    box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                                + torch.FloatTensor(cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                    box_deltas = box_deltas.view(1, -1, 4 * len(pascal_classes))

            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        pred_boxes /= im_scales[0]

        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()

        obj_dets, hand_dets = None, None
        for j in range(1, len(pascal_classes)):
            # inds = torch.nonzero(scores[:,j] > thresh).view(-1)
            if pascal_classes[j] == 'hand':
                inds = torch.nonzero(scores[:,j]>thresh_hand).view(-1)
            elif pascal_classes[j] == 'targetobject':
                inds = torch.nonzero(scores[:,j]>thresh_obj).view(-1)

            # if there is det
            if inds.numel() > 0:
                cls_scores = scores[:,j][inds]
                _, order = torch.sort(cls_scores, 0, True)
                if cfg.class_agnostic:
                    cls_boxes = pred_boxes[inds, :]
                else:
                    cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
            
                cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1), contact_indices[inds], offset_vector.squeeze(0)[inds], lr[inds]), 1)
                cls_dets = cls_dets[order]
                keep = nms(cls_boxes[order, :], cls_scores[order], cfg.TEST.NMS)
                cls_dets = cls_dets[keep.view(-1).long()]
                if pascal_classes[j] == 'targetobject':
                    obj_dets = cls_dets.cpu().numpy()
                if pascal_classes[j] == 'hand':
                    hand_dets = cls_dets.cpu().numpy()        
        
        return obj_dets, hand_dets
    


def get_center(det):
    center_x = (det[0] + det[2]) / 2
    center_y = (det[1] + det[3]) / 2
    return (center_x, center_y)


def get_hand_object_interactions(hand_dets, obj_dets, object_threshold: float = 0, hand_threshold: float = 0):
    """Match the hands to objects based on the hand offset vector that the model
    uses to predict the location of the interacted object.

    Args:
        object_threshold: Object score threshold above which to consider objects
            for matching
        hand_threshold: Hand score threshold above which to consider hands for
            matching.

    Returns:
        A dictionary mapping hand detections to objects by indices
    """
    interactions = dict()
    object_idxs = [
        i for i, obj_det in obj_dets if obj_det[4] >= object_threshold
    ]
    object_centers = np.array(
        [get_center(obj_dets[object_id]) for object_id in object_idxs]
    )
    for hand_idx, hand_det in enumerate(hand_dets):
        if (
            hand_det[5] == HandState.NO_CONTACT.value
            or hand_det[4] <= hand_threshold
        ):
            continue
        estimated_object_position = (
            np.array(get_center(hand_det)) +
            np.array(hand_det.object_offset.coord)
        )
        distances = ((object_centers - estimated_object_position) ** 2).sum(
                axis=-1)
        interactions[hand_idx] = object_idxs[cast(int, np.argmin(distances))]
    return interactions


def locate_active_side(hand_dets, obj_dets, hand_threshold=0.1, obj_threshold=0.1):
    if len(hand_dets) == 1:
        return hand_dets[0]
    else:
        hand_counter = {"LEFT": 0, "RIGHT": 0}

        hands = [hand for hand in hand_dets if hand[4] >= hand_threshold]
        objs = [obj for obj in hand_dets if obj[4] >= obj_threshold]

        if len(hands) > 0 and len(objs) > 0:
            hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                                    hand_threshold=hand_threshold)
            for hand_idx, object_idx in hand_object_idx_correspondences.items():
                hand_bbox = np.array(annot.hands[hand_idx].bbox.coords_int).reshape(-1)
                obj_bbox = np.array(annot.objects[object_idx].bbox.coords_int).reshape(-1)
                xA, yA, xB, yB, iou = bbox_inter(hand_bbox, obj_bbox)
                if iou > 0:
                    hand_side = annot.hands[hand_idx].side.name
                    if annot.hands[hand_idx].state.value == HandState.PORTABLE_OBJECT.value:
                        hand_counter[hand_side] += 1
                    elif annot.hands[hand_idx].state.value == HandState.STATIONARY_OBJECT.value:
                        hand_counter[hand_side] += 0.5
        if hand_counter["LEFT"] == hand_counter["RIGHT"]:
            return "RIGHT"
        else:
            return max(hand_counter, key=hand_counter.get)