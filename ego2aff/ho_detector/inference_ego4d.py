import torch
import numpy as np
import cv2
import pickle
from os.path import join as opj
from tqdm import tqdm
import os

from model.faster_rcnn.resnet import resnet
from model.utils.blob import im_list_to_blob
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import bbox_transform_inv, clip_boxes
from model.roi_layers import nms
from model.utils.net_utils import save_net, load_net, vis_detections, vis_detections_PIL, vis_detections_filtered_objects_PIL, vis_detections_filtered_objects # (1) here add a function to viz
from ho_types import FrameDetections, HandDetection, ObjectDetection

from src.convert_raw_to_releasable_detections import Converter


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

if __name__ == '__main__':

  model_dir = '/home/gen/video_affordance/ego2aff/ho_detector/models/res101_handobj_100K/pascal_voc/faster_rcnn_1_8_132028.pth'
  clip_path = '/home/gen/Ego4d/data/v2/clips/'
  clip_uids_path = './ego4d_interested_clip_uids_1.pkl'
  save_path = '/home/gen/Ego4d/data/v2/clips_ho_detections/'
  cfg.CUDA = True 
  cfg.class_agnostic = False
  cfg_file = 'cfgs/res101.yml'
  cfg_from_file(cfg_file)
  thresh_hand = 0.1
  thresh_obj = 0.01

  with open(clip_uids_path, 'rb') as f:
    clip_uids = pickle.load(f)


  pascal_classes = np.asarray(['__background__', 'targetobject', 'hand']) 
  fasterRCNN = resnet(pascal_classes, 101, pretrained=False, class_agnostic=cfg.class_agnostic)
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

  for clip_uid in tqdm(clip_uids):
    frame_num = 0
    clip_dets = []
    clip_file = opj(clip_path, clip_uid + '.mp4')
    pkl_save_path = opj(save_path, clip_uid + '.pkl')
    if os.path.exists(pkl_save_path):
      continue
  
    cap = cv2.VideoCapture(clip_file)
    
    if not cap.isOpened():
      print("Error: Could not open video.")
      continue
    
    v_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    v_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    
    print(f"Processing video: {clip_uid}")
    
    while True:
      ret, frame = cap.read()
      
      if not ret:
          break
      
      blobs, im_scales = _get_image_blob(frame)
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
        box_info.resize_(1, 1, 5).zero_() 

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

      frame_det = FrameDetections.from_detections(video_id=clip_uid, frame_number=frame_num, hand_detections=hand_dets, object_detections=obj_dets)        
      clip_dets.append(frame_det)
      # im2show = vis_detections_filtered_objects_PIL(frame, obj_dets, hand_dets, thresh_hand, thresh_obj)
      # im2show.save(f'./demo/out_{clip_uid}_{frame_num}.png')
      frame_num += 1
      # if frame_num > 10:
        # cap.release()
        # cv2.destroyAllWindows()   
        # break
    
    cap.release()
    cv2.destroyAllWindows()   
    converter = Converter(v_height, v_width)
    releasable_video_annotations = converter.convert_video_annotations(clip_dets)
    with open(pkl_save_path, "wb") as f:
      pickle.dump(
          [det.to_protobuf().SerializeToString() for det in releasable_video_annotations], f, 
      )
  