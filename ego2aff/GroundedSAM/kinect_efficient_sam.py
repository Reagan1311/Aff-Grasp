import cv2
import numpy as np
import supervision as sv

# from pyk4a import PyK4A

import torch
import torchvision
from torchvision.transforms import ToTensor

from groundingdino.util.inference import Model

import sys
from os import path
sys.path.append(path.dirname(__file__))

from EfficientSAM.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits


def efficient_sam_box_prompt_segment(image, pts_sampled, model):
    bbox = torch.reshape(torch.tensor(pts_sampled), [1, 1, 2, 2])
    bbox_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = model(
        img_tensor[None, ...].cuda(),
        bbox.cuda(),
        bbox_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# GroundingDINO config and checkpoint
GROUNDING_DINO_CONFIG_PATH = "GroundedSAM/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
GROUNDING_DINO_CHECKPOINT_PATH = "GroundedSAM/groundingdino_swint_ogc.pth"
SEG_MODEL = 'vits'

# Building GroundingDINO inference model
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

if SEG_MODEL == 'vits':
    efficientsam = build_efficient_sam_vits().cuda()
elif SEG_MODEL == 'vitt':
    efficientsam = build_efficient_sam_vitt().cuda()
else:
    raise ValueError(f'The {SEG_MODEL} is not supported in the EfficientSAM')

# Building MobileSAM predictor
# EFFICIENT_SAM_CHECHPOINT_PATH = "./EfficientSAM/efficientsam_s_gpu.jit"
# EFFICIENT_SAM_CHECHPOINT_PATH = "./EfficientSAM/efficient_sam_vits.pt"

efficientsam = build_efficient_sam_vits().cuda()

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8

# load azure kinect camera
def ov_detect_seg(rgb, classes, SAMseg=True, box_th=BOX_THRESHOLD, text_th=TEXT_THRESHOLD, nms_th=NMS_THRESHOLD):
    # detect objects
    detections = grounding_dino_model.predict_with_classes(
        image=rgb,
        classes=classes,
        box_threshold=box_th,
        text_threshold=text_th
    )

    # annotate image with detections
    # box_annotator = sv.BoxAnnotator()
    # labels = [
    #     f"{classes[class_id]} {confidence:0.2f}" 
    #     for _, _, confidence, class_id, _, _ 
    #     in detections]
    # annotated_frame = box_annotator.annotate(scene=rgb.copy(), detections=detections, labels=labels)

    # save the annotated grounding dino image
    # cv2.imwrite("EfficientSAM/LightHQSAM/groundingdino_annotated_image.jpg", annotated_frame)

    # NMS post process
    # print(f"Before NMS: {len(detections.xyxy)} boxes")
    nms_idx = torchvision.ops.nms(
        torch.from_numpy(detections.xyxy), 
        torch.from_numpy(detections.confidence), 
        nms_th
    ).numpy().tolist()

    detections.xyxy = detections.xyxy[nms_idx]
    detections.confidence = detections.confidence[nms_idx]
    detections.class_id = detections.class_id[nms_idx]

    # print(f"After NMS: {len(detections.xyxy)} boxes")

    # collect segment results from EfficientSAM
    if SAMseg:
        result_masks = []
        for box in detections.xyxy:
            mask = efficient_sam_box_prompt_segment(rgb, box, efficientsam)
            result_masks.append(mask)

        detections.mask = np.array(result_masks)

        # annotate image with detections
        # box_annotator = sv.BoxAnnotator()
        # mask_annotator = sv.MaskAnnotator()
        # labels = [
        #     f"{classes[class_id]} {confidence:0.2f}" 
        #     for _, _, confidence, class_id, _, _ 
        #     in detections]
        # annotated_image = mask_annotator.annotate(scene=rgb.copy(), detections=detections)
        # annotated_image = box_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # save the annotated grounded-sam image
        # if len(detections.xyxy) > 0:
            # cv2.imwrite(f"./figs/{detections.xyxy[0]}.jpg", annotated_image)

    return detections


def efficient_sam_box_prompt(image, prompt, prompt_labels=None, box=True):
    if box:
        prompt = torch.reshape(torch.tensor(prompt), [1, 1, 2, 2])
        prompt_labels = torch.reshape(torch.tensor([2, 3]), [1, 1, 2])
    else:
        if prompt_labels is None:
            ValueError('For point prompts, label should be provided.')
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = ToTensor()(image)

    predicted_logits, predicted_iou = efficientsam(
        img_tensor[None, ...].cuda(),
        prompt.cuda(),
        prompt_labels.cuda(),
    )
    predicted_logits = predicted_logits.cpu()
    all_masks = torch.ge(torch.sigmoid(predicted_logits[0, 0, :, :, :]), 0.5).numpy()
    predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()

    max_predicted_iou = -1
    selected_mask_using_predicted_iou = None
    for m in range(all_masks.shape[0]):
        curr_predicted_iou = predicted_iou[m]
        if (
            curr_predicted_iou > max_predicted_iou
            or selected_mask_using_predicted_iou is None
        ):
            max_predicted_iou = curr_predicted_iou
            selected_mask_using_predicted_iou = all_masks[m]
    return selected_mask_using_predicted_iou
