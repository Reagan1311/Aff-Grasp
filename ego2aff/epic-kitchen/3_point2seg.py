import cv2
import torch
import numpy as np
import os
import glob
from PIL import Image
import pickle
import pandas as pd
import argparse
from tqdm import tqdm
from preprocess.dataset_util import FrameDetections, load_ho_annot, load_img, get_mask
from preprocess.affordance_util import *
from model_utils.extractor_dino import ViTExtractor
from utils import box_expansion, find_contour_draw

from GroundedSAM.EfficientSAM.kinect_efficient_sam import ov_detect_seg, efficient_sam_box_prompt
# from segment_anything import sam_model_registry, SamPredictor
from segment_anything_hq import sam_model_registry, SamPredictor


filter_nouns = ["spoon", "knife", "pan", "cup", "fork", "spatula", "scissors", "ladle"]
filter_corr_nouns = ['knife', 'scissors', 'spoon', 'spatula']
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/EPIC-KITCHENS", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./outputs_kitchen", type=str, help="generated results save path")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('-p', '--participant_id', nargs='*', default=None, type=str)
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path
    
    train_save = os.path.join(save_path, 'train_data')
    os.makedirs(train_save, exist_ok=True)

    if args.participant_id is None:
        par_list = glob.glob(os.path.join(args.dataset_path, 'P*'))
        par_list.sort()
    else:
        par_list = [os.path.join(args.dataset_path, p) for p in args.participant_id ]

    # build SAM & DINO model
    # dinov2 = ViTExtractor('dinov2_vitb14', stride=14, device='cuda')
    dinov2 = torch.hub.load("mhamilton723/FeatUp", 'dinov2', use_norm=True)
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_checkpoint = "sam_hq_vit_h.pth"
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).cuda()
    
    for par_path in par_list:
        par_id = par_path.split('/')[-1]
        video_id_path = os.path.join(par_path, "rgb_frames")
        video_id_list = os.listdir(video_id_path)
        video_id_list.sort()

        for video_id in tqdm(video_id_list):
            # if video_id == 'P01_01':
                print(f"======================= Checking video: {video_id} =======================")
                sub_save_path = os.path.join(save_path, par_id, video_id)
                save_pickle_path = os.path.join(sub_save_path, f"{video_id}_multi_affpts.pkl")
                if not os.path.exists(save_pickle_path):
                    continue
                
                with open(save_pickle_path, "rb") as f:
                    obs_frame_affpts = pickle.load(f)
                
                frames_path = os.path.join(video_id_path, video_id)
                for of in obs_frame_affpts:
                    noun = of['noun']
                    if noun not in filter_nouns:
                        continue
                    obs_frame_idx = of['obs_frame_idx']  # contact / obs frames idxs
                    pos_pt, neg_pt = of['prompt_pts']
                    pos_pt = pos_pt.reshape(-1, 2)
                    obj_box = of['prompt_box']
                    
                    obs_frame = load_img(frames_path, obs_frame_idx)
                    
                    #################################
                    # ex_obj_box = box_expansion(obj_box, size=obs_frame.shape[:2], ratio=0.05)
                    ex_obj_box = obj_box
                    
                    if noun in filter_corr_nouns:
                        support_path = './support_pts/epic_kitchens'
                        neg_pt = []
                        neg_pts = neg_pt_featup(noun, obs_frame, ex_obj_box, dinov2, support_path)
                        for n_p in neg_pts:
                            obj_w, obj_h = ex_obj_box[2] - ex_obj_box[0], ex_obj_box[3] - ex_obj_box[1]
                            p = [ex_obj_box[0] + n_p[0] * obj_w, ex_obj_box[1] + n_p[1] * obj_h]
                            n_p = np.array(p).round().astype(int).reshape(1, -1)
                            neg_pt.append(n_p)
                        neg_pt = np.vstack(neg_pt)
                    # print(ex_obj_box[:2], neg_pt)

                    #################################
                    # ex_obj_box = box_expansion(obj_box, obs_frame.shape[:2], ratio=0.05)
                    # x1, y1, x2, y2 = ex_obj_box
                    # cropped_frame = obs_frame[y1:y2, x1:x2]
                    # cropped_frame = cv2.resize(cropped_frame, (224, 224))

                    # pos_pt = ((pos_pt - (x1, y1)) * 224 / (x2 - x1, y2 - y1)).round().astype(int)
                    # neg_pt = ((neg_pt - (x1, y1)) * 224 / (x2 - x1, y2 - y1)).round().astype(int)

                    # import pdb;pdb.set_trace()
                    # pos_pt, neg_pt = [pos_pt], [neg_pt]
                    prompt_pts = np.vstack([pos_pt, neg_pt])
                    grasp_labels = np.hstack(([1] * len(pos_pt), [0] * len(neg_pt)))
                    fun_labels = np.hstack(([0] * len(pos_pt), [1] * len(neg_pt)))
                    # import pdb; pdb.set_trace()
                    # grasp_labels = np.array([1, 0])
                    # fun_labels = np.array([0, 1])
                    # grasp_mask = efficient_sam_box_prompt(obs_frame, prompt_pts, grasp_labels, box=False)
                    # fun_mask = efficient_sam_box_prompt(obs_frame, prompt_pts, fun_labels, box=False)

                    predictor = SamPredictor(sam)
                    predictor.set_image(obs_frame)
                    
                    grasp_mask, scores, logits = predictor.predict(
                        point_coords=prompt_pts,
                        point_labels=grasp_labels,
                        # box = obj_box,
                        multimask_output=False,
                        hq_token_only = False 
                        )
                    grasp_mask = grasp_mask[0].astype(np.uint8)
                    # grasp_mask = masks[np.argmax(scores), :, :]  # Choose the model's best mask
                    fun_mask, scores, logits = predictor.predict(
                        point_coords=prompt_pts,
                        point_labels=fun_labels,
                        # box = obj_box,
                        multimask_output=False,
                        hq_token_only = False 
                        )
                    fun_mask = fun_mask[0].astype(np.uint8)
                    # fun_mask = masks[np.argmax(scores), :, :]  # Choose the model's best mask
                    
                    if len(grasp_mask[grasp_mask == 1]) == 0 or len(fun_mask[fun_mask == 1]) == 0:
                        continue
                    
                    # grasp_mask[grasp_mask == True] = 255
                    # fun_mask[fun_mask == True] = 255

                    # box_mask = efficient_sam_box_prompt(obs_frame, prompt_pts, pt_labels)

                    out = obs_frame.copy()
                    out = cv2.rectangle(out, tuple(obj_box[:2]), 
                                            tuple(obj_box[2:]), color=(0, 255, 255), thickness=1)
                    
                    _, out = find_contour_draw(out, grasp_mask, color=(255, 0, 255))
                    _, out = find_contour_draw(out, fun_mask, color=(0, 255, 0))
                    
                    for p in pos_pt:
                        cv2.circle(out,p,1,(255,0,255), 2)
                    for n in neg_pt:
                        cv2.circle(out,n,1,(0,255,0), 2)
                    
                    # img_name = os.path.join(sub_save_path, f'{noun}-{obs_frame_idx}-affseg.jpg')
                    # img_name = os.path.join('outputs/cropped_raw', f'{noun}-{video_id}-{obs_frame_idx}-affseg.jpg')
                    if args.vis:
                        # img_name = os.path.join('outputs/', f'{noun}-{video_id}-{obs_frame_idx}-affseg.jpg')
                        img_name = f'{noun}-{video_id}-{obs_frame_idx}.jpg'
                        img_name = os.path.join(train_save, img_name)
                        cv2.imwrite(img_name, out)
                    # else:                        
                    temp_label = np.zeros(obs_frame.shape[:2], dtype=np.uint8)
                    temp_label[grasp_mask == 1] = 128
                    temp_label[fun_mask == 1] = 255
                    temp_label = temp_label[ex_obj_box[1]:ex_obj_box[3], ex_obj_box[0]:ex_obj_box[2]]

                    cropped_out = obs_frame[ex_obj_box[1]:ex_obj_box[3], ex_obj_box[0]:ex_obj_box[2]]
                    img_name = os.path.join(train_save, f'{noun}-{video_id}-{obs_frame_idx}-img.jpg')
                    label_name = os.path.join(train_save, f'{noun}-{video_id}-{obs_frame_idx}-label.png')
                    
                    cv2.imwrite(img_name, cropped_out)
                    cv2.imwrite(label_name, temp_label)

                    # cropped_out = cv2.resize(cropped_out, (224, 224))
                    
                    # cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                    # cv2.imshow("image", out)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                