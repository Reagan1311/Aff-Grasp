import cv2
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
from utils import *


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/EPIC-KITCHENS", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./outputs_kitchen", type=str, help="generated results save path")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('-p', '--participant_id', nargs='*', default=None, type=str)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    label_path = os.path.join(args.dataset_path, 'EPIC_100_train.csv')
    annotations = pd.read_csv(label_path)

    if args.participant_id is None:
        par_list = glob.glob(os.path.join(args.dataset_path, 'P*'))
        par_list.sort()
    else:
        par_list = [os.path.join(args.dataset_path, p) for p in args.participant_id ]

    for par_path in tqdm(par_list):
        par_id = par_path.split('/')[-1]
        video_id_path = os.path.join(par_path, "rgb_frames")
        video_id_list = os.listdir(video_id_path)
        video_id_list.sort()
        
        for video_id in video_id_list:
            # if video_id == 'P01_01':
                
            print(f"======================= Checking video: {video_id} =======================")
            sub_save_path = os.path.join(save_path, par_id, video_id)
            obs_pickle_path = os.path.join(sub_save_path, f"{video_id}.pkl")
            if not os.path.exists(obs_pickle_path):
                continue
            
            frames_path = os.path.join(video_id_path, video_id)
            ho_path = os.path.join(args.dataset_path, "hand-objects", par_id, "{}.pkl".format(video_id))    
            
            with open(ho_path, "rb") as f:
                video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
            with open(obs_pickle_path, "rb") as f:
                obs_frames = pickle.load(f)

            pickle_save = []
            for of in obs_frames:
                noun = of['noun']
                con_frame_idx, obs_frame_idx = of['frame_idxs']  # contact / obs frames idxs
                con_hand_box, con_obj_box = of['con_ho_boxes']
                con_hand_mask, con_obj_mask = of['con_ho_masks']
                obs_obj_box, obs_obj_mask = of['obs_obj_box'], of['obs_obj_mask']
                
                obs_obj_mask_copy = obs_obj_mask.copy().astype(np.uint8)
                obs_obj_mask_copy[obs_obj_mask_copy==True] = 255
                erode_obs_obj_mask = cv2.erode(obs_obj_mask_copy, np.ones((5, 5), np.uint8), iterations=1)
                x1, y1, x2, y2 = obs_obj_box
                x1_, y1_, x2_, y2_ = con_obj_box

                con_frame = load_img(frames_path, con_frame_idx)
                con_annot = load_ho_annot(video_detections, con_frame_idx)

                obs_frame = load_img(frames_path, obs_frame_idx)
                obs_annot = load_ho_annot(video_detections, obs_frame_idx)

                # Compute homography with SURF
                descriptor = cv2.xfeatures2d.SURF_create()

                con_mask = get_mask(con_frame, con_annot)
                obs_mask = get_mask(obs_frame, obs_annot)
                obs_mask[y1:y2, x1:x2] = 0

                # compute homography
                (kpsA, featuresA) = descriptor.detectAndCompute(con_frame, mask=con_mask)
                (kpsB, featuresB) = descriptor.detectAndCompute(obs_frame, mask=obs_mask)
                
                try:
                    _, H_AB, _ = match_keypoints(kpsA, kpsB, featuresA, featuresB)  # contact to observe
                    _, H_BA, _ = match_keypoints(kpsB, kpsA, featuresB, featuresA)
                    # H_BA = np.linalg.inv(H_AB)
                except Exception:
                    print("compute homography failed!")
                    continue
                
                if H_AB is None or H_BA is None:
                    continue
                
                # # infer corner points of bboxes
                # obj_pts_con = get_corner_point(con_obj_box).astype(np.float32)
                # obj_pts_obs = get_corner_point(obs_obj_box).astype(np.float32)

                # compute box aspect ratio
                con_obj_box_ar = (y2_ - y1_) * (x2_ - x1_)
                obs_obj_box_ar = (y2 - y1) * (x2 - x1)
                obs_to_con_obj_box, trans_mask = None, None
                area_ratio = con_obj_box_ar / obs_obj_box_ar
                print(f'area_ratio: {area_ratio}')
                if area_ratio < 0.8:
                # if len(obs_obj_mask[obs_obj_mask == True]) > 0:
                    obs_to_con_obj_box, trans_mask = get_box_after_masktrans(obs_obj_mask, H_BA)
                    # obs_to_con_pts, obs_to_con_xyxy = get_box_after_trans(obj_pts_obs, H_BA)
                
                    # compute offset
                    
                    if obs_to_con_obj_box is not None:
                        offset = con_obj_box - obs_to_con_obj_box
                        offset_tl, offset_br = np.absolute(offset[:2]).sum(), np.absolute(offset[2:]).sum()
                        if offset_tl > offset_br:
                            offset = offset[2:]
                        else:
                            offset = offset[:2]
                        obs_to_con_obj_box[:2] += offset
                        obs_to_con_obj_box[2:] += offset

                    # obs_to_con_mask = np.roll(obs_to_con_mask, offset[1], axis=0)
                    # obs_to_con_mask = np.roll(obs_to_con_mask, offset[0], axis=1)

                # compute overlapping bbox in the contact frame, and map it back to the obs frame
                if obs_to_con_obj_box is not None:
                    ovlap_box, iou = bbox_inter(con_hand_box, obs_to_con_obj_box)
                else:
                    ovlap_box, iou = bbox_inter(con_hand_box, con_obj_box)
                
                con_hand_mask[con_hand_mask==True] = 255
                con_hand_mask = con_hand_mask.astype(np.uint8)
                # con_hand_mask = skin_extract(con_frame)
                select_points = compute_affordance(con_hand_mask, ovlap_box)
                
                if select_points is not None:
                    select_points_homo = np.concatenate((select_points, np.ones((select_points.shape[0], 1), dtype=np.float32)), axis=1)
                    select_points_homo = np.dot(select_points_homo, H_AB.T)
                    select_points_homo = select_points_homo[:, :2] / select_points_homo[:, None, 2]
                    select_points_homo_mean = select_points_homo.mean(axis=0).round().astype(int)
                    select_points_homo = select_points_homo.round().astype(int)
                    
                    select_points = select_points.round().astype(np.int32)
                    select_mean = select_points.mean(axis=0).round().astype(int)
                    obs_obj_mask_pts = np.column_stack(np.where(obs_obj_mask==True))[:, ::-1]

                    for i, p in enumerate(select_points_homo):
                        if not np.all(p == obs_obj_mask_pts, axis=1).any():
                            dist = np.linalg.norm(p[None, :] - obs_obj_mask_pts, axis=1)
                            select_points_homo[i] = obs_obj_mask_pts[np.argmin(dist)]
                    
                    if not np.all(select_points_homo_mean == obs_obj_mask_pts, axis=1).any():
                        # print(select_points_homo_mean, obs_obj_mask_pts)
                        dist = np.linalg.norm(select_points_homo_mean[None, :] - obs_obj_mask_pts, axis=1)
                        select_points_homo_mean = obs_obj_mask_pts[np.argmin(dist)]

                    neg_pt = neg_pt_sampling(obs_obj_mask_pts, select_points_homo_mean.reshape(-1, 2))

                    # print(f'IoU of H-O Boxes in contact: {iou}')
                    # ovlap_box_pts = get_corner_point(ovlap_box).astype(np.float32)
                    # ovlap_box_pts_trans, ovlap_box_trans_obs = get_box_after_trans(ovlap_box_pts, H_AB)
                    # ovlap_box_trans_obs_expand = box_expansion(ovlap_box_trans_obs, con_frame.shape[:2])
                    # ovlap_box_trans_obs = ovlap_box_trans_obs.round().astype(int)
                    # ovlap_box_trans_obs_expand = ovlap_box_trans_obs_expand.round().astype(int)
                
                    # plot contact object bbox
                    im2show1 = cv2.rectangle(con_frame, tuple(con_obj_box[:2]), 
                                            tuple(con_obj_box[2:]), color=(255, 0, 0), thickness=1)
                    # plot contact hand bbox
                    im2show1 = cv2.rectangle(im2show1, tuple(con_hand_box[:2]), 
                                            tuple(con_hand_box[2:]), color=(0, 255, 255), thickness=1)
                    if obs_to_con_obj_box is not None:
                        im2show1 = cv2.rectangle(im2show1, tuple(obs_to_con_obj_box[:2]), 
                                                tuple(obs_to_con_obj_box[2:]), color=(0, 255, 0), thickness=1)
                    
                        _, contours, _ = cv2.findContours(trans_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(im2show1, contours, -1, (255, 255, 255), 1)
                    # plot original overlapping box
                    # im2show1 = cv2.rectangle(im2show1, tuple(ovlap_box[:2]), 
                    #                         tuple(ovlap_box[2:]), color=(0, 0, 255), thickness=1)
                    
                    _, hand_contours, _ = cv2.findContours(con_hand_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im2show1, hand_contours, -1, (0, 0, 0), 1)

                    for point in select_points:
                        cv2.circle(im2show1,tuple(point),1,(0,0,255), 2)
                    cv2.circle(im2show1,tuple(select_mean),1,(0,255,255), 2)

                    # plot obs knife box
                    im2show2 = cv2.rectangle(obs_frame, tuple(obs_obj_box[:2]), 
                                            tuple(obs_obj_box[2:]), color=(255, 0, 0), thickness=1)
                    # plot contact-2-obs overlapping box
                    # im2show2 = cv2.rectangle(im2show2, tuple(ovlap_box_trans_obs[:2]), 
                    #                         tuple(ovlap_box_trans_obs[2:]), color=(0, 0, 255), thickness=1)
                    # im2show2 = cv2.rectangle(im2show2, tuple(ovlap_box_trans_obs_expand[:2]), 
                    #                         tuple(ovlap_box_trans_obs_expand[2:]), color=(200, 200, 255), thickness=1)
                    # im2show2 = cv2.polylines(im2show2, [ovlap_box_pts_trans.astype(np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)
                    _, obs_obj_contour, _ = cv2.findContours(obs_obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(im2show2, obs_obj_contour, -1, (255, 255, 255), 1)
                    
                    for point in select_points_homo:
                        cv2.circle(im2show2,tuple(point),1,(0,0,255), 2)
                    cv2.circle(im2show2,tuple(select_points_homo_mean),1,(0,255,255), 2)
                    for n_pt in neg_pt:
                        cv2.circle(im2show2,tuple(n_pt), 1,(0,255,0), 2)

                    img = cv2.hconcat([im2show1, im2show2])
                    img_name = os.path.join(sub_save_path, f'{noun}-{con_frame_idx}-{obs_frame_idx}-affpts.jpg')
                    cv2.imwrite(img_name, img)

                    cv2.namedWindow(img_name, cv2.WINDOW_NORMAL)
                    cv2.imshow(img_name, img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()

                    pickle_save.append({'noun': noun, 'obs_frame_idx': obs_frame_idx, 
                                        'prompt_pts': [select_points_homo, neg_pt], 
                                        'prompt_box': obs_obj_box})
                else:
                    print('No point is selected.')
                    
            with open(os.path.join(sub_save_path,  f'{video_id}_multi_affpts.pkl'), 'wb') as f:
                pickle.dump(pickle_save, f)
