import os
from os.path import join as opj
import pickle
import argparse
import pandas as pd
import cv2
import glob
from tqdm import tqdm

from utils import *
from preprocess.dataset_util import FrameDetections, HandState, bbox_inter, sample_action_anticipation_frames, fetch_data, save_video_info
from preprocess.obj_util import find_active_side
from GroundedSAM.EfficientSAM.kinect_efficient_sam import ov_detect_seg, efficient_sam_box_prompt

filter_nouns = ["spoon", "knife", "bottle", "pan", "cup", "fork", "spatula", "scissors", "ladle", "jar"]
filter_verb_ids = [0] # filter_verbs: take

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/EPIC-KITCHENS", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./outputs_kitchen", type=str, help="generated results save path")
    # parser.add_argument('--filter_verb', default='take', type=str, help="extract related actions")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('-p', '--participant_id', nargs='*', default=None, type=str)
    parser.add_argument('--fps', default=60, type=int, help="sample frames per second")
    parser.add_argument('--box_threshold', default=0.35, type=float)
    parser.add_argument('--ho_threshold', default=0.1, type=float)
    parser.add_argument('--t_buffer', default=1, type=float)
    # parser.add_argument('--iou_threshold', default=0.5, type=float)

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    label_path = opj(args.dataset_path, 'EPIC_100_train.csv')
    annotations = pd.read_csv(label_path)

    if args.participant_id is None:
        par_list = glob.glob(opj(args.dataset_path, 'P*'))
        par_list.sort()
    else:
        par_list = [opj(args.dataset_path, p) for p in args.participant_id ]

    for par_path in tqdm(par_list):
        par_id = par_path.split('/')[-1]
        video_id_path = opj(par_path, "rgb_frames")
        video_id_list = os.listdir(video_id_path)
        video_id_list.sort()

        for video_id in video_id_list:
            # if video_id == 'P01_01':
        
            print(f"======================= Checking video: {video_id} =======================")
            sub_save_path = opj(save_path, par_id, video_id)
            os.makedirs(sub_save_path, exist_ok=True)

            frames_path = opj(video_id_path, video_id)
            ho_path = opj(args.dataset_path, "hand-objects", par_id, "{}.pkl".format(video_id))
            with open(ho_path, "rb") as f:
                video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
            
            pickle_save = []
            for index, row in annotations.iterrows():
                # row['verb_class'] in filter_verb_ids and 
                if row['verb_class'] in filter_verb_ids and row['participant_id'] == par_id and \
                        row['video_id'] == video_id and row['noun'] in filter_nouns:

                    obs_frame_idx, con_hand_box, con_obj_box, obs_obj_box, obs_obj_mask = None, None, None, None, None
                    print(f"current action: {row['narration']}")
                    noun = row['noun']
                    verb = row['verb']
                    start_act_frame = row['start_frame']
                    frames_idxs = sample_action_anticipation_frames(start_act_frame, fps=args.fps, t_buffer=args.t_buffer)

                    # Detect if contact exists, and fetch frames, annots, and hand sides (active hands)
                    results = fetch_data(frames_path, video_detections, frames_idxs)  
                    
                    if results is None:
                        print("data fetch failed")
                        continue
                    else:
                        frames_idxs, frames, annots, hand_sides = results
                    
                    contact_frame, contact_annot = frames[-1], annots[-1]
                    hands = [hand for hand in contact_annot.hands if hand.score >= args.hand_threshold]
                    objs = [obj for obj in contact_annot.objects if obj.score >= args.obj_threshold]

                    active_side = find_active_side(contact_annot, hand_sides, args.hand_threshold, args.obj_threshold)
                    active_side = 1 if active_side == 'RIGHT' else 0
                    hand_object_idx_correspondences = contact_annot.get_hand_object_interactions(object_threshold=args.obj_threshold,
                                                                                                    hand_threshold=args.hand_threshold)
                    
                    # Extract the hand & object boxes in contact
                    for h_idx, o_idx in hand_object_idx_correspondences.items():
                        if hands[h_idx].side.value == active_side:
                            con_obj_box = np.array(objs[o_idx].bbox.coords_int).reshape(-1)
                            con_hand_box = np.array(hands[h_idx].bbox.coords_int).reshape(-1)

                            con_obj_mask = efficient_sam_box_prompt(contact_frame, con_obj_box)
                            con_hand_mask = efficient_sam_box_prompt(contact_frame, con_hand_box)
                    
                    ################################ Locate the observation frame ################################
                    for frames_idx, frame, annot in zip(frames_idxs[:-1][::-1], frames[:-1][::-1], annots[:-1][::-1]):
                        
                        # first check if target objects exist
                        detection = ov_detect_seg(frame, [noun], SAMseg=True, box_th=args.box_threshold)
                        if len(detection.xyxy) == 0:
                            continue
                        
                        hands = [hand for hand in annot.hands if hand.score >= args.hand_threshold]
                        semantic_bboxes = detection.xyxy
                        semantic_masks = detection.mask
                        # temp_dist = 1000
                        # temp_obj_iou = 0
                        temp_metric = 0
                        temp_area_ratio = 0

                        for h in hands:
                            # no contact detected & active hand side
                            if h.state.value == HandState.NO_CONTACT.value and h.side.value == active_side:
                                obs_hand_bbox = np.array(h.bbox.coords_int).reshape(-1)

                                for i, sb in enumerate(semantic_bboxes):
                                    # compute center distance between semantic obj bbox and hand bbox
                                    sb_center = ((sb[0] + sb[2]) / 2, (sb[1] + sb[3]) / 2)
                                    obs_hand_center = np.array(h.bbox.center)
                                    dist = np.linalg.norm(sb_center - obs_hand_center)

                                    # compute IoU between semantic obj bbox and hand bbox / contact obj bbox
                                    hand_iou = bbox_inter(obs_hand_bbox, sb)[-1]
                                    obj_iou = bbox_inter(con_obj_box, sb)[-1]

                                    # compute size ratio
                                    con_obj_box_area = (con_obj_box[3] - con_obj_box[1]) * \
                                                    (con_obj_box[2] - con_obj_box[0])
                                    sb_area = (sb[3] - sb[1]) * (sb[2] - sb[0])
                                    area_ratio = con_obj_box_area / sb_area
                                    
                                    # similar size & has certain overlapping with contact object box (does not change dramatically)
                                    #  & No major occlusions with hand (small iou) & the distance with contact hand box is small 
                                    # and obj_iou > 0.2 
                                    metric = obj_iou / dist
                                    if (0.5 < area_ratio < 1.5) and hand_iou < args.ho_threshold and metric > temp_metric:
                                        temp_metric = metric
                                        # temp_dist = dist
                                        # temp_obj_iou = obj_iou
                                        obs_obj_box = sb.round().astype(int)
                                        obs_obj_mask = semantic_masks[i]
                            
                        if obs_obj_box is not None:
                            print(f"Found observation frame: obs / contact frames --- {frames_idx} / {frames_idxs[-1]}")
                            obs_frame_idx = frames_idx
                            break
                                        
                    if obs_frame_idx is not None:
                        pickle_save.append({'noun': noun, 'frame_idxs': [frames_idxs[-1], obs_frame_idx], 
                                            'obs_obj_box': obs_obj_box, 'obs_obj_mask': obs_obj_mask, 
                                            'con_ho_boxes': [con_hand_box, con_obj_box],
                                            'con_ho_masks': [con_hand_mask, con_obj_mask]})
                        # print(obs_frame_idx, con_hand_box, con_obj_box, obs_obj_box, hand_ious, obj_ious)

                        im2show1 = cv2.rectangle(frames[-1], tuple(con_hand_box[:2]), 
                                    tuple(con_hand_box[2:]), color=(0, 0, 255), thickness=1)
                        im2show1 = cv2.rectangle(im2show1, tuple(con_obj_box[:2]), 
                                                tuple(con_obj_box[2:]), color=(255, 0, 0), thickness=1)

                        # plot obs knife box
                        im2show2 = frames[frames_idxs.index(obs_frame_idx)]
                        for bb in semantic_bboxes:
                            box_each = bb.round().astype(int)
                            im2show2 = cv2.rectangle(im2show2, tuple(box_each[:2]), 
                                                tuple(box_each[2:]), color=(0, 255, 0), thickness=1)
                        im2show2 = cv2.rectangle(im2show2, tuple(obs_obj_box[:2]), 
                                                tuple(obs_obj_box[2:]), color=(255, 0, 0), thickness=1)
                            
                        # plot contact-2-obs overlapping box
                        im2show2 = cv2.rectangle(im2show2, tuple(obs_hand_bbox[:2]), 
                                                tuple(obs_hand_bbox[2:]), color=(0, 0, 255), thickness=1)
                        obs_obj_contour, _ = cv2.findContours(obs_obj_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(im2show2, obs_obj_contour, -1, (255, 255, 255), 1)

                        file_name = f"{obs_frame_idx}-{frames_idxs[-1]}-{verb}-{noun}.jpg"
                        img = cv2.hconcat([im2show1, im2show2])
                        cv2.imwrite(opj(sub_save_path, file_name), img)
                        
                        # visualization
                        # cv2.namedWindow(file_name, cv2.WINDOW_NORMAL)
                        # cv2.imshow(file_name, img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()
                    
                    else:
                        print('no valid observation frame in the current video clips')

            if len(pickle_save) > 0:
                with open(opj(sub_save_path, f'{video_id}.pkl'), 'wb') as f:
                    pickle.dump(pickle_save, f)
            else:
                os.rmdir(sub_save_path)
