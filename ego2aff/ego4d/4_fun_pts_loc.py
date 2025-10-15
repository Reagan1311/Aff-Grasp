import os
from os.path import join as opj
import pickle
import argparse
import pandas as pd
import cv2
import glob
import json
from tqdm import tqdm

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent)) 

from utils import *
from preprocess.dataset_util import FrameDetections, bbox_inter, sample_action_anticipation_frames, fetch_data_ego4d, save_video_info
from preprocess.obj_util import find_active_side
from GroundedSAM.EfficientSAM.kinect_efficient_sam import ov_detect_seg, efficient_sam_box_prompt


filter_nouns = ['knife_(knife,_machete)', 'axe', 'scissor', 'screwdriver', 'spatula', 'ladle', 'hammer_(hammer,_mallet)']
filter_verbs = ["cut_(trim,_slice,_chop)", "hit_(knock,_hit,_hammer)", "drill", "scoop"]
aff_obj_dict = {"cut_(trim,_slice,_chop)": ['knife_(knife,_machete)', 'axe', 'scissor'],
                "hit_(knock,_hit,_hammer)": ['hammer_(hammer,_mallet)'],
                "scoop": ['spatula', 'ladle'],
                "drill": ['screwdriver'],
                "screw": ['screwdriver'], 
                "unscrew": ['screwdriver']}


def find_key_for_value(d, target):
    for key, value_list in d.items():
        if target in value_list:
            return key
    return None  # Return None if the target is not found in any of the lists


def bbox_dict2np(box_dict):
    x, y = box_dict['x'], box_dict['y']
    w, h = box_dict['width'], box_dict['height']
    out = np.array([x, y, x + w, y + h]).round().astype(int)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/Ego4d/data/v2/", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./support_pts/ego4d", type=str, help="generated results save path")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('--fps', default=30, type=int, help="sample frames per second")
    parser.add_argument('--box_threshold', default=0.25, type=float)
    parser.add_argument('--ho_threshold', default=0.1, type=float)
    parser.add_argument('--t_buffer', default=1, type=float)
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    annot_path = opj(args.dataset_path, 'annotations/fho_main.json')
    taxnomoy = opj(args.dataset_path, 'annotations/fho_main_taxonomy.json')
    ho_annot_path = opj(args.dataset_path, 'clips_ho_detections')
    video_path = opj(args.dataset_path, 'clips')
    
    interest_clip_uids = os.listdir(ho_annot_path)
    interest_clip_uids = [uid[:-4] for uid in interest_clip_uids]

    with open(annot_path, 'r') as f:
        load_annot = json.load(f)['videos']

    for video in tqdm(load_annot):
        annots = video['annotated_intervals']
        meta_data = video['video_metadata']
        fps = meta_data['fps']
        frame_size = (meta_data['width'], meta_data['height'])
        
        for annot in annots:
            clip_name = annot['clip_uid']
            if clip_name not in interest_clip_uids:
                continue
            
            clip_path = opj(video_path, clip_name + '.mp4')
            clip = cv2.VideoCapture(clip_path)

            narr_actions = annot['narrated_actions']
            if len(narr_actions) == 0:
                continue

            ho_pkl_path = os.path.join(ho_annot_path, f"{clip_name}.pkl")
            with open(ho_pkl_path, "rb") as f:
                video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
        
            active_aff_obj = {"cut_(trim,_slice,_chop)": None,
                "hit_(knock,_hit,_hammer)": None,
                "scoop": None, "drill": None, "screw": None, "unscrew": None}
            
            for narr_dict in narr_actions:
                verb = narr_dict['structured_verb']
                frames = narr_dict['frames']
                if not narr_dict['is_valid_action'] or narr_dict['is_invalid_annotation'] or \
                    frames is None or verb not in filter_verbs:
                    continue
                
                start_frame, end_frame = narr_dict['start_frame'], narr_dict['end_frame']
                
                # pre_45/30/15, post_frame, contact_frame, pre_frame, pnr_frame
                clip_critical_frames = narr_dict['clip_critical_frames']
                pre_frame_idx = clip_critical_frames['pre_frame']
                con_frame_idx = clip_critical_frames['contact_frame']
                
                # frames is a list, containing dicts that include 
                for f in frames:
                    frame_type = f['frame_type']
                    if frame_type != 'contact_frame':
                        continue
                    # contact_frame = f['frame_number']
                    boxes = f['boxes']
                    
                    lh_box, rh_box, obj_box = None, None, None
                    for bbox in boxes:
                        if bbox['object_type'] == 'object_of_change':
                            noun = bbox['structured_noun']
                            obj_box = bbox_dict2np(bbox['bbox'])
                        elif bbox['object_type'] == 'left_hand':
                            lh_box = bbox_dict2np(bbox['bbox'])
                        elif bbox['object_type'] == 'right_hand':
                            rh_box = bbox_dict2np(bbox['bbox'])
                    all_box = [lh_box, rh_box, obj_box]
                    
                    if noun not in filter_nouns:
                        break
                    else:
                        active_obj = noun
                        active_aff = find_key_for_value(aff_obj_dict, active_obj)
                        active_aff_obj[active_aff] = active_obj
        
                    print(f"======================= Checking video: {clip_name} =======================")

                    frames_idxs = sample_action_anticipation_frames(con_frame_idx, fps=args.fps, fps_init=30, t_buffer=args.t_buffer)
                    results = fetch_data_ego4d(clip, video_detections, frames_idxs, frame_size)   
                    
                    if results is None:
                        print("data fetch failed")
                        continue
                    else:
                        frames_idxs, frames, annots, hand_sides = results
                        
                    ################################ Locate the functional point ################################
                    # flag = False
                    # MOVE_FRAMES = -5
                    for frames_idx, frame, annot in zip(frames_idxs[::-1], frames[::-1], annots[::-1]):

                        # first check if target objects exist
                        print(active_aff_obj, verb, noun)
                        obj_dec = ov_detect_seg(frame, [noun], SAMseg=True, box_th=args.box_threshold)
                        if active_aff_obj[verb] is None:
                            break
                        
                        hands = [hand for hand in annot.hands if hand.score >= args.hand_threshold]
                        objs = [obj for obj in annot.objects if obj.score >= args.obj_threshold]
                        if len(hands) == 0 or len(objs) == 0:
                            continue
                                
                        hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=args.obj_threshold,
                                                                                            hand_threshold=args.hand_threshold)
                        tool_dec, tool_mask = None, None
                        for h_idx, o_idx in hand_object_idx_correspondences.items():
                            if hands[h_idx].side.value == 1:
                                tool_dec = np.array(objs[o_idx].bbox.coords_int).reshape(-1)
                                tool_mask = efficient_sam_box_prompt(frame, tool_dec)
                        
                        num_obj = len(obj_dec.xyxy)
                        if num_obj == 0 or tool_dec is None or tool_mask is None:
                            continue
                        
                        tool_box = tool_dec.round().astype(int)
                        tool_center = np.array([(tool_box[0] + tool_box[2]) / 2, (tool_box[1] + tool_box[3]) / 2])
                        tool_mask = tool_mask.astype(np.uint8)
                        erode_tool_mask = cv2.erode(tool_mask, np.ones((3, 3), np.uint8), iterations=1)
                        if len(erode_tool_mask[erode_tool_mask==1]) == 0:
                            continue

                        if num_obj > 1:
                            obj_center = np.array([[(i[0] + i[2]) / 2, (i[1] + i[3]) / 2] for i in obj_dec.xyxy])
                            obj_tool_dist = [np.linalg.norm(oc - tool_center) for oc in obj_center]
                            obj_tool_metric = obj_dec.confidence / obj_tool_dist
                            obj_idx = np.argmax(obj_tool_metric)
                            obj_mask = obj_dec.mask[obj_idx]
                            obj_box = obj_dec.xyxy[obj_idx].round().astype(int)

                        else:
                            obj_mask = obj_dec.mask[0]
                            obj_box = obj_dec.xyxy[0].round().astype(int)

                        tool_obj_iou = bbox_inter(tool_box, obj_box)[-1]
                        if tool_obj_iou > 0.3:
                            continue

                        obj_mask = obj_dec.mask[np.argmax(obj_dec.confidence)]
                        obj_box = obj_dec.xyxy[np.argmax(obj_dec.confidence)].round().astype(int)
                        
                        obj_mask_pts = np.column_stack(np.where(obj_mask==True))[:, ::-1]
                        tool_mask_pts = np.column_stack(np.where(tool_mask==True))[:, ::-1]
                        
                        def nearest_sample(a, b, num_pt=10, move_dist=5):
                            a = a[:, np.newaxis, :]
                            b = b[np.newaxis, :, :]
                            squared_diff = np.sum((a - b) ** 2, axis=-1)
                            distances = np.sqrt(squared_diff)

                            # dist_min = np.min(distances, axis=1)
                            # max_idxes = np.argsort(-dist_min, axis=0)
                            # fun_pt = a[max_idxes[np.random.choice(num_pt)]]

                            min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
                            a_idx = min_distance_idx[0]
                            # min_distance = distances[min_distance_idx]
                            
                            point1 = a[a_idx]
                            # point2 = b[:, min_distance_idx[1]]
                            max_idxes = np.argsort(distances[a_idx], axis=0)
                            point2 = b[:, max_idxes[0]]
                            
                            # point2 = b[:, max_idxes[np.random.choice(num_pt)]]
                            
                            # x, y = point2[0]
                            # potential_moves = [(x + dx, y + dy) for dx in range(-move_dist, move_dist + 1) for dy in range(-move_dist, move_dist + 1)]
                            
                            # valid_moves = [pt for pt in potential_moves 
                            #                if cv2.pointPolygonTest(b, (float(pt[0]), float(pt[1])), measureDist=True) > 1.0]
                            # # If there are valid moves, choose one randomly
                            # if valid_moves:
                            #     point2 = np.array(valid_moves[np.random.randint(len(valid_moves))]).reshape(1, 2)

                            return point1, point2
                        
                        curr_frame = frame.copy()
                        obj_c, curr_frame = find_contour_draw(curr_frame, obj_mask, color=(255, 0, 255))
                        tool_c, curr_frame = find_contour_draw(curr_frame, tool_mask, color=(0, 255, 0))
                        
                        p1, p2 = nearest_sample(obj_c, tool_c)    # shape: 1 x 2
                        
                        if args.vis:
                            curr_frame = cv2.rectangle(curr_frame, tuple(obj_box[:2]), tuple(obj_box[2:]), 
                                                    color=(255, 0, 255), thickness=1)
                            curr_frame = cv2.rectangle(curr_frame, tuple(tool_box[:2]), tuple(tool_box[2:]), 
                                                    color=(0, 255, 0), thickness=1)
                            curr_frame = cv2.circle(curr_frame, p1[0], 1, (0,0,255), 2)
                            curr_frame = cv2.circle(curr_frame, p2[0], 1, (0,0,255), 2)
                            save_img = cv2.hconcat([frames[-1], curr_frame])
                            img_name = f'{active_aff_obj[verb]}-{verb}-{noun}-{clip_name}-{frames_idx}-{frames_idxs[-1]}-{p2[0][0]}-{p2[0][1]}.jpg'
                            cv2.imwrite(opj(args.save_path, img_name), save_img)

                        # else:
                        ex_tool_box = box_expansion(tool_box, size=frame.shape[:2], ratio=0.05)
                        cropped_frame = frame[ex_tool_box[1]:ex_tool_box[3], ex_tool_box[0]:ex_tool_box[2]]
                        p2_crop = p2 - ex_tool_box[:2]
                        # import pdb; pdb.set_trace()
                        img_name = f'{active_aff_obj[verb]}-{verb}-{noun}--{clip_name}-{frames_idx}-{frames_idxs[-1]}-{p2_crop[0][0]}-{p2_crop[0][1]}.jpg'
                        cv2.imwrite(opj(args.save_path, img_name), cropped_frame)
                        
                        break

                        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                        # cv2.imshow('img', img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # exit()
