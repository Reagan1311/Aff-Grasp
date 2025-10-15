import os
from os.path import join as opj
import pickle
import argparse
import pandas as pd
import cv2
import glob
from tqdm import tqdm

from utils import *
from preprocess.dataset_util import FrameDetections, bbox_inter, sample_action_anticipation_frames, fetch_data, save_video_info
from preprocess.obj_util import find_active_side
from GroundedSAM.EfficientSAM.kinect_efficient_sam import ov_detect_seg, efficient_sam_box_prompt

# filter_nouns = ["tap", "spoon", "knife", "pan", "lid", "bowl", "glass", "cup", "fork", "bottle", "spatula", "peeler", "scissors", "chopstick", "blender", "grater", "ladle", "tongs", "opener", "processor", "rolling pin", "presser", "whisk", "slicer", "cutter", "masher", "pestle", "container", "jar"]
filter_verbs = ["cut", "stir", 'scoop', 'stick']
# filter_verb_ids = [7]   # 7 cut; 10 mix; 16 scoop; 
filter_nouns = ['knife', 'scissors', 'spoon', 'spatula']
aff_obj_dict = {'cut': ['knife', 'scissors'], 'stir': ['spatula', 'spoon'], "stick": ['fork'], "scoop": ['spoon', 'ladle']}


def find_key_for_value(d, target):
    for key, value_list in d.items():
        if target in value_list:
            return key
    return None  # Return None if the target is not found in any of the lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/EPIC-KITCHENS", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./support_pts/epic_kitchens_low_threshold", type=str, help="generated results save path")
    # parser.add_argument('--filter_verb', default='take', type=str, help="extract related actions")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('-p', '--participant_id', nargs='*', default=None, type=str)
    parser.add_argument('--fps', default=60, type=int, help="sample frames per second")
    parser.add_argument('--box_threshold', default=0.35, type=float)
    parser.add_argument('--t_buffer', default=1, type=float)
    # parser.add_argument('--iou_threshold', default=0.5, type=float)
    parser.add_argument('--vis', action='store_true')

    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)
    save_path = args.save_path

    # label_path = os.path.join(args.dataset_path, 'EPIC_train_action_labels.csv')
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
        
            print(f"======================= Checking video: {video_id} =======================")
            # sub_save_path = opj(save_path, par_id, video_id)
            # os.makedirs(sub_save_path, exist_ok=True)

            frames_path = opj(video_id_path, video_id)
            ho_path = opj(args.dataset_path, "hand-objects", par_id, "{}.pkl".format(video_id))
            with open(ho_path, "rb") as f:
                video_detections = [FrameDetections.from_protobuf_str(s) for s in pickle.load(f)]
        
            active_aff_obj = {'cut': None, 'stir': None, 'stick': None, 'scoop': None}
            for index, row in annotations.iterrows():
                if row['noun'] in filter_nouns:
                    active_obj = row['noun']
                    active_aff = find_key_for_value(aff_obj_dict, active_obj)
                    active_aff_obj[active_aff] = active_obj

                if row['verb'] in filter_verbs and row['participant_id'] == par_id and \
                        row['video_id'] == video_id:

                    obs_frame_idx, con_hand_box, con_obj_box, obs_obj_box, obs_obj_mask = None, None, None, None, None
                    # print(f"current action: {row['narration']}")
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
                    
                    
                    ################################ Locate the functional point ################################
                    MOVE_FRAMES = 10
                    for frames_idx, frame, annot in zip(frames_idxs[:-MOVE_FRAMES][::-1], frames[:-MOVE_FRAMES][::-1], annots[:-MOVE_FRAMES][::-1]):
                    # for frames_idx, frame, annot in zip(frames_idxs[::-1], frames[::-1], annots[::-1]):

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
                        
                        # tool_dec = ov_detect_seg(frame, [active_aff_obj[verb]], SAMseg=True, box_th=args.box_threshold)
                        
                        num_obj = len(obj_dec.xyxy)
                        if num_obj == 0 or tool_dec is None or tool_mask is None:
                            continue

                        tool_box = tool_dec.round().astype(int)
                        tool_center = np.array([(tool_box[0] + tool_box[2]) / 2, (tool_box[1] + tool_box[3]) / 2])
                        tool_mask = tool_mask.astype(np.uint8)
                        erode_tool_mask = cv2.erode(tool_mask, np.ones((3, 3), np.uint8), iterations=1)
                        if len(erode_tool_mask[erode_tool_mask==1]) == 0:
                            continue

                        # if multiple objects, pick the one that is closest to the target object
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

                        # obj_mask = obj_dec.mask[np.argmax(obj_dec.confidence)]
                        # obj_box = obj_dec.xyxy[np.argmax(obj_dec.confidence)].round().astype(int)
                        
                        obj_mask_pts = np.column_stack(np.where(obj_mask==True))[:, ::-1]
                        tool_mask_pts = np.column_stack(np.where(tool_mask==True))[:, ::-1]
                        
                        def nearest_sample(a, b, num_pt=10, move_dist=5):
                            a = a[:, np.newaxis, :]
                            b = b[np.newaxis, :, :]
                            squared_diff = np.sum((a - b) ** 2, axis=-1)
                            distances = np.sqrt(squared_diff)
                            
                            point1 = None
                            point2 = b[0, np.argmin(np.sum(distances, axis=0))]
                        
                            # min_distance_idx = np.unravel_index(np.argmin(distances), distances.shape)
                            # a_idx = min_distance_idx[0]
                            # # min_distance = distances[min_distance_idx]
                            
                            # point1 = a[a_idx]
                            # max_idxes = np.argsort(distances[a_idx], axis=0)
                            # point2 = b[:, max_idxes[0]]

                            return point1, point2
                        
                        curr_frame = frame.copy()
                        obj_c, curr_frame = find_contour_draw(curr_frame, obj_mask, color=(255, 0, 255))
                        tool_c, curr_frame = find_contour_draw(curr_frame, erode_tool_mask, color=(0, 255, 0))
                        
                        p1, p2 = nearest_sample(obj_c, tool_c)    # shape: 1 x 2
                        
                        if args.vis:
                            for bbox in obj_dec.xyxy:
                                bb = bbox.round().astype(int)
                                curr_frame = cv2.rectangle(curr_frame, tuple(bb[:2]), tuple(bb[2:]), 
                                                    color=(255, 0, 255), thickness=1)    
                            # curr_frame = cv2.rectangle(curr_frame, tuple(obj_box[:2]), tuple(obj_box[2:]), 
                                                    # color=(255, 0, 255), thickness=1)
                            curr_frame = cv2.rectangle(curr_frame, tuple(tool_box[:2]), tuple(tool_box[2:]), 
                                                    color=(0, 255, 0), thickness=1)
                            # curr_frame = cv2.circle(curr_frame, p1[0], 1, (0,0,255), 2)
                            curr_frame = cv2.circle(curr_frame, p2, 1, (0,0,255), 2)
                            save_img = cv2.hconcat([frames[-1], curr_frame])
                            img_name = f'{active_aff_obj[verb]}-{verb}-{noun}-{video_id}-{frames_idx}-{frames_idxs[-1]}-{p2[0]}-{p2[1]}.jpg'
                            cv2.imwrite(opj(args.save_path, img_name), save_img)

                        else:
                            ex_tool_box = box_expansion(tool_box, size=frame.shape[:2], ratio=0.05)
                            cropped_frame = frame[ex_tool_box[1]:ex_tool_box[3], ex_tool_box[0]:ex_tool_box[2]]
                            p2_crop = p2 - ex_tool_box[:2]
                            # import pdb; pdb.set_trace()
                            img_name = f'{active_aff_obj[verb]}-{verb}-{noun}--{video_id}-{frames_idx}-{frames_idxs[-1]}-{p2_crop[0][0]}-{p2_crop[0][1]}.jpg'
                            cv2.imwrite(opj(args.save_path, img_name), cropped_frame)
                        
                        break

                        # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
                        # cv2.imshow('img', img)
                        # cv2.waitKey(0)
                        # cv2.destroyAllWindows()

                        # exit()
