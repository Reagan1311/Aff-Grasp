import pickle
import argparse
import cv2
import json
import numpy as np
from os.path import join as opj


filter_nouns = ['knife_(knife,_machete)', 'axe', 'shovel_(hoe,_shovel,_spade)', 
                'scissor', 'screwdriver', 'spatula', 'paintbrush', 'ladle', 
                'trowel', 'hammer_(hammer,_mallet)', 'toothbrush', 'pliers', 'racket']
filter_verbs = ['hold_(support,_grip,_grasp)', 'take_(pick,_grab,_get)']


def bbox_dict2np(box_dict):
    x, y = box_dict['x'], box_dict['y']
    w, h = box_dict['width'], box_dict['height']
    out = np.array([x, y, x + w, y + h]).round().astype(int)
    return out


clip_uids = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default="/home/gen/Ego4d/data", type=str, help='dataset root')
    parser.add_argument('--save_path', default="./outputs", type=str, help="generated results save path")
    parser.add_argument('--filter_verb', default='take', type=str, help="extract related actions")
    parser.add_argument('--hand_threshold', default=0.1, type=float, help="hand detection threshold")
    parser.add_argument('--obj_threshold', default=0.1, type=float, help="object detection threshold")
    parser.add_argument('-p', '--participant_id', nargs='*', default=None, type=str)
    parser.add_argument('--fps', default=30, type=int, help="sample frames per second")
    parser.add_argument('--box_threshold', default=0.4, type=float)
    parser.add_argument('--ho_threshold', default=0.1, type=float)

    args = parser.parse_args()
    # os.makedirs(args.save_path, exist_ok=True)
    # save_path = args.save_path
    
    annot_path = opj(args.dataset_path, 'v2/annotations/fho_main.json')
    taxnomoy = opj(args.dataset_path, 'v2/annotations/fho_main_taxonomy.json')
    video_path = opj(args.dataset_path, 'v2/clips')
    with open(annot_path, 'r') as f:
        load_annot = json.load(f)['videos']

    # with open(taxnomoy, 'r') as f:
    #     load_tax = json.load(f)
    # nouns = load_tax['nouns']
    # verbs = load_tax['verbs']
    # print(nouns, verbs)
    # exit()

    for i, video in enumerate(load_annot):
        annots = video['annotated_intervals']
        meta_data = video['video_metadata']
        fps = meta_data['fps']
        w, h = meta_data['width'], meta_data['height']
        
        for annot in annots:
            clip_name = annot['clip_uid']
            clip_path = opj(video_path, clip_name + '.mp4')
            clip = cv2.VideoCapture(clip_path)

            narr_actions = annot['narrated_actions']
            if len(narr_actions) == 0:
                continue

            for narr_dict in narr_actions:
                # print(narr_dict['is_valid_action'])
                # print(narr_dict['is_valid_action'] is False)
                verb = narr_dict['structured_verb']
                frames = narr_dict['frames']
                if not narr_dict['is_valid_action'] or narr_dict['is_invalid_annotation'] or \
                    frames is None or verb not in filter_verbs:
                    continue
                
                # noun = narr_dict['structured_noun']
                start_frame, end_frame = narr_dict['start_frame'], narr_dict['end_frame']
                
                # pre_45/30/15, post_frame, contact_frame, pre_frame, pnr_frame
                clip_critical_frames = narr_dict['clip_critical_frames']
                pre_frame_idx = clip_critical_frames['pre_frame']
                con_frame_idx = clip_critical_frames['contact_frame']
                
                # frames is a list, containing dicts that include 
                
                # print(frames)
                
                flag = False
                for f in frames:
                    frame_type = f['frame_type']
                    if frame_type != 'contact_frame':
                        continue
                    frame_idx = f['frame_number']
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

                    if clip_name not in clip_uids:
                        clip_uids.append(clip_name)
                        flag = True
                        break
                
                if flag:
                    break
                
                    # clip.set(cv2.CAP_PROP_POS_FRAMES, con_frame_idx)
                    # res, frame = clip.read()
                    
                    # for b in all_box:
                    #     if b is not None:
                    #         frame = cv2.rectangle(frame, tuple(b[:2]), 
                    #                         tuple(b[2:]), color=(255, 0, 0), thickness=1)
                    
                    # win_name = verb + ' ' + noun
                    # cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                    # cv2.imshow(win_name, frame)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # break
                
                # print(narr_dict, len(narr_dict))
                # exit()
                # start_frame = narr_dict
                
print(len(clip_uids))
with open('ego4d_interested_clip_uids_1.pkl', 'wb') as f:
    pickle.dump(clip_uids, f)
    