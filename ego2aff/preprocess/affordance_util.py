import os
import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from preprocess.dataset_util import bbox_inter
from PIL import Image
from utils import resize
import matplotlib.pyplot as plt
import torchvision.transforms as T


def skin_extract(image):
    def color_segmentation():
        lower_HSV_values = np.array([0, 40, 0], dtype="uint8")
        upper_HSV_values = np.array([25, 255, 255], dtype="uint8")
        lower_YCbCr_values = np.array((0, 138, 67), dtype="uint8")
        upper_YCbCr_values = np.array((255, 173, 133), dtype="uint8")
        mask_YCbCr = cv2.inRange(YCbCr_image, lower_YCbCr_values, upper_YCbCr_values)
        mask_HSV = cv2.inRange(HSV_image, lower_HSV_values, upper_HSV_values)
        binary_mask_image = cv2.add(mask_HSV, mask_YCbCr)
        return binary_mask_image

    HSV_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    YCbCr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    binary_mask_image = color_segmentation()
    image_foreground = cv2.erode(binary_mask_image, None, iterations=3)
    dilated_binary_image = cv2.dilate(binary_mask_image, None, iterations=3)
    ret, image_background = cv2.threshold(dilated_binary_image, 1, 128, cv2.THRESH_BINARY)

    image_marker = cv2.add(image_foreground, image_background)
    image_marker32 = np.int32(image_marker)
    cv2.watershed(image, image_marker32)
    m = cv2.convertScaleAbs(image_marker32)
    ret, image_mask = cv2.threshold(m, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((20, 20), np.uint8)
    image_mask = cv2.morphologyEx(image_mask, cv2.MORPH_CLOSE, kernel)
    return image_mask


def farthest_sampling(pcd, n_samples, init_pcd=None):
    def compute_distance(a, b):        
        return np.linalg.norm(a - b, ord=2, axis=2)

    n_pts, dim = pcd.shape[0], pcd.shape[1]
    selected_pts_expanded = np.zeros(shape=(n_samples, 1, dim))
    remaining_pts = np.copy(pcd)

    if init_pcd is None:
        if n_pts > 1:
            # 随机初始化一个点
            start_idx = np.random.randint(low=0, high=n_pts - 1)
        else:
            start_idx = 0
        selected_pts_expanded[0] = remaining_pts[start_idx]
        n_selected_pts = 1
    else:
        num_points = min(init_pcd.shape[0], n_samples)
        selected_pts_expanded[:num_points] = init_pcd[:num_points, None, :]
        n_selected_pts = num_points

    for _ in range(1, n_samples):
        if n_selected_pts < n_samples:
            dist_pts_to_selected = compute_distance(remaining_pts, selected_pts_expanded[:n_selected_pts]).T
            dist_pts_to_selected_min = np.min(dist_pts_to_selected, axis=1, keepdims=True)
            res_selected_idx = np.argmax(dist_pts_to_selected_min)
            selected_pts_expanded[n_selected_pts] = remaining_pts[res_selected_idx]
            n_selected_pts += 1

    selected_pts = np.squeeze(selected_pts_expanded, axis=1)
    return selected_pts


def neg_pt_sampling(mask_pts, pos_pt, num_pt=2, radius=5):
    def compute_distance(a, b):
        return np.linalg.norm(a - b, ord=2, axis=2)
    
    dist_pts_to_selected = compute_distance(mask_pts, pos_pt[:, None]).T
    # dist_pts_to_selected_avg = np.mean(dist_pts_to_selected, axis=1)
    # dist_pts_to_selected_min = np.min(dist_pts_to_selected, axis=1, keepdims=True)
    # max_idx = np.argmax(dist_pts_to_selected_min, axis=0)
    max_idxes = np.argsort(-dist_pts_to_selected, axis=0)
    # neg_pt = mask_pts[max_idxes[:num_pt]]
    neg_pt = mask_pts[max_idxes[0]]
    
    dist = dist_pts_to_selected[max_idxes[0]]

    neg_pt_extra = (neg_pt + 0.1 * (pos_pt - neg_pt)).round().astype(neg_pt.dtype)
    
    if not np.all(neg_pt_extra == mask_pts, axis=1).any():
        dist = np.linalg.norm(neg_pt_extra - mask_pts, ord=2, axis=1)
        neg_pt_extra = mask_pts[np.argmin(dist)].reshape(-1, 2)

    neg_pt = np.vstack([neg_pt, neg_pt_extra])
    return neg_pt


def neg_pt_dino_correspondence(noun, obs_frame, obj_box, vit, support_path, size=560):
    img_path = os.path.join(support_path, noun)
    img_list = os.listdir(img_path)
    
    corr_pt_ratio_list = []
    tar_frame = cv2.cvtColor(obs_frame.copy(), cv2.COLOR_BGR2RGB)
    for i in img_list:
        img_info = i[:-4].split('-')
        src_pt = (int(img_info[-2]), int(img_info[-1]))
        x, y = src_pt

        with torch.no_grad():
            p_size = size // 14
            img = Image.open(os.path.join(img_path, i)).convert('RGB')
            o_w, o_h = img.size
            
            re_w, re_h = 0, 0
            if o_w > o_h:
                re_w = size
                re_h = round(size / o_w * o_h)
                re_x, re_y = x / o_w * re_w, y / o_h * re_h
                re_y += (size - re_h) / 2
            else:
                re_h = size
                re_w = round(size / o_h * o_w)
                re_x, re_y = x / o_w * re_w, y / o_h * re_h
                re_x += (size - re_w) / 2
            
            img_dino_input = resize(img, target_res=size, resize=True, to_pil=True)
            img_batch = vit.preprocess_pil(img_dino_input)
            src_feat = vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, p_size, p_size)
            src_feat = nn.Upsample(size=(size, size), mode='bilinear')(src_feat) # 1, C, H, W
            src_feat = F.normalize(src_feat)
            src_vec = src_feat[0, :, int(re_y), int(re_x)].view(1, -1, 1, 1)

            obj_w, obj_h = obj_box[2] - obj_box[0], obj_box[3] - obj_box[1]
            frame = tar_frame[obj_box[1]:obj_box[3], obj_box[0]:obj_box[2]]
            frame = Image.fromarray(frame)
            frame = resize(frame, target_res=size, resize=True, to_pil=True)
            tar_input = vit.preprocess_pil(frame)
            tar_feat = vit.extract_descriptors(tar_input.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, p_size, p_size)
            tar_feat = nn.Upsample(size=(size, size), mode='bilinear')(tar_feat) # 1, C, H, W
            tar_feat = F.normalize(tar_feat)

            cos = nn.CosineSimilarity(dim=1)
            cos_map = cos(src_vec, tar_feat).cpu().numpy()
            max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
            # corr_pt = max_yx[1], max_yx[0]

            if obj_w > obj_h:
                short_len = round(size / obj_w * obj_h)
                corr_pt_ratio = max_yx[1] / size, (max_yx[0] - ((size - short_len) / 2)) / short_len
            else:
                short_len = round(size / obj_h * obj_w)
                corr_pt_ratio = (max_yx[1] - ((size - short_len) / 2)) / short_len, max_yx[0] / size
    
        corr_pt_ratio_list.append(corr_pt_ratio)
        # print(corr_pt_ratio)
        # import pdb;pdb.set_trace()
        
        heatmap = cos_map[0]
        heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        plt.tight_layout()
        axes[0].clear()
        axes[0].imshow(img_dino_input)
        axes[0].axis('off')
        axes[0].scatter(re_x, re_y, c='r', s=70)
        axes[0].set_title('source image')

        axes[1].clear()
        axes[1].imshow(frame)
        axes[1].imshow(255 * heatmap, alpha=0.45, cmap='viridis')
        axes[1].axis('off')
        axes[1].scatter(max_yx[1], max_yx[0], c='r', s=70)
        axes[1].set_title('source image')
        plt.show()
    # exit()

    # corr_pt_re = np.round(corr_pt_re + obj_box[:2]).astype(int)
    return corr_pt_ratio_list


def neg_pt_featup(noun, obs_frame, obj_box, vit, support_path, size=224, r_size=256):
    if 'knife' in noun:
        noun = 'knife'
    elif 'scissor' in noun:
        noun = 'scissors'
    
    img_path = os.path.join(support_path, noun)
    img_list = os.listdir(img_path)
    
    corr_pt_ratio_list = []
    tar_frame = cv2.cvtColor(obs_frame.copy(), cv2.COLOR_BGR2RGB)
    # from featup.util import norm, unnorm
    # from featup.plotting import plot_feats
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    for i in img_list:
        img_info = i[:-4].split('-')
        src_pt = (int(img_info[-2]), int(img_info[-1]))
        x, y = src_pt

        with torch.no_grad():
            img = Image.open(os.path.join(img_path, i)).convert('RGB')
            o_w, o_h = img.size
            
            re_w, re_h = 0, 0
            if o_w > o_h:
                re_w = r_size
                re_h = round(r_size / o_w * o_h)
                re_x, re_y = x / o_w * re_w, y / o_h * re_h
                re_y += (r_size - re_h) / 2
            else:
                re_h = r_size
                re_w = round(r_size / o_h * o_w)
                re_x, re_y = x / o_w * re_w, y / o_h * re_h
                re_x += (r_size - re_w) / 2
            
            img_resize = resize(img, target_res=size, resize=True, to_pil=True)
            image_tensor = transform(img_resize).unsqueeze(0).cuda()
            src_feat = vit(image_tensor)
            # lr_src_feats = vit.model(image_tensor)
            # lr_up_src_feat = nn.Upsample(size=(size, size), mode='bilinear')(lr_src_feats) # 1, C, H, W
            # src_feat = vit.extract_descriptors(img_batch.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, p_size, p_size)
            # src_feat = vit(img_batch)
            # src_feat = F.normalize(src_feat)
            
            src_vec = src_feat[0, :, int(re_y), int(re_x)].view(1, -1, 1, 1)

            obj_w, obj_h = obj_box[2] - obj_box[0], obj_box[3] - obj_box[1]
            frame = tar_frame[obj_box[1]:obj_box[3], obj_box[0]:obj_box[2]]
            frame = Image.fromarray(frame)
            frame = resize(frame, target_res=size, resize=True, to_pil=True)
            frame_tensor = transform(frame).unsqueeze(0).cuda()
            tar_feat = vit(frame_tensor)
            # lr_tar_feats = vit.model(frame_tensor)
            # lr_up_tar_feat = nn.Upsample(size=(size, size), mode='bilinear')(lr_tar_feats) # 1, C, H, W
            # tar_feat = vit.extract_descriptors(tar_input.cuda(), layer=11, facet='token').permute(0, 1, 3, 2).reshape(1, -1, p_size, p_size)
            
            # tar_feat = F.normalize(tar_feat)
            cos = nn.CosineSimilarity(dim=1)
            cos_map = cos(src_vec, tar_feat).cpu().numpy()
            max_yx = np.unravel_index(cos_map[0].argmax(), cos_map[0].shape)
            # corr_pt = max_yx[1], max_yx[0]

            # corr_pt_ratio = max_yx[1] / size, max_yx[0] / size
            # plot_feats(unnorm(image_tensor)[0], lr_src_feats[0], src_feat[0])
            # plot_feats(unnorm(frame_tensor)[0], lr_tar_feats[0], tar_feat[0])
            if obj_w > obj_h:
                short_len = round(r_size / obj_w * obj_h)
                corr_pt_ratio = max_yx[1] / r_size, (max_yx[0] - ((r_size - short_len) / 2)) / short_len
            else:
                short_len = round(r_size / obj_h * obj_w)
                corr_pt_ratio = (max_yx[1] - ((r_size - short_len) / 2)) / short_len, max_yx[0] / r_size
    
        corr_pt_ratio_list.append(corr_pt_ratio)
        
        # # import pdb;pdb.set_trace()
        
        # heatmap = cos_map[0]
        # heatmap = (heatmap - np.min(heatmap)) / (np.max(heatmap) - np.min(heatmap))
        
        # fig, axes = plt.subplots(1, 2, figsize=(6, 3))
        # plt.tight_layout()
        # axes[0].clear()
        # axes[0].imshow(img_resize.resize((r_size, r_size)))
        # axes[0].axis('off')
        # axes[0].scatter(re_x, re_y, c='r', s=70)
        # axes[0].set_title('source image')

        # axes[1].clear()
        # axes[1].imshow(frame.resize((r_size, r_size)))
        # axes[1].imshow(255 * heatmap, alpha=0.45, cmap='viridis')
        # axes[1].axis('off')
        # axes[1].scatter(max_yx[1], max_yx[0], c='r', s=70)
        # axes[1].set_title('source image')
        # plt.show()

    # corr_pt_re = np.round(corr_pt_re + obj_box[:2]).astype(int)
    return corr_pt_ratio_list


def compute_heatmap(points, image_size, k_ratio=3.0):
    points = np.asarray(points)
    heatmap = np.zeros((image_size[0], image_size[1]), dtype=np.float32)
    n_points = points.shape[0]
    for i in range(n_points):
        x = points[i, 0]
        y = points[i, 1]
        col = int(x)
        row = int(y)
        try:
            heatmap[col, row] += 1.0
        except:
            col = min(max(col, 0), image_size[0] - 1)
            row = min(max(row, 0), image_size[1] - 1)
            heatmap[col, row] += 1.0
    k_size = int(np.sqrt(image_size[0] * image_size[1]) / k_ratio)
    if k_size % 2 == 0:
        k_size += 1
    heatmap = cv2.GaussianBlur(heatmap, (k_size, k_size), 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    heatmap = heatmap.transpose()
    return heatmap


def select_points_bbox(bbox, points, tolerance=2):
    x1, y1, x2, y2 = bbox
    ind_x = np.logical_and(points[:, 0] > x1-tolerance, points[:, 0] < x2+tolerance)
    ind_y = np.logical_and(points[:, 1] > y1-tolerance, points[:, 1] < y2+tolerance)
    ind = np.logical_and(ind_x, ind_y)
    indices = np.where(ind == True)[0]
    return points[indices]


def find_contour_points(mask):
    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        c = c.squeeze(axis=1)
        return c
    else:
        return None


def get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj):
    # active_obj_traj: active obj traj in last observation frame
    # obj_bboxs_traj: active obj bbox traj in last observation frame
    select_points_homo = np.concatenate((select_points, np.ones((select_points.shape[0], 1), dtype=np.float32)), axis=1)
    select_points_homo = np.dot(select_points_homo, homography.T)
    select_points_homo = select_points_homo[:, :2] / select_points_homo[:, None, 2]

    obj_point_last_observe = np.array(active_obj_traj[0])
    obj_point_future_start = np.array(active_obj_traj[-1])

    # 这里算上了物体移动的扰动
    future2last_trans = obj_point_last_observe - obj_point_future_start
    select_points_homo = select_points_homo + future2last_trans

    fill_indices = [idx for idx, points in enumerate(obj_bboxs_traj) if points is not None]
    contour_last_observe = obj_bboxs_traj[fill_indices[0]]
    contour_future_homo = obj_bboxs_traj[fill_indices[-1]] + future2last_trans
    contour_last_observe = contour_last_observe[:, None, :].astype(np.int)
    contour_future_homo = contour_future_homo[:, None, :].astype(np.int)
    filtered_points = []
    for point in select_points_homo:
        if cv2.pointPolygonTest(contour_last_observe, (point[0], point[1]), False) >= 0 \
                or cv2.pointPolygonTest(contour_future_homo, (point[0], point[1]), False) >= 0:
            filtered_points.append(point)
    filtered_points = np.array(filtered_points)
    return filtered_points


def compute_affordance(skin_mask, ho_overlap_box, num_points=5, num_sampling=20):
    # skin_mask = skin_extract(frame)
    xA, yA, xB, yB = ho_overlap_box
    ho_center = ((xA + xB) / 2, (yA + yB) / 2)
    select_points, init_points = None, None
    contact_points = find_contour_points(skin_mask)

    if contact_points is not None and contact_points.shape[0] > 0:
        contact_points = select_points_bbox((xA, yA, xB, yB), contact_points)
        if contact_points.shape[0] >= num_points:
            if contact_points.shape[0] > num_sampling:
                contact_points = farthest_sampling(contact_points, n_samples=num_sampling)
            distance = np.linalg.norm(contact_points - ho_center, ord=2, axis=1)
            indices = np.argsort(distance)[:num_points]
            select_points = contact_points[indices]
        elif contact_points.shape[0] > 0:
            print("no enough boundary points detected, sampling points in interaction region")
            init_points = contact_points
        else:
            print("no boundary points detected, use farthest point sampling")
    else:
        print("no boundary points detected, use farthest point sampling")
    if select_points is None:
        ho_mask = np.zeros_like(skin_mask, dtype=np.uint8)
        ho_mask[yA:yB, xA:xB] = 255
        ho_mask = cv2.bitwise_and(skin_mask, ho_mask)
        points = np.array(np.where(ho_mask[yA:yB, xA:xB] > 0)).T
    
        # 如果没有点的话，就直接在overlapping区域采样
        if points.shape[0] == 0:
            xx, yy = np.meshgrid(np.arange(xB - xA), np.arange(yB - yA))
            xx += xA
            yy += yA
            points = np.vstack([xx.reshape(-1), yy.reshape(-1)]).T
        # 这里加入hand mask和ho overlapping的重合点
        else:
            points = points[:, [1, 0]]
            points[:, 0] += xA
            points[:, 1] += yA
        if not points.shape[0] > 0:
            return None
        contact_points = farthest_sampling(points, n_samples=min(num_sampling, points.shape[0]), init_pcd=init_points)
        distance = np.linalg.norm(contact_points - ho_center, ord=2, axis=1)
        indices = np.argsort(distance)[:num_points]
        select_points = contact_points[indices]
    return select_points


def compute_obj_affordance(frame, annot, active_obj, active_obj_idx, homography,
                           active_obj_traj, obj_bboxs_traj,
                           num_points=5, num_sampling=20,
                           hand_threshold=0.1, obj_threshold=0.1):
    affordance_info = {}
    hand_object_idx_correspondences = annot.get_hand_object_interactions(object_threshold=obj_threshold,
                                                                         hand_threshold=hand_threshold)
    select_points = None
    for hand_idx, object_idx in hand_object_idx_correspondences.items():
        if object_idx == active_obj_idx:
            active_hand = annot.hands[hand_idx]
            affordance_info[active_hand.side.name] = np.array(active_hand.bbox.coords_int).reshape(-1)
            cmap_points = compute_affordance(frame, active_hand, active_obj, num_points=num_points, num_sampling=num_sampling)
            if select_points is None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = cmap_points
            elif select_points is not None and (cmap_points is not None and cmap_points.shape[0] > 0):
                select_points = np.concatenate((select_points, cmap_points), axis=0)
    if select_points is None:
        print("affordance contact points filtered out")
        return None
    select_points_homo = get_points_homo(select_points, homography, active_obj_traj, obj_bboxs_traj)
    if len(select_points_homo) == 0:
        print("affordance contact points filtered out")
        return None
    else:
        affordance_info["select_points"] = select_points
        affordance_info["select_points_homo"] = select_points_homo

        obj_bbox = np.array(active_obj.bbox.coords_int).reshape(-1)
        affordance_info["obj_bbox"] = obj_bbox
        return affordance_info