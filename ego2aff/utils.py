import cv2
import numpy as np
from PIL import Image

def resize(img, target_res=224, resize=True, to_pil=True, edge=False):
    original_width, original_height = img.size
    original_channels = len(img.getbands())
    if not edge:
        canvas = np.zeros([target_res, target_res, 3], dtype=np.uint8)
        if original_channels == 1:
            canvas = np.zeros([target_res, target_res], dtype=np.uint8)
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[(width - height) // 2: (width + height) // 2] = img
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            canvas[:, (height - width) // 2: (height + width) // 2] = img
    else:
        if original_height <= original_width:
            if resize:
                img = img.resize((target_res, int(np.around(target_res * original_height / original_width))), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            top_pad = (target_res - height) // 2
            bottom_pad = target_res - height - top_pad
            img = np.pad(img, pad_width=[(top_pad, bottom_pad), (0, 0), (0, 0)], mode='edge')
        else:
            if resize:
                img = img.resize((int(np.around(target_res * original_width / original_height)), target_res), Image.Resampling.LANCZOS)
            width, height = img.size
            img = np.asarray(img)
            left_pad = (target_res - width) // 2
            right_pad = target_res - width - left_pad
            img = np.pad(img, pad_width=[(0, 0), (left_pad, right_pad), (0, 0)], mode='edge')
        canvas = img
    if to_pil:
        canvas = Image.fromarray(canvas)
    return canvas


def find_contour_draw(img, mask, color, alpha=0.5, thickness=1):
    out = img.copy()
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    c = c.squeeze(axis=1)
    
    overlay_mask = np.zeros_like(img, np.uint8)
    cv2.fillPoly(overlay_mask, [c], color)
    mask_pos = overlay_mask.astype(bool)
    out[mask_pos] = cv2.addWeighted(img, alpha, overlay_mask, 1 - alpha, 0)[mask_pos]
    cv2.drawContours(out, [c], -1, color, 1)

    return c, out

def match_keypoints(kpsA, kpsB, featuresA, featuresB, ratio=0.7, reprojThresh=4.0):
    matcher = cv2.DescriptorMatcher_create("BruteForce")
    rawMatches = matcher.knnMatch(featuresA, featuresB, 2)
    matches = []

    for m in rawMatches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            matches.append((m[0]))

    if len(matches) > 4:
        ptsA = np.float32([kpsA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpsB[m.trainIdx].pt for m in matches])
        (H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reprojThresh)
        matchesMask = status.ravel().tolist()
        return matches, H, matchesMask
    return None


def bbox_inter(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    box = np.array([xA, yA, xB, yB])

    if interArea == 0:
        return box, 0

    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    iou = interArea / float(boxAArea + boxBArea - interArea)

    return box, iou


def get_homo_point(point, homography):
    cx, cy = point
    center = np.array((cx, cy, 1.0), dtype=np.float32)
    x, y, z = np.dot(homography, center)
    x, y = x / z, y / z
    point = np.array((x, y), dtype=np.float32)
    return point


def get_corner_point(box):
    xA, yA, xB, yB = box
    tl, tr, br, bl = (xA, yA), (xB, yA), (xB, yB), (xA, yB)
    pts = np.array([tl, tr, br, bl]).reshape(-1, 1, 2)
    return pts


def box_expansion(box, size, ratio=0.01):
    H, W = size
    xA, yA, xB, yB = box
    xA = max(xA * (1 - ratio), 0)
    yA = max(yA * (1 - ratio), 0)
    xB = min(xB * (1 + ratio), W)
    yB = min(yB * (1 + ratio), H)
    expand_box = np.array([xA, yA, xB, yB]).round().astype(int)
    return expand_box


def get_box_after_trans(corner_pts, homography):
    trans_pts = cv2.perspectiveTransform(corner_pts, homography)
    xA, yA = np.min(trans_pts[:, :, 0], axis=0), np.min(trans_pts[:, :, 1], axis=0)
    xB, yB = np.max(trans_pts[:, :, 0], axis=0), np.max(trans_pts[:, :, 1], axis=0)
    box = np.array([xA, yA, xB, yB]).reshape(4)
    return trans_pts, box


def get_box_after_masktrans(obj_mask, homography):
    H, W = obj_mask.shape

    trans_mask = np.zeros_like(obj_mask).astype(np.uint8)
    trans_mask[obj_mask==True] = 255
    trans_mask = cv2.warpPerspective(trans_mask, homography, (W, H))
    y_indx, x_indx = np.where(trans_mask == 255)
    
    box = None
    if len(y_indx) > 0 and len(x_indx) > 0:
        yA, xA = np.min(y_indx), np.min(x_indx)
        yB, xB = np.max(y_indx), np.max(x_indx)
        box = np.array([xA, yA, xB, yB]).reshape(-1)
    return box, trans_mask