import numpy as np
import torch


def cal_kl(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    kld = np.sum(map2 * np.log(map2 / (map1 + eps) + eps))
    return kld


def cal_sim(pred: np.ndarray, gt: np.ndarray, eps=1e-12) -> np.ndarray:
    map1, map2 = pred / (pred.sum() + eps), gt / (gt.sum() + eps)
    intersection = np.minimum(map1, map2)

    return np.sum(intersection)


def image_binary(image, threshold):
    output = np.zeros(image.size).reshape(image.shape)
    for xx in range(image.shape[0]):
        for yy in range(image.shape[1]):
            if (image[xx][yy] > threshold):
                output[xx][yy] = 1
    return output


def cal_nss(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    pred = pred / 255.0
    gt = gt / 255.0
    std = np.std(pred)
    u = np.mean(pred)

    smap = (pred - u) / (std + 1e-12)
    fixation_map = (gt - np.min(gt)) / (np.max(gt) - np.min(gt) + 1e-12)
    fixation_map = image_binary(fixation_map, 0.1)

    nss = smap * fixation_map

    nss = np.sum(nss) / np.sum(fixation_map + 1e-12)

    return nss


def compute_cls_acc(preds, label):
    pred = torch.max(preds, 1)[1]
    # label = torch.max(labels, 1)[1]
    num_correct = (pred == label).sum()
    return float(num_correct) / float(preds.size(0))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.cnt = 0.0

    def update(self, val, n=1.0):
        self.val = val
        self.sum += val * n
        self.cnt += n
        if self.cnt == 0:
            self.avg = 1
        else:
            self.avg = self.sum / self.cnt


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def single_iou(pred, target, index, eps=1e-12):
    assert pred.dim() == 2
    assert pred.shape == target.shape
    assert len(index) == 2

    iou_out = []
    acc_out = []
    fp_out = 0
    num_aff = pred.shape[0]
    for i in range(num_aff):
        if i in index:
            inter = torch.logical_and(pred[i], target[i])
            union = torch.logical_or(pred[i], target[i])
            iou = inter.sum() / (union.sum() + eps)
            iou_out.append(iou)
            acc = inter.sum() / (target[i].sum() + eps)
            acc_out.append(acc)
        else:
            fp_out += (pred[i].sum() / pred.shape[1])

    return iou_out, acc_out, fp_out


def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) + label_pred[mask],
        minlength=n_class ** 2,
    ).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class, ignore_zero=True):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls[1:]) if ignore_zero else np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    valid = hist.sum(axis=1) > 0  # added
    mean_iu = np.nanmean(iu[1:][valid[1:]]) if ignore_zero else np.nanmean(iu[valid])
    cls_iu = dict(zip(range(n_class), iu))
    
    # F-measure
    precision = np.diag(hist) / hist.sum(axis=0)
    recall = np.diag(hist) / hist.sum(axis=1)
    
    f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores)  # Convert NaNs to 0
    mean_f1 = np.nanmean(f1_scores[1:]) if ignore_zero else np.nanmean(f1_scores)
    cls_f1 = dict(zip(range(n_class), f1_scores))

    return {
        "Pixel Accuracy": acc,
        "Mean Accuracy": acc_cls,
        "Mean IoU": mean_iu,
        "Class IoU": cls_iu,
        "F1": mean_f1,
        "Class F1": cls_f1,
    }

def process_seg(sim, gt, td=0.7):
    _, num_aff, w, h = sim.shape
    sim = sim.squeeze(0).flatten(1).cpu().numpy()
    gt = gt.squeeze(0).flatten(1).cpu().numpy()
    max_idx = np.argmax(sim, axis=0)
    bg_idx = np.all((sim < td), axis=0)

    out = np.zeros(sim.shape[-1], dtype=np.int16)
    out_gt = np.zeros(sim.shape[-1], dtype=np.int16)

    for i in range(num_aff):
        out[max_idx == i] = i + 1
        out_gt[gt[i] == 1] = i + 1
    out[bg_idx] = 0
    return out, out_gt
    # sim_bi = (sim > td).astype(np.uint8)
