import cv2
import os
import random
import torch
import numpy as np
from PIL import Image
from matplotlib import cm


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_gt(args):
    assert args.divide in ["Seen", "Unseen"], "The divide argument should be Seen or Unseen"
    files = os.listdir(args.mask_root)
    dict_1 = {}
    for file in files:
        file_path = os.path.join(args.mask_root, file)
        objs = os.listdir(file_path)
        for obj in objs:
            obj_path = os.path.join(file_path, obj)
            images = os.listdir(obj_path)
            for img in images:
                img_path = os.path.join(obj_path, img)
                mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                key = file + "_" + obj + "_" + img
                dict_1[key] = mask

    torch.save(dict_1, args.divide + "_gt.t7")


def normalize_minmax(cams, eps=1e-15):
    B, _, _ = cams.shape
    min_value, _ = cams.view(B, -1).min(1)
    cams_minmax = cams - min_value.view(B, 1, 1)
    max_value, _ = cams_minmax.view(B, -1).max(1)
    cams_minmax /= max_value.view(B, 1, 1) + eps
    return cams_minmax


def normalize_map(atten_map, crop_size):
    atten_map = cv2.resize(atten_map, dsize=(crop_size, crop_size))
    min_val = np.min(atten_map)
    max_val = np.max(atten_map)
    atten_norm = (atten_map - min_val) / (max_val - min_val + 1e-10)
    return atten_norm


def get_optimizer(model, logger, args):
    lr = args.lr
    weight_list = []
    ft_list = []
    for name, value in model.named_parameters():
        if value.requires_grad:
            logger.info("Train layer:" + name)
            # if 'lora' in name:
                # ft_list.append(value)
            # else:
            weight_list.append(value)
    logger.info("Optimizer: AdamW")
    optimizer = torch.optim.AdamW([{'params': weight_list,
                                    'lr': lr},
                                #    {'params': ft_list,
                                #    'lr': lr * 0.1}
                                   ], weight_decay=args.weight_decay)
    # logger.info("Optimizer: SGD")
    # optimizer = torch.optim.SGD([{'params': weight_list,
                                #   'lr': lr},
                                #  {'params': ft_list,
                                #   'lr': lr * 0.01},
                                #  ],
                                # momentum=args.momentum,
                                # weight_decay=args.weight_decay,
                                # nesterov=True)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 12], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.iters)
    return optimizer, scheduler


def overlay_mask(img: Image.Image, mask: Image.Image, colormap: str = "jet", alpha: float = 0.7) -> Image.Image:
    if not isinstance(img, Image.Image) or not isinstance(mask, Image.Image):
        raise TypeError("img and mask arguments need to be PIL.Image")

    if not isinstance(alpha, float) or alpha < 0 or alpha >= 1:
        raise ValueError("alpha argument is expected to be of type float between 0 and 1")

    cmap = cm.get_cmap(colormap)
    # Resize mask and apply colormap
    overlay = mask.resize(img.size, resample=Image.BICUBIC)
    overlay = (255 * cmap(np.asarray(overlay) ** 2)[:, :, :3]).astype(np.uint8)
    # Overlay the image with the mask
    overlayed_img = Image.fromarray((alpha * np.asarray(img) + (1 - alpha) * overlay).astype(np.uint8))

    return overlayed_img


def _freeze_stages(model, exclude_key=None):
    """Freeze stages param and norm stats."""
    for n, m in model.named_parameters():
        if exclude_key:
            if isinstance(exclude_key, str):
                if not exclude_key in n:
                    m.requires_grad = False
            elif isinstance(exclude_key, list):
                count = 0
                for i in range(len(exclude_key)):
                    i_layer = str(exclude_key[i])
                    if i_layer in n:
                        count += 1
                if count == 0:
                    m.requires_grad = False
                elif count > 0:
                    print('Finetune layer in backbone:', n)
            else:
                assert AttributeError("Dont support the type of exclude_key!")
        else:
            m.requires_grad = False
