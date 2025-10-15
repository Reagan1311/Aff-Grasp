import os
import torch
import numpy as np
from PIL import Image
from utils.util import normalize_map, overlay_mask
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm
import cv2


# visualize the prediction of the first batch
def viz_pred_train(args, ego, exo, masks, aff_list, aff_label, epoch, step):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=ego.dtype, device=ego.device).view(-1, 1, 1)

    ego_0 = ego[0].squeeze(0) * std + mean
    ego_0 = ego_0.detach().cpu().numpy() * 255
    ego_0 = Image.fromarray(ego_0.transpose(1, 2, 0).astype(np.uint8))

    exo_img = []
    num_exo = exo.shape[1]
    for i in range(num_exo):
        name = 'exo_' + str(i)
        locals()[name] = exo[0][i].squeeze(0) * std + mean
        locals()[name] = locals()[name].detach().cpu().numpy() * 255
        locals()[name] = Image.fromarray(locals()[name].transpose(1, 2, 0).astype(np.uint8))
        exo_img.append(locals()[name])

    exo_cam = masks['exo_aff'][0]

    sim_maps, exo_sim_maps, part_score, ego_pred = masks['pred']
    num_clu = sim_maps.shape[1]
    part_score = np.array(part_score[0].squeeze().data.cpu())

    ego_pred = np.array(ego_pred[0].squeeze().data.cpu())
    ego_pred = normalize_map(ego_pred, args.crop_size)
    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(ego_0, ego_pred, alpha=0.5)

    ego_sam = masks['ego_sam']
    ego_sam = np.array(ego_sam[0].squeeze().data.cpu())
    ego_sam = normalize_map(ego_sam, args.crop_size)
    ego_sam = Image.fromarray(ego_sam)
    ego_sam = overlay_mask(ego_0, ego_sam, alpha=0.1)

    aff_str = aff_list[aff_label[0].item()]

    for i in range(num_exo):
        name = 'exo_aff' + str(i)
        locals()[name] = np.array(exo_cam[i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(exo_img[i], locals()[name], alpha=0.5)

    for i in range(num_clu):
        name = 'sim_map' + str(i)
        locals()[name] = np.array(sim_maps[0][i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(ego_0, locals()[name], alpha=0.5)

        # Similarity maps for the first exocentric image
        name = 'exo_sim_map' + str(i)
        locals()[name] = np.array(exo_sim_maps[0, 0][i].squeeze().data.cpu())
        locals()[name] = normalize_map(locals()[name], args.crop_size)
        locals()[name] = Image.fromarray(locals()[name])
        locals()[name] = overlay_mask(locals()['exo_' + str(0)], locals()[name], alpha=0.5)

    # Exo&Ego plots
    fig, ax = plt.subplots(4, max(num_clu, num_exo), figsize=(8, 8))
    for axi in ax.ravel():
        axi.set_axis_off()
    for k in range(num_exo):
        ax[0, k].imshow(eval('exo_aff' + str(k)))
        ax[0, k].set_title("exo_" + aff_str)
    for k in range(num_clu):
        ax[1, k].imshow(eval('sim_map' + str(k)))
        ax[1, k].set_title('PartIoU_' + str(round(part_score[k], 2)))
        ax[2, k].imshow(eval('exo_sim_map' + str(k)))
        ax[2, k].set_title('sim_map_' + str(k))
    ax[3, 0].imshow(ego_pred)
    ax[3, 0].set_title(aff_str)
    ax[3, 1].imshow(ego_sam)
    ax[3, 1].set_title('Saliency')

    os.makedirs(os.path.join(args.save_path, 'viz_train'), exist_ok=True)
    fig_name = os.path.join(args.save_path, 'viz_train', 'cam_' + str(epoch) + '_' + str(step) + '.jpg')
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close()


def viz_cls_mask(args, image, cls_mask, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    cls_mask = np.array(cls_mask.squeeze().data.cpu())
    cmap = cm.get_cmap('viridis')

    # Resize mask and apply colormap
    cls_mask0 = normalize_map(cls_mask[0], args.crop_size)
    cls_mask0 = Image.fromarray(cls_mask0)
    cls_mask0 = cls_mask0.resize(img.size, resample=Image.BICUBIC)
    cls_mask0 = (255 * cmap(np.asarray(cls_mask0) ** 2)[:, :, :3]).astype(np.uint8)

    cls_mask1 = normalize_map(cls_mask[1], args.crop_size)
    cls_mask1 = Image.fromarray(cls_mask1)
    cls_mask1 = cls_mask1.resize(img.size, resample=Image.BICUBIC)
    cls_mask1 = (255 * cmap(np.asarray(cls_mask1) ** 2)[:, :, :3]).astype(np.uint8)

    # cls_mask2 = normalize_map(cls_mask[2], args.crop_size)
    # cls_mask2 = Image.fromarray(cls_mask2)
    # cls_mask2 = cls_mask2.resize(img.size, resample=Image.BICUBIC)
    # cls_mask2 = (255 * cmap(np.asarray(cls_mask2) ** 2)[:, :, :3]).astype(np.uint8)

    fig, ax = plt.subplots(1, 3, figsize=(9, 4))
    for axi in ax.ravel():
        axi.set_axis_off()
    ax[0].imshow(img)
    ax[0].set_title('ego')
    ax[1].imshow(cls_mask0)
    ax[1].set_title('cls_mask0')
    ax[2].imshow(cls_mask1)
    ax[2].set_title('cls_mask1')
    # ax[3].imshow(cls_mask2)
    # ax[3].set_title('cls_mask2')

    os.makedirs(os.path.join(args.save_path, 'viz_cls_mask'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_cls_mask', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_cls_mask', img_name + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()


def viz_pred_gray(args, ego_pred, img_name):

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)

    fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')

    # print(ego_pred, ego_pred.max(), ego_pred.shape)
    # exit()
    cv2.imwrite(fig_name, ego_pred * 255)
    # plt.savefig(fig_name)
    # plt.close()


def viz_pred_test(args, image, ego_pred, GT_mask, aff_list, aff_label, img_name, epoch=None):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image.squeeze(0) * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = Image.fromarray(GT_mask)
    gt_result = overlay_mask(img, gt, alpha=0.5)
    aff_str = aff_list[aff_label.item()]

    os.makedirs(os.path.join(args.save_path, 'viz_gray'), exist_ok=True)
    gray_name = os.path.join(args.save_path, 'viz_gray', img_name + '.jpg')
    cv2.imwrite(gray_name, ego_pred * 255)

    ego_pred = Image.fromarray(ego_pred)
    ego_pred = overlay_mask(img, ego_pred, alpha=0.5)

    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    for axi in ax.ravel():
        axi.set_axis_off()
    # ax[0].imshow(img)
    # ax[0].set_title('ego')
    ax[0].imshow(ego_pred)
    ax[0].set_title(aff_str)
    ax[1].imshow(gt_result)
    ax[1].set_title('GT')

    # egos_ol = []
    # for e in ego_pred:
    #     e = normalize_map(e, 224)
    #     e = Image.fromarray(e)
    #     egos_ol.append(overlay_mask(img, e, alpha=0.5))
    #
    # fig, axes = plt.subplots(5, 5, figsize=(12, 10))
    # axes = axes.flatten()
    # for i, ax in enumerate(axes):
    #     if i < len(egos_ol):
    #         ax.imshow(egos_ol[i])
    #         ax.set_title(f"{i + 1}")
    #     ax.axis('off')  # Turn off the axis labels
    #
    # for i in range(len(egos_ol), len(axes)):
    #     fig.delaxes(axes[i])

    os.makedirs(os.path.join(args.save_path, 'viz_test'), exist_ok=True)
    if epoch:
        fig_name = os.path.join(args.save_path, 'viz_test', "epoch" + str(epoch) + '_' + img_name + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_test', img_name + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()


def viz_pred_os(args, image, pred, GT_mask, aff_list, iter, phase='train'):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image[0] * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = GT_mask[0]
    # gt = Image.fromarray(GT_mask[0])

    for i in range(len(aff_list)):
        name = 'gt' + str(i)
        pred_name = 'pred' + str(i)
        locals()[name] = Image.fromarray(gt[i].data.cpu().numpy())
        locals()[name] = overlay_mask(img, locals()[name], alpha=0.5)

        locals()[pred_name] = np.array(pred[0][i].squeeze().data.cpu())
        # locals()[pred_name] = cv2.resize(locals()[pred_name], dsize=(args.crop_size, args.crop_size))

        # locals()[pred_name] = (locals()[pred_name] * 255).astype('uint8')
        locals()[pred_name] = normalize_map(locals()[pred_name], args.crop_size)
        locals()[pred_name] = Image.fromarray(locals()[pred_name])
        locals()[pred_name] = overlay_mask(img, locals()[pred_name], alpha=0.5)

    fig, ax = plt.subplots(2, len(aff_list), figsize=(10, 4))
    for i, axi in enumerate(ax.ravel()):
        axi.set_axis_off()
        if i < len(aff_list):
            axi.imshow(eval('pred' + str(i)))
            axi.set_title(aff_list[i])
        else:
            axi.imshow(eval('gt' + str(i-len(aff_list))))
            axi.set_title(aff_list[i-len(aff_list)])

    os.makedirs(os.path.join(args.save_path, 'viz_' + phase), exist_ok=True)
    if iter is not None:
        fig_name = os.path.join(args.save_path, 'viz_' + phase, "iter" + str(iter) + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_' + phase + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()


def viz_pred_os(args, image, pred, GT_mask, aff_list, iter, phase='train'):
    mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    std = torch.as_tensor([0.229, 0.224, 0.225], dtype=image.dtype, device=image.device).view(-1, 1, 1)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img = image[0] * std + mean
    img = img.detach().cpu().numpy() * 255
    img = Image.fromarray(img.transpose(1, 2, 0).astype(np.uint8))

    gt = GT_mask[0]
    # gt = Image.fromarray(GT_mask[0])

    for i in range(len(aff_list)):
        name = 'gt' + str(i)
        pred_name = 'pred' + str(i)
        locals()[name] = Image.fromarray(gt[i].data.cpu().numpy())
        locals()[name] = overlay_mask(img, locals()[name], alpha=0.5)

        locals()[pred_name] = np.array(pred[0][i].squeeze().data.cpu())
        # locals()[pred_name] = cv2.resize(locals()[pred_name], dsize=(args.crop_size, args.crop_size))

        # locals()[pred_name] = (locals()[pred_name] * 255).astype('uint8')
        locals()[pred_name] = normalize_map(locals()[pred_name], args.crop_size)
        locals()[pred_name] = Image.fromarray(locals()[pred_name])
        locals()[pred_name] = overlay_mask(img, locals()[pred_name], alpha=0.5)
            
    fig, ax = plt.subplots(2, len(aff_list) + 1, figsize=(200, 40))
    for i, axi in enumerate(ax.ravel()):
        axi.set_axis_off()
        if i == 0 or i == len(aff_list) + 1:
            axi.imshow(img)
            axi.set_title('RGB')
        elif i < len(aff_list) + 1:
            axi.imshow(eval('pred' + str(i-1)))
            axi.set_title(aff_list[i-1])
        else:
            axi.imshow(eval('gt' + str(i-len(aff_list)-2)))
            axi.set_title(aff_list[i-len(aff_list)-2])


    os.makedirs(os.path.join(args.save_path, 'viz_' + phase), exist_ok=True)
    if iter is not None:
        fig_name = os.path.join(args.save_path, 'viz_' + phase, "iter" + str(iter) + '.jpg')
    else:
        fig_name = os.path.join(args.save_path, 'viz_' + phase + '.jpg')
    plt.savefig(fig_name, bbox_inches='tight', dpi=300)
    plt.close()
