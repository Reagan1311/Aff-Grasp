import os
from os.path import join as opj
import torch
import torch.nn.functional as F
import random
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torchvision.transforms.functional as TF

AFF_LIST = ['grasp', 'cut', 'scoop', 'pound', 'support', 'screw', 'contain', 'stick']
OBJ_LIST = ['fork', 'hammer', 'knife', 'scissors', 'pan', 
            'cup', 'spoon', 'ladle', 'spatula', 'shovel', 'trowel', 'screwdriver']
AFF2OBJ_dict = {'cut':['knife', 'scissors'], 'scoop':['spoon', 'ladle'], 'stick': ['fork'], 
                'pound':['hammer'], 'support':['spatula', 'shovel', 'trowel'], 
                'screw':['screwdriver'], 'contain':['pan', 'cup']}


class TrainData(data.Dataset):
    def __init__(self, data_root, resize_size=476, crop_size=448, depth_gray=True):
        self.resize_size = resize_size
        self.crop_size = crop_size
        self.data_root = os.path.join(data_root, 'ego_train')
        self.data_list = []
        for i in os.listdir(self.data_root):
            if i.endswith('-img.jpg'):
                if depth_gray:
                    dep_file = i.replace('-img.jpg', '-img_graydepth.png')
                else:
                    dep_file = i.replace('-img.jpg', '-img_depth.png')
                label_file = i.replace('-img.jpg', '-label.png')
                
                if os.path.exists(opj(os.path.dirname(data_root), 'depth', dep_file)) and \
                    os.path.exists(opj(data_root, label_file)):                        
                        self.data_list.append((i, dep_file, label_file))
    
    def __getitem__(self, item):
        img_file, dep_file, label_file = self.data_list[item]
        noun = img_file.split('_')[0]
        if '-' in noun:
            noun = noun.split('-')[0]
        img = Image.open(opj(self.data_root, img_file)).convert('RGB')
        depth = Image.open(opj(os.path.dirname(self.data_root), 'depth', dep_file)).convert('RGB')
        ann = Image.open(opj(self.data_root, label_file))
        
        img, depth, ann = self.transform(img, depth, ann)
        
        ann = self.assign_afflabel(noun, ann)
        oh_ann = F.one_hot(ann.to(torch.int64), num_classes=(len(AFF_LIST) + 1))
        oh_ann = oh_ann.permute(2, 0, 1)

        return img, depth, oh_ann

    
    def assign_afflabel(self, noun, ann):
        ann[ann==128] = AFF_LIST.index('grasp') + 1
        for k, v in AFF2OBJ_dict.items():
            if noun in v:
                ann[ann==255] = AFF_LIST.index(k) + 1
        return ann
        
    
    def transform(self, img, depth, mask):
        resize = transforms.Resize(size=(self.resize_size, self.resize_size), antialias=None)
        mask_resize = transforms.Resize(size=(self.resize_size, self.resize_size), 
                                        interpolation=Image.NEAREST, 
                                        antialias=None)
        img, depth = resize(img), resize(depth)
        mask = mask_resize(mask)
        
        # angle = random.uniform(-30, 30)
        # img, depth = transforms.functional.rotate(img, angle), transforms.functional.rotate(depth, angle)
        
        i, j, h, w = transforms.RandomCrop.get_params(
            img, output_size=(self.crop_size, self.crop_size))
        img = TF.crop(img, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        if random.random() > 0.5:
            img = TF.hflip(img)
            depth = TF.hflip(depth)
            mask = TF.hflip(mask)
        
        if random.random() > 0.5:
            img = TF.vflip(img)
            depth = TF.vflip(depth)
            mask = TF.vflip(mask)

        img = TF.to_tensor(img)
        depth = TF.to_tensor(depth)
        mask = torch.from_numpy(np.array(mask))

        img = TF.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        depth = TF.normalize(depth, mean=(0.5, 0.5 ,0.5), std=(0.5, 0.5 ,0.5))
        
        return img, depth, mask

    def __len__(self):
        return len(self.data_list)


class TestData(data.Dataset):
    def __init__(self, data_root, crop_size=224, test_dir='test'):
        data_root = opj(data_root, test_dir)

        self.data_root = opj(data_root, 'JPEGImages')
        self.dep_root = opj(data_root, 'depth/depth_gray')
        self.ann_root = opj(data_root, 'SegmentationClassNpy')
        self.crop_size = crop_size
        self.img_list = os.listdir(self.data_root)

        self.transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])
        
        
        self.dep_transform = transforms.Compose([
            transforms.Resize((crop_size, crop_size), antialias=None),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5 ,0.5),
                                 std=(0.5, 0.5 ,0.5))])

    def __getitem__(self, item):
        img_name = self.img_list[item]
        dep_name = img_name.replace('.jpg', '_graydepth.png')
        img = Image.open(opj(self.data_root, img_name)).convert('RGB')
        ori_size = img.size
        img = self.transform(img)
    
        ann_file = opj(self.ann_root, img_name.replace('jpg', 'npy'))
        if os.path.exists(ann_file):
            depth = Image.open(opj(self.dep_root, dep_name)).convert('RGB')
            depth = self.dep_transform(depth)
            ann = torch.from_numpy(np.load(ann_file))
            ann = F.interpolate(ann[None, None, ...].float(), size=(self.crop_size, self.crop_size), mode='nearest').squeeze()
            ann = F.one_hot(ann.to(torch.int64), num_classes=(len(AFF_LIST) + 1))
            ann = ann.permute(2, 0, 1)

        else:
            depth = torch.zeros(3, self.crop_size, self.crop_size)
            ann = torch.zeros((len(AFF_LIST), self.crop_size, self.crop_size))
        
        obj_name = img_name.rsplit('_', 1)[0]
        return img, depth, ann, obj_name, ori_size, img_name

    def __len__(self):
        return len(self.img_list)
