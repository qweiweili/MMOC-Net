from torch.utils.data import DataLoader, Dataset
import os
import torch
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip, Resize,
                            CropNonEmptyMaskIfExists, RandomCrop)
import pickle
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
from skimage.morphology import disk, dilation
import numpy as np

class Dataset_stage1(Dataset):
    def __init__(self, data_root='data', split_path='CMB_train_val_5folds_211026.pkl', fold=0, mode='train',
                 input_size=(352, 448)):
        self.data_root = data_root
        split_data = pickle.load(open(split_path, 'rb'))
        self.data_path_list = split_data[fold][mode]
        self.len = len(self.data_path_list)
        self.kernel = disk(2)
        if mode == 'train':
            self.transforms = Compose([Resize(height=input_size[0], width=input_size[1]),
                                       ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                                                        rotate_limit=180, p=0.3,
                                                        border_mode=cv2.BORDER_CONSTANT, value=0,
                                                        interpolation=cv2.INTER_NEAREST),
                                       HorizontalFlip(p=0.3), VerticalFlip(p=0.3)])
        else:
            self.transforms = Resize(height=input_size[0], width=input_size[1])

    def __getitem__(self, item):
        pic_path = self.data_path_list[item]
        img = nib.load(os.path.join(self.data_root, 'image', pic_path)).get_data()[:, :, 0]
        label = nib.load(os.path.join(self.data_root, 'mask', pic_path)).get_data()[:, :, 0]
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        augmented = self.transforms(image=img, mask=label)
        img = augmented['image']
        label = augmented['mask'].astype(np.uint8)
        label = dilation(label, self.kernel)
        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)
        return img, label

    def __len__(self):
        return self.len

class Dataset_stage2(Dataset):
    def __init__(self, data_root='data', split_path='CMB_train_val_5folds_211026.pkl', fold=0, mode='train',
                 input_size=(32, 32)):
        self.data_root = data_root
        split_data = pickle.load(open(split_path, 'rb'))
        self.data_path_list = split_data[fold][mode]
        self.len = len(self.data_path_list)
        self.transforms = Compose([CropNonEmptyMaskIfExists(height=input_size[0], width=input_size[1], p=0.5),
                                   RandomCrop(height=input_size[0], width=input_size[1]),
                                   ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2,
                                                    rotate_limit=180, p=0.3,
                                                    border_mode=cv2.BORDER_CONSTANT, value=0,
                                                    interpolation=cv2.INTER_NEAREST),
                                   HorizontalFlip(p=0.3), VerticalFlip(p=0.3)])

    def __getitem__(self, item):
        pic_path = self.data_path_list[item]
        img = nib.load(os.path.join(self.data_root, 'image', pic_path)).get_data()[:, :, 0]
        label = nib.load(os.path.join(self.data_root, 'mask', pic_path)).get_data()[:, :, 0]
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        augmented = self.transforms(image=img, mask=label)
        img = augmented['image']
        label = augmented['mask'].astype(np.uint8)
        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    data = Dataset_stage2(data_root='data', split_path='CMB_train_val_5folds_211026.pkl', fold=0, mode='train',
                 input_size=(64, 64))
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        img = img[0, 0].numpy()
        label = label[0, 0].numpy()
        plt.subplot(121)
        plt.imshow(img, cmap='gray')
        plt.subplot(122)
        plt.imshow(label, cmap='gray')
        plt.show()
        # print(img, label)