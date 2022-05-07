from torch.utils.data import DataLoader, Dataset
import os
import torch
from albumentations import (ShiftScaleRotate, Compose, HorizontalFlip, VerticalFlip, Resize)
import pickle
import cv2
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np

class Dataset_CMB(Dataset):
    def __init__(self, data_root='data', split_path='CMB_train_val_5folds_211026.pkl', fold=0, mode='train',
                 input_size=(448, 352)):
        self.data_root = data_root
        split_data = pickle.load(open(split_path, 'rb'))
        self.data_path_list = split_data[fold][mode]
        self.len = len(self.data_path_list)
        if mode == 'train':
            self.transforms = Compose([Resize(height=input_size[0], width=input_size[1]),
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
        img = torch.from_numpy(img).unsqueeze(0)
        label = torch.from_numpy(label).unsqueeze(0)
        return img, label

    def __len__(self):
        return self.len

if __name__ == '__main__':
    data = Dataset_CMB(data_root='data', split_path='CMB_train_val_5folds_211026.pkl', fold=0, mode='train',
                 input_size=(448, 352))
    dataloader = DataLoader(dataset=data, batch_size=2, shuffle=True)
    for i, (img, label) in enumerate(dataloader):
        img = img[0, 0].numpy()
        label = label[0, 0].numpy()
        # plt.subplot(121)
        # plt.imshow(img, cmap='gray')
        # plt.subplot(122)
        # plt.imshow(label, cmap='gray')
        # plt.show()
        print(img, label)