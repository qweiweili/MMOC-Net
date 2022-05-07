import os
import numpy as np
import torch
import argparse
import shutil
import nibabel as nib
import segmentation_models_pytorch as smp
import pickle
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
from utils import fit_Ellipse, find_mask_centroid, centroid_crop, compute_dice, compute_iou, \
    False_positives, False_negatives
import cv2
from skimage.morphology import disk, dilation
from utils import fit_Ellipse
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--net', type=str, default='baseline', help='net')
parser.add_argument('--model_dir', type=str, help='trained model dir')
parser.add_argument('--dilate', type=int, default=0, help='dilate')
parser.add_argument('--fit_ellipse', type=int, default=0, help='fit_ellipse')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--model_mode', type=str, default='best_MAE', help='best_MAE or best_dice')
parser.add_argument('--gpus_training', type=bool, default=False, help='whether Multiple GPUs training')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_dir = 'data'
split_path = 'CMB_train_val_5folds_211110.pkl'
input_size = (352, 448)
split_data = pickle.load(open(split_path, 'rb'))
data_path_list = split_data[args.fold]['val']

new_dir = os.path.join(args.model_dir, '{}_dilate{}_fitellipse{}'.format(args.model_mode, args.dilate, args.fit_ellipse))
if os.path.exists(new_dir):
    shutil.rmtree(new_dir)
print(new_dir)
os.makedirs(os.path.join(new_dir, 'predictions'), exist_ok=True)
os.makedirs(os.path.join(new_dir, 'predictions_logits'), exist_ok=True)
name_all = []
recall_all = []
precision_all = []
dice_all = []
iou_all = []
acc_all = []
FP_all = []
FN_all = []

if args.net.lower() == 'unet_resnet34':
    net = smp.Unet(in_channels=1, classes=1, encoder_name='resnet34').cuda()

if args.gpus_training:
    net = torch.nn.DataParallel(net)

net.load_state_dict(torch.load(os.path.join(args.model_dir, '{}.pth'.format(args.model_mode))))
net.eval()

kernel = disk(1)

with torch.no_grad():
    for i, img_path in enumerate(data_path_list):
        print('\n', i, img_path)
        img = nib.load(os.path.join(data_dir, 'image', img_path)).get_data()[:, :, 0]
        label_nii = nib.load(os.path.join(data_dir, 'mask', img_path))
        label = label_nii.get_data()[:, :, 0]
        img_min = img.min()
        img_max = img.max()
        img = (img - img_min) / (img_max - img_min)
        original_size = img.shape

        img = cv2.resize(img, (input_size[1], input_size[0]))
        img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).float().cuda()
        pred = torch.sigmoid(net(img))
        pred = pred[0, 0].cpu().numpy()
        pred = cv2.resize(pred, (original_size[1], original_size[0]))
        pred_logits = np.copy(pred)
        pred[pred <= 0.5] = 0
        pred[pred > 0.5] = 1
        if args.dilate > 0:
            pred = dilation(pred, kernel)
        if args.fit_ellipse > 0:
            pred = fit_Ellipse(pred)

        precision_one = precision_score(label.reshape(-1), pred.reshape(-1))
        recall_one = recall_score(label.reshape(-1), pred.reshape(-1))
        acc_one = accuracy_score(label.reshape(-1), pred.reshape(-1))
        dice_one = compute_dice(pred, label)
        iou_one = compute_iou(pred, label)
        FP_one = False_positives(pred, label)
        FN_one = False_negatives(pred, label)

        pred_nii = nib.Nifti1Image(pred[:, :, np.newaxis], affine=label_nii.affine, header=label_nii.header)
        os.makedirs(os.path.join(new_dir, 'predictions', img_path.split('/')[0]), exist_ok=True)
        nib.save(pred_nii, os.path.join(new_dir, 'predictions', img_path))

        pred_nii = nib.Nifti1Image(pred_logits[:, :, np.newaxis], affine=label_nii.affine, header=label_nii.header)
        os.makedirs(os.path.join(new_dir, 'predictions_logits', img_path.split('/')[0]), exist_ok=True)
        nib.save(pred_nii, os.path.join(new_dir, 'predictions_logits', img_path))

        print('{}: {} {}'.format(img_path, precision_one, recall_one))
        name_all.append(img_path)
        recall_all.append(recall_one)
        precision_all.append(precision_one)
        acc_all.append(acc_one)
        dice_all.append(dice_one)
        iou_all.append(iou_one)
        FP_all.append(FP_one)
        FN_all.append(FN_one)
recall_mean = np.mean(recall_all)
precision_mean = np.mean(precision_all)
acc_mean = np.mean(acc_all)
dice_mean = np.mean(dice_all)
iou_mean = np.mean(iou_all)
FP_mean = np.mean(FP_all)
FN_mean = np.mean(FN_all)
recall_std = np.std(recall_all)
precision_std = np.std(precision_all)
acc_std = np.std(acc_all)
dice_std = np.std(dice_all)
iou_std = np.std(iou_all)
FP_std = np.std(FP_all)
FN_std = np.std(FN_all)
print('recall: {}   precision: {}   acc: {}   dice: {}   iou: {}   FP: {}   FN: {}'.format(
    recall_mean, precision_mean, acc_mean, dice_mean, iou_mean, FP_mean, FN_mean))
name_all.append('mean')
recall_all.append(recall_mean)
precision_all.append(precision_mean)
acc_all.append(acc_mean)
dice_all.append(dice_mean)
iou_all.append(iou_mean)
FP_all.append(FP_mean)
FN_all.append(FN_mean)
name_all.append('std')
recall_all.append(recall_std)
precision_all.append(precision_std)
acc_all.append(acc_std)
dice_all.append(dice_std)
iou_all.append(iou_std)
FP_all.append(FP_std)
FN_all.append(FN_std)
frame = pd.DataFrame({'img': name_all, 'precision': precision_all, 'recall': recall_all, 'acc': acc_all,
                      'dice': dice_all, 'iou': iou_all, 'FP': FP_all, 'FN': FN_all})
frame.to_csv(os.path.join(new_dir, 'metrics.csv'), index=False)