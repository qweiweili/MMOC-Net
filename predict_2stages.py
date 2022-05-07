import os
import numpy as np
import torch
import argparse
import shutil
import nibabel as nib
import segmentation_models_pytorch as smp
import pickle
from sklearn.metrics import recall_score, f1_score, precision_score, accuracy_score
import cv2
from Network import Full_Resolution_Net, Full_Resolution_Net_without_ASPP
from skimage.morphology import disk, dilation
from utils import fit_Ellipse, find_mask_centroid, centroid_crop, compute_dice, compute_iou, \
    False_positives, False_negatives
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='which gpu is used')
parser.add_argument('--stage1_model_dir', type=str, help='trained model dir')
parser.add_argument('--stage2_model_dir', type=str, help='trained model dir')
parser.add_argument('--stage2_size', type=int, default=64, help='stage2_size')
parser.add_argument('--dilate', type=int, default=0, help='dilate')
parser.add_argument('--net2', type=str, help='net')
parser.add_argument('--threshold', type=float, default=0, help='threshold')
parser.add_argument('--fit_ellipse', type=int, default=0, help='fit_ellipse')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--model_mode', type=str, default='best_MAE', help='best_MAE or best_dice')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

data_dir = 'data'
split_path = 'CMB_train_val_5folds_211110.pkl'
stage1_size = (352, 448)
stage2_size = (args.stage2_size, args.stage2_size)
split_data = pickle.load(open(split_path, 'rb'))
data_path_list = split_data[args.fold]['val']

new_dir = os.path.join(args.stage2_model_dir, '{}_dilate{}_fitellipse{}_threshold{}'.format(
    args.model_mode, args.dilate, args.fit_ellipse, int(args.threshold * 100)))
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

stage1_net = smp.Unet(in_channels=1, classes=1, encoder_name='resnet34').cuda()
if args.net2.lower() == 'frn':
    stage2_net = Full_Resolution_Net(in_channel=1, classes=1).cuda()
elif args.net2.lower() == 'frn_without_aspp':
    stage2_net = Full_Resolution_Net_without_ASPP(in_channel=1, classes=1).cuda()

stage1_net.load_state_dict(torch.load(os.path.join(args.stage1_model_dir, '{}.pth'.format(args.model_mode))))
stage1_net.eval()
stage2_net.load_state_dict(torch.load(os.path.join(args.stage2_model_dir, '{}.pth'.format(args.model_mode))))
stage2_net.eval()

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

        img_resize = cv2.resize(img, (stage1_size[1], stage1_size[0]))
        img_resize = torch.from_numpy(img_resize).unsqueeze(0).unsqueeze(0).float().cuda()
        pred_stage1 = torch.sigmoid(stage1_net(img_resize))
        pred_stage1 = pred_stage1[0, 0].cpu().numpy()
        pred_stage1 = cv2.resize(pred_stage1, (original_size[1], original_size[0]))
        pred_stage1[pred_stage1 <= args.threshold] = 0
        pred_stage1[pred_stage1 > args.threshold] = 1

        centroids = find_mask_centroid(pred_stage1)
        final_pred = np.zeros(original_size)
        pred_logits = np.zeros(original_size)
        for c in centroids:
            x1, x2, y1, y2 = centroid_crop(original_size, c, stage2_size)
            patch = img[x1: x2, y1: y2]
            patch = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).float().cuda()
            pred_stage2 = torch.sigmoid(stage2_net(patch))
            pred_stage2 = pred_stage2[0, 0].cpu().numpy()
            pred_logits[x1: x2, y1: y2] = pred_stage2
            pred_stage2[pred_stage2 <= 0.5] = 0
            pred_stage2[pred_stage2 > 0.5] = 1
            final_pred[x1: x2, y1: y2] = pred_stage2
        if args.dilate > 0:
            final_pred = dilation(final_pred, kernel)
        if args.fit_ellipse > 0:
            final_pred = fit_Ellipse(final_pred)

        precision_one = precision_score(label.reshape(-1), final_pred.reshape(-1))
        recall_one = recall_score(label.reshape(-1), final_pred.reshape(-1))
        acc_one = accuracy_score(label.reshape(-1), final_pred.reshape(-1))
        dice_one = compute_dice(final_pred, label)
        iou_one = compute_iou(final_pred, label)
        FP_one = False_positives(final_pred, label)
        FN_one = False_negatives(final_pred, label)

        pred_nii = nib.Nifti1Image(final_pred[:, :, np.newaxis], affine=label_nii.affine, header=label_nii.header)
        os.makedirs(os.path.join(new_dir, 'predictions', img_path.split('/')[0]), exist_ok=True)
        nib.save(pred_nii, os.path.join(new_dir, 'predictions', img_path))

        pred_nii = nib.Nifti1Image(pred_logits[:, :, np.newaxis], affine=label_nii.affine, header=label_nii.header)
        os.makedirs(os.path.join(new_dir, 'predictions_logits', img_path.split('/')[0]), exist_ok=True)
        nib.save(pred_nii, os.path.join(new_dir, 'predictions_logits', img_path))

        print('{}: {} {} {} {} {} {} {}'.format(img_path, precision_one, recall_one, acc_one, dice_one,
                                                iou_one, FP_one, FN_one))
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