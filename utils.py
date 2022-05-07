import numpy as np
import cv2
from skimage.measure import *

def fit_Ellipse(mask):
    Ellipse_mask = np.zeros(mask.shape)
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        for c, contour in enumerate(contours):
            if len(contour) > 5:
                ellipse = cv2.fitEllipse(contour)
                cv2.ellipse(Ellipse_mask, ellipse, 1, -1)
            else:
                cv2.drawContours(Ellipse_mask, contours, c, 1, -1)
    return Ellipse_mask

def find_mask_centroid(mask):
    connect_regions = label(mask.astype(np.uint8), connectivity=1, background=0)
    pprops = regionprops(connect_regions)
    mask_certers = []
    for m in pprops:
        mask_certers.append(m.centroid)
    return mask_certers

def centroid_crop(img_size, centroid, crop_size):
    crop_x_half = crop_size[0] // 2
    crop_y_half = crop_size[1] // 2
    x1 = 0 if int(centroid[0]) - crop_x_half <= 0 else int(centroid[0]) - crop_x_half
    x2 = img_size[0] if int(centroid[0]) + crop_x_half >= img_size[0] else int(centroid[0]) + crop_x_half
    y1 = 0 if int(centroid[1]) - crop_y_half <= 0 else int(centroid[1]) - crop_y_half
    y2 = img_size[1] if int(centroid[1]) + crop_y_half >= img_size[1] else int(centroid[1]) + crop_y_half
    return x1, x2, y1, y2

def compute_dice(pred, label):
    intersection = pred * label
    dice_sco = (2 * intersection.sum()) / (pred.sum() + label.sum())
    return dice_sco

def compute_iou(pred, label):
    intersection = pred * label
    iou_sco = intersection.sum() / (pred.sum() + label.sum() - intersection.sum())
    return iou_sco

def False_positives(pred, label):
    intersection = pred * label
    FP = pred.sum() - intersection.sum()
    return FP

def False_negatives(pred, label):
    intersection = pred * label
    FN = label.sum() - intersection.sum()
    return FN