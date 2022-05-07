import os
import nibabel as nib
from skimage.measure import *
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

Sensitivitys=[]
aucs=[]
mean_precision = np.linspace(0, 1, 100)
data_dir = r'C:\Users\Administrator\Desktop\CMB\trained_models\stage2_size16'
for fold_dir in os.listdir(data_dir):
    precision = np.load(os.path.join(data_dir, fold_dir, 'best_precision_recall_dilate0_fitellipse0_threshold50', 'precision.npy'))
    Sensitivity = np.load(os.path.join(data_dir, fold_dir, 'best_precision_recall_dilate0_fitellipse0_threshold50', 'recall.npy'))
    Sensitivitys.append(np.interp(mean_precision, precision, Sensitivity))
    Sensitivitys[-1][-1] = 0.0
mean_Sensitivitys = np.mean(Sensitivitys, axis=0)
# print(mean_precision)
# print(mean_Sensitivitys)
# plt.plot(mean_precision, mean_Sensitivitys, color='y', label=r'patch16', lw=2, alpha=.8)

f = interp1d(mean_Sensitivitys, mean_precision, kind = 'cubic')
print(f(np.array(0.9)))