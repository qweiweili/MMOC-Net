import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import StratifiedKFold

id_all = os.listdir('data/data_all/image')
label_all = [1] * len(id_all)

stratifiedKFolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

split_id = []
for (train_idx, val_idx) in stratifiedKFolds.split(id_all, label_all):
    train_data_one_fold = []
    val_data_one_fold = []
    for i in train_idx:
        train_data_one_fold.append(id_all[i])
    for i in val_idx:
        val_data_one_fold.append(id_all[i])
    split_id.append({'train': train_data_one_fold, 'val': val_data_one_fold})

print(split_id)

split_pic_name = []
for tv_dict in split_id:
    train_data_one_fold = []
    val_data_one_fold = []
    train_id = tv_dict['train']
    val_id = tv_dict['val']
    for id in train_id:
        pic_name = os.listdir(os.path.join('data/data_all/image', id))
        for name in pic_name:
            train_data_one_fold.append('{}/{}'.format(id, name))
    for id in val_id:
        pic_name = os.listdir(os.path.join('data/data_all/image', id))
        for name in pic_name:
            val_data_one_fold.append('{}/{}'.format(id, name))
    split_pic_name.append({'train': train_data_one_fold, 'val': val_data_one_fold})
print(split_pic_name)

for i, data_one_fold in enumerate(split_pic_name):
    print(i, len(data_one_fold['train']), len(data_one_fold['val']))

with open(r'CMB_train_val_5folds_211110.pkl', 'wb') as f:
    pickle.dump(split_pic_name, f, pickle.HIGHEST_PROTOCOL)

# for id in os.listdir('data/image'):
#
#     for pic_name in os.listdir(os.path.join('data/image', id)):
#         img = nib.load(os.path.join('data/image', id, pic_name)).get_data()
#         label = nib.load(os.path.join('data/mask', id, pic_name)).get_data()
#         print(id, pic_name)
#         print(img.shape)