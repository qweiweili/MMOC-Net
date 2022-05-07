import os
from dataset_2stages import Dataset_stage2
from torch.utils.data import DataLoader
import torch.optim as optim
from tensorboardX import SummaryWriter
import torch.nn as nn
import numpy as np
import torch
import math
from loss_function.DICE import DiceLoss, recall_Loss
import shutil
from sklearn.metrics import recall_score, precision_score
from Network import Full_Resolution_Net_without_ASPP
from torchvision.utils import make_grid
import argparse
import random

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='which gpu is used')
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--name', type=str, default='stage1', help='stage')
parser.add_argument('--stage2_size', type=int, default=64, help='stage2_size')
parser.add_argument('--fold', type=int, default=0, help='fold')
parser.add_argument('--epoch', type=int, default=200, help='all_epochs')
parser.add_argument('--seed', type=int, default=15, help='random seed')
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
random.seed(args.seed)

lr_max = 0.0002
data_dir = 'data'
split_path = 'CMB_train_val_5folds_211110.pkl'
L2 = 0.0001

save_name = 'bs{}_epoch{}_fold{}_seed{}'.format(args.bs, args.epoch, args.fold, args.seed)
save_dir = os.path.join('trained_models/{}_size{}'.format(args.name, args.stage2_size), save_name)
if os.path.exists(save_dir):
    shutil.rmtree(save_dir)
os.makedirs(save_dir, exist_ok=True)
train_writer = SummaryWriter(os.path.join(save_dir, 'log/train'), flush_secs=2)
val_writer = SummaryWriter(os.path.join(save_dir, 'log/val'), flush_secs=2)
input_size = (args.stage2_size, args.stage2_size)
train_data = Dataset_stage2(data_root=data_dir, split_path=split_path, fold=0, mode='train', input_size=input_size)
val_data = Dataset_stage2(data_root=data_dir, split_path=split_path, fold=0, mode='val', input_size=input_size)
net = Full_Resolution_Net_without_ASPP(in_channel=1, classes=1).cuda()

print(save_dir)

train_dataloader = DataLoader(dataset=train_data, batch_size=args.bs, shuffle=True, num_workers=2, pin_memory=True)
val_dataloader = DataLoader(dataset=val_data, batch_size=args.bs, shuffle=False, num_workers=2, pin_memory=True)

train_data_len = train_data.len
val_data_len = val_data.len
print('train_lenth: %i  val_lenth: %i' % (train_data_len, val_data_len))

Diceloss = DiceLoss()
BCELoss = nn.BCEWithLogitsLoss()
RCLoss = recall_Loss()

optimizer = optim.Adam(net.parameters(), lr=lr_max, weight_decay=L2)

best_precision = 0
best_recall = 0
best_precision_recall = 0
print('training')

for epoch in range(args.epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        break
    net.train()
    epoch_train_total_loss = []
    for i, (img, label) in enumerate(train_dataloader):
        img, label = img.float().cuda(), label.float().cuda()
        print('[%d/%d, %5d/%d]' % (epoch + 1, args.epoch, i + 1, math.ceil(train_data_len / args.bs)))
        optimizer.zero_grad()
        pred = net(img)
        BCEloss = BCELoss(pred, label)
        pred = torch.sigmoid(pred)
        diceloss = Diceloss(pred, label)
        rcloss = RCLoss(pred, label)
        loss_Seg = BCEloss + diceloss + rcloss
        loss_Seg.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 12)
        optimizer.step()

        epoch_train_total_loss.append(loss_Seg.item())
        print('train_total_loss: %.3f' % (loss_Seg.item()))

    net.eval()
    epoch_val_total_loss = []
    epoch_val_precision = []
    epoch_val_recall = []
    image_save = []
    label_save = []
    pred_save = []
    with torch.no_grad():
        for i, (img, label) in enumerate(val_dataloader):
            img, label = img.float().cuda(), label.float().cuda()
            pred = net(img)
            BCEloss = BCELoss(pred, label)
            pred = torch.sigmoid(pred)
            diceloss = Diceloss(pred, label)
            rcloss = RCLoss(pred, label)
            loss_Seg = BCEloss + diceloss + rcloss

            predictions = pred.view(pred.size(0), -1).cpu().numpy()
            predictions[predictions <= 0.5] = 0
            predictions[predictions > 0.5] = 1
            label_array = label.view(label.size(0), -1).cpu().numpy()
            for b in range(pred.size(0)):
                epoch_val_precision.append(precision_score(label_array[b], predictions[b], zero_division=1))
                epoch_val_recall.append(recall_score(label_array[b], predictions[b], zero_division=1))
            epoch_val_total_loss.append(loss_Seg.item())

            if i in [0, 1] and epoch % (args.epoch // 20) == 0:
                image_save.append(img[0:2, :, :, :].cpu())
                label_save.append(label[0:2, :, :, :].cpu())
                pred_save.append(pred[0:2, :, :, :].cpu())
    epoch_train_total_loss = np.mean(epoch_train_total_loss)

    epoch_val_total_loss = np.mean(epoch_val_total_loss)
    epoch_val_precision = np.mean(epoch_val_precision)
    epoch_val_recall = np.mean(epoch_val_recall)

    print(
        '[%d/%d] train_total_loss: %.3f val_total_loss: %.3f val_precision: %.3f val_recall: %.3f'
        % (epoch + 1, args.epoch, epoch_train_total_loss, epoch_val_total_loss, epoch_val_precision, epoch_val_recall))

    if epoch_val_precision > best_precision:
        best_precision = epoch_val_precision
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_precision.pth'))
    if epoch_val_recall > best_recall:
        best_recall = epoch_val_recall
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_recall.pth'))
    if epoch_val_precision + epoch_val_recall > best_precision_recall:
        best_precision_recall = epoch_val_precision + epoch_val_recall
        torch.save(net.state_dict(), os.path.join(save_dir, 'best_precision_recall.pth'))

    train_writer.add_scalar('lr', lr, epoch)
    train_writer.add_scalar('total_loss', epoch_train_total_loss, epoch)

    val_writer.add_scalar('total_loss', epoch_val_total_loss, epoch)
    val_writer.add_scalar('precision', epoch_val_precision, epoch)
    val_writer.add_scalar('recall', epoch_val_recall, epoch)
    val_writer.add_scalar('precision_recall', epoch_val_precision + epoch_val_recall, epoch)
    val_writer.add_scalar('best_precision', best_precision, epoch)
    val_writer.add_scalar('best_recall', best_recall, epoch)
    val_writer.add_scalar('best_precision_recall', best_precision_recall, epoch)
    if epoch % (args.epoch // 20) == 0:
        image_save = torch.cat(image_save, dim=0)
        label_save = torch.cat(label_save, dim=0)
        image_save = make_grid(image_save, 2, normalize=True)
        label_save = make_grid(label_save, 2, normalize=True)
        val_writer.add_image('image_save', image_save, epoch)
        val_writer.add_image('label_save', label_save, epoch)
        pred_save = torch.cat(pred_save, dim=0)
        pred_save = make_grid(pred_save, 2, normalize=True)
        val_writer.add_image('pred_save', pred_save, epoch)
    if (epoch + 1) == args.epoch:
        torch.save(net.state_dict(), os.path.join(save_dir, 'epoch' + str(epoch + 1) + '.pth'))

train_writer.close()
val_writer.close()
print('saved_model_name:', save_dir)
