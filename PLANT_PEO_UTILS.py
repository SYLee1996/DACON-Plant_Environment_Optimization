
import os 
import cv2
import copy
import math
import torch
import argparse
import numpy as np
import pandas as pd
import albumentations

from glob import glob
from sklearn.metrics import f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from albumentations.pytorch import ToTensorV2

from torch.utils.data import Dataset
from torch.nn.modules.loss import _WeightedLoss
from torch.optim.lr_scheduler import _LRScheduler
from albumentations.core.transforms_interface import ImageOnlyTransform

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def img_load(path):
        img = cv2.imread(path)[:,:,::-1]
        img = cv2.resize(img, (512, 512))
        return img


def score_function(real, pred):
        score = f1_score(real, pred, average="macro")
        return score


def get_train_data(data_dir, mode='train'):
    img_path_list = []
    meta_path_list = []
    label_list = []
    
    for case_name in os.listdir(data_dir):
        current_path = os.path.join(data_dir, case_name)
        if os.path.isdir(current_path):
            # get image path
            img_path_list.extend(glob(os.path.join(current_path, 'image', '*.jpg')))
            img_path_list.extend(glob(os.path.join(current_path, 'image', '*.png')))
            
            # get meta path
            meta_path_list.extend(glob(os.path.join(current_path, 'meta', '*.csv')))
            
            # get label
            label_df = pd.read_csv(current_path+'/label.csv')
            label_list.extend(label_df['leaf_weight'])
                
    return pd.DataFrame({'all_img_path' : img_path_list,
                        'all_meta_path' : meta_path_list,
                        'all_label' : label_list})


def get_test_data(data_dir):
        
    # get image path
    img_path_list = glob(os.path.join(data_dir, 'image', '*.jpg'))
    img_path_list.extend(glob(os.path.join(data_dir, 'image', '*.png')))
    img_path_list.sort(key=lambda x:int(x.split('/')[-1].split('.')[0]))
    
    # get meta path
    meta_path_list = glob(os.path.join(data_dir, 'meta', '*.csv'))
    
    # get label
    label_list = ["tmp"]*len(meta_path_list)
    return pd.DataFrame({'all_img_path' : img_path_list,
                        'all_meta_path' : meta_path_list,
                        'all_label' : label_list})


class Sobel_Unsharp_compose(ImageOnlyTransform):
    def __init__(self, dx=1, dy=0, ksize=3, blur_limit=(1,5), sigmaX=2.0, always_apply=False, p=0.5):
        super(Sobel_Unsharp_compose, self).__init__(always_apply=always_apply, p=p)
        self.dx = dx
        self.dy = dy
        self.ksize = ksize
        self.blur_limit = blur_limit
        self.sigmaX = sigmaX
        
    def apply(self, img, **params):
        img_sobel_x = cv2.Sobel(img, cv2.CV_8U, self.dx, self.dy, ksize=self.ksize)
        img_sobel_y = cv2.Sobel(img, cv2.CV_8U, self.dy, self.dx, ksize=self.ksize)
        gaussian = cv2.GaussianBlur(img, self.blur_limit, self.sigmaX)
        unsharp_image = cv2.addWeighted(img, self.sigmaX, gaussian, -1.0, 0)

        return cv2.add(unsharp_image, img_sobel_x + img_sobel_y)



class Custom_dataset(Dataset):
    def __init__(self, dset, max_len, mode='train'):
        self.dset = dset
        self.mode = mode
        self.max_len = max_len
        self.csv_feature_dict= {'내부온도관측치': [15.10000038, 40.59999847412109],
                                '외부온도관측치': [6.18789834529161, 89.4000015258789],
                                '내부습도관측치': [13.67999983, 110.6199913],
                                '외부습도관측치': [0.0, 201.0],
                                'CO2관측치': [259.0, 1757.0],
                                'EC관측치': [-0.00080744, 43.59114074707031],
                                '최근분무량': [0.0, 642997.3999999999],
                                '화이트 LED동작강도': [0.0, 201.0],
                                '레드 LED동작강도': [0.0, 201.0],
                                '블루 LED동작강도': [0.0, 201.0],
                                '냉방온도': [0.0, 28.0],
                                '냉방부하': [0.0, 403.5995],
                                '난방온도': [14.0, 26.0],
                                '난방부하': [0.0, 68.50000381469727],
                                '총추정광량': [0.0, 631.54],
                                '백색광추정광량': [0.0, 309.41],
                                '적색광추정광량': [0.0, 165.48],
                                '청색광추정광량': [0.0, 156.65],
                                'image_sum': [0.1031112670898437, 175.06153106689453]}
        
        self.augmentation = albumentations.Compose([
            Sobel_Unsharp_compose(dx=1, dy=0, ksize=1, blur_limit=(1,5), sigmaX=2.0, p=1),
            albumentations.Sharpen(p=1),
            albumentations.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=1),
            albumentations.FancyPCA(alpha=0.1, p=1),
            albumentations.Emboss(p=1),
            
            albumentations.Transpose(p=0.3),
            albumentations.HorizontalFlip(p=0.5),
            albumentations.RandomRotate90(p=0.5),
            albumentations.VerticalFlip(p=0.5),
            
            albumentations.CLAHE(clip_limit=5, p=1),
            albumentations.ElasticTransform(alpha_affine=30, p=0.4),
            albumentations.Posterize(p=1),
            
            albumentations.GaussNoise(p=1),
            albumentations.GaussianBlur(blur_limit=(1, 5), p=1),
            # albumentations.GlassBlur(sigma=0.1, max_delta=2, iterations=1, p=1),
            albumentations.GridDistortion(num_steps=20, distort_limit=0.3, border_mode=1, p=1),
            ])

    def __len__(self):
        return len(self.dset)

    def __getitem__(self, idx):
        
        if self.mode=='test':
            meta_path = self.dset['all_meta_path'][idx]
            image_sum = self.dset['image_sum'][idx]
            img_path = self.dset['all_img_path'][idx]
            mask_img_path = self.dset['mask_img'][idx]
            all_label = self.dset['all_label'][idx]
        else:
            meta_path = self.dset[idx]['all_meta_path']
            image_sum = self.dset[idx]['image_sum']
            img_path = self.dset[idx]['all_img_path']
            mask_img_path = self.dset[idx]['mask_img']
            all_label = self.dset[idx]['all_label']
            
        # csv -----------------------------------------------------------
        df = pd.read_csv(meta_path)
        df.set_index('시간', inplace=True)
        df['image_sum'] = image_sum
        df.interpolate(inplace=True)
        # MinMax scaling
        for col in df.columns:
            df[col] = df[col].astype(float) - self.csv_feature_dict[col][0]
            df[col] = df[col] / (self.csv_feature_dict[col][1]-self.csv_feature_dict[col][0])
            df[col].fillna(df[col].mean(), inplace=True)
            df[col].fillna(0, inplace=True)
        # # zero padding
        pad = np.zeros((self.max_len, len(df.columns)))
        length = min(self.max_len, len(df))
        pad[-length:] = df.to_numpy()[-length:]
        # transpose to sequential data
        csv_feature = pad[-length:].T

        # image ----------------------------------------------------------
        img = cv2.imread(img_path)[:,:,::-1]
        img = cv2.resize(img, (512, 512))
        
        mask_img = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)
        mask_img = cv2.resize(mask_img, (512, 512)).reshape(512,512,1)
        
        img = np.concatenate([img, mask_img], 2)
        
        if self.mode=='train':
            # augmented = self.augmentation(image=img) 
            # img = augmented['image']
            pass
        else:
            # augmented = self.test_augmentation(image=img) 
            # img = augmented['image']
            pass
        img = transforms.ToTensor()(img)
        # img = self.transforms(img)
        
        # label ----------------------------------------------------------
        label = all_label

        return {'img' : img,
                'csv_feature' : csv_feature, 
                'label' : label}


class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class FocalLossWithSmoothing(torch.nn.Module):
    def __init__(
            self,
            num_classes: int,
            gamma: int = 1,
            lb_smooth: float = 0.1,
            ignore_index: int = None):

        super(FocalLossWithSmoothing, self).__init__()
        self._num_classes = num_classes
        self._gamma = gamma
        self._lb_smooth = lb_smooth
        self._ignore_index = ignore_index
        self._log_softmax = torch.nn.LogSoftmax(dim=1)

        if self._num_classes <= 1:
            raise ValueError('The number of classes must be 2 or higher')
        if self._gamma < 0:
            raise ValueError('Gamma must be 0 or higher')

    def forward(self, logits, label):

        logits = logits.float()
        difficulty_level = self._estimate_difficulty_level(logits, label)

        with torch.no_grad():
            label = label.clone().detach()
            if self._ignore_index is not None:
                ignore = label.eq(self._ignore_index)
                label[ignore] = 0
            lb_pos, lb_neg = 1. - self._lb_smooth, self._lb_smooth / (self._num_classes - 1)
            lb_one_hot = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
        logs = self._log_softmax(logits)
        loss = -torch.sum(difficulty_level * logs * lb_one_hot, dim=1)
        if self._ignore_index is not None:
            loss[ignore] = 0
        return loss.mean()

    def _estimate_difficulty_level(self, logits, label):

        one_hot_key = torch.nn.functional.one_hot(label, num_classes=self._num_classes)
        if len(one_hot_key.shape) == 4:
            one_hot_key = one_hot_key.permute(0, 3, 1, 2)
        if one_hot_key.device != logits.device:
            one_hot_key = one_hot_key.to(logits.device)
        pt = one_hot_key * F.softmax(logits)
        difficulty_level = torch.pow(1 - pt, self._gamma)
        return difficulty_level


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

    def forward(self, output, target):

        cross_entropy = F.cross_entropy(output, target)
        cross_entropy_log = torch.log(cross_entropy)
        logpt = - F.cross_entropy(output, target)
        pt    = torch.exp(logpt)

        focal_loss = -((1 - pt) ** self.gamma) * logpt

        balanced_focal_loss = self.alpha * focal_loss

        return balanced_focal_loss


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"

        self.base_optimizer.step()  # do the actual "sharpness-aware" update

        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass

        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
                    torch.stack([
                        ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
                        for group in self.param_groups for p in group["params"]
                        if p.grad is not None
                    ]),
                    p=2
                )
        return norm

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
