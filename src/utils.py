#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   utils.py
@Time    :   2021/12/05 03:48:01
@Author  :   Vedant Thapa 
@Contact :   thapavedant@gmail.com
'''
import os
import numpy as np
import torch
import random
import cv2
import yaml
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset


class Custom_Dataset(Dataset):
    def __init__(self, img_path, targets, augmentations=None):
        self.img_path = img_path
        self.targets = targets
        self.augmentations = augmentations

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        target = self.targets[idx]
        img = cv2.imread(self.img_path[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.augmentations:
            augmented = self.augmentations(image=img)
            img = augmented['image']

        return img, target


def set_all_seeds(seed):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def load_config(config_name):
    with open(config_name) as file:
        config = yaml.safe_load(file)
    return config


def compute_accuracy(model, data_loader, device):
    model.eval()
    correct_pred, num_examples = 0, 0
    for i, (features, targets) in enumerate(data_loader):

        features = features.to(device)
        targets = targets.to(device)

        logits = model(features)
        _, predicted_labels = torch.max(logits, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100


def compute_epoch_loss(model, data_loader, device):
    model.eval()
    curr_loss, num_examples = 0., 0
    with torch.no_grad():
        for features, targets in data_loader:
            features = features.to(device)
            targets = targets.to(device)
            logits = model(features)
            loss = F.cross_entropy(logits, targets, reduction='sum')
            num_examples += targets.size(0)
            curr_loss += loss

        curr_loss = curr_loss / num_examples
        return curr_loss


def get_dataloaders(dataset, df, batch_size):
    train_idx = df[df.set == 'train'].index.values
    val_idx = df[df.set == 'test'].index.values

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    dl_train = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True
    )
    dl_val = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=True
    )

    return {'train': dl_train, 'val': dl_val}
