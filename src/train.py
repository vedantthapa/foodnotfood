#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   train.py
@Time    :   2021/12/05 03:46:19
@Author  :   Vedant Thapa 
@Contact :   thapavedant@gmail.com
'''

import utils
import time
import timm
import wandb
import torch
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2


def train_model(CFG, dataloader, model, optimizer, criterion, experiment='foodnotfood-v1'):
    epochs = CFG['EPOCHS']
    device = CFG['DEVICE']
    start_time = time.time()

    with wandb.init(project=experiment, config=CFG):
        for epoch in range(epochs):

            model.train()
            for batch_idx, (features, targets) in enumerate(dataloader['train']):
                features = features.to(device)
                targets = targets.to(device)

                logits = model(features)
                cost = criterion(logits, targets)
                optimizer.zero_grad()
                cost.backward()

                optimizer.step()

                if not batch_idx % 50:
                    print(
                        f"Epoch: {epoch + 1:03d}/{epochs:03d} | Batch: {batch_idx:04d}/{len(dataloader['train']):04d} | Cost: {cost:.4f}")

            model.eval()
            with torch.set_grad_enabled(False):
                train_acc = utils.compute_accuracy(
                    model, dataloader['train'], device=device)
                val_acc = utils.compute_accuracy(
                    model, dataloader['val'], device=device)

                train_loss = utils.compute_epoch_loss(
                    model, dataloader['train'], device=device)
                val_loss = utils.compute_epoch_loss(
                    model, dataloader['val'], device=device)

                wandb.log(
                    {'Acc(Train)': train_acc,
                     'Acc(Val)': val_acc,
                     'Loss(Train)': train_loss,
                     'Loss(Val)': val_loss
                     }
                )

                print(f'Epoch: {epoch+1:03d}/{epochs:03d}\n'
                      f'Train - ACC: {train_acc:.2f} | LOSS: {train_loss:.2f}\n'
                      f'Val - ACC: {val_acc:.2f} | LOSS: {val_loss:.2f}\n')

            print(f'Time elapsed: {((time.time() - start_time)/60):.2f} min')

    print(f'Total Training Time: {((time.time() - start_time)/60):.2f} min')


if __name__ == "__main__":
    utils.set_all_seeds(42)
    CFG = utils.load_config('../config.yaml')
    CFG['DEVICE'] = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    wandb.login()
    df = pd.read_csv(CFG['DATASET_PATH'])

    aug = A.Compose([
        A.Resize(*CFG['IMG_SIZE']),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    dataset = utils.Custom_Dataset(
        df.img,
        df.enc_label,
        augmentations=aug
    )
    label_map = dict(zip(df.enc_label, df.label))

    dataloaders = utils.get_dataloaders(
        dataset,
        df,
        CFG['BATCH_SIZE']
    )

    efficientnet_b0 = timm.create_model(CFG['MODEL_NAME'], pretrained=True)
    for param in efficientnet_b0.parameters():
        param.required_grad = False
    in_features = efficientnet_b0.classifier.in_features
    efficientnet_b0.classifier = nn.Linear(
        in_features=in_features,
        out_features=CFG['NUM_CLASSES']
    )
    efficientnet_b0 = efficientnet_b0.to(CFG['DEVICE'])

    optimizer = optim.Adam(efficientnet_b0.parameters())
    criterion = nn.CrossEntropyLoss()

    train_model(
        CFG,
        dataloaders,
        efficientnet_b0,
        optimizer,
        criterion,
        experiment='foodnotfoodv1'
    )

    torch.save(efficientnet_b0, 'assets/model/efficientnet-B0-v1.pth')
