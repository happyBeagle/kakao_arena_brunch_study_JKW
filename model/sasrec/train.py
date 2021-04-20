#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader.dataset import BrunchDataset
from data_loader.dataprocessing import Preprocessing_sasrec
from model.sasrec.model.model import SASRec
from model.sasrec.config import get_config
from trainer.trainer import Trainer
import time
import torch
from torch.utils.data.dataset import random_split
import argparse

from tqdm import tqdm
from utils import *

def main(config):
    #%%
    preprocessing = Preprocessing_sasrec('/content/drive/MyDrive/brunch/data')
    item_num = max(preprocessing.content2num.values())
    
    train_dataset = BrunchDataset(preprocessing.train_df, item_num)
    train_len = int(0.9*len(train_dataset))
    valid_len = len(train_dataset) - train_len
    train_dataset, val_dataset = random_split(train_dataset, [train_len, valid_len])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    model = SASRec(item_num, config)
    model.to(config.SYSTEM.DEVICE)
    criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR, betas=(0.9, 0.98))

    trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, config)
    trainer.train(config.TRAIN.EPOCH)
# %%
if __name__ == '__main__':
    config = get_config()
    main(config)