#%%
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from data_loader.dataset import BrunchDataset
from preprocessing.Preprocessing_sasrec import Preprocessing_sasrec
import time
import torch
import argparse

from tqdm import tqdm
from utils import *
# from predict import get_predict_user

# def str2bool(s):
#     if s not in {'false', 'true'}:
#         raise ValueError('Not a valid boolean string')
#     return s == 'true'


# def main(config):
#     f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')
#     # Data Loader
#     dataset = data_partition(args.dataset)
#     [user_train, user_valid, user_test, usernum, itemnum] = dataset

#     num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)

#     cc = 0.0
#     for u in user_train:
#         cc += len(user_train[u])
#     print('average sequence length: %.2f' % (cc / len(user_train)))
    
#     sampler = WarpSampler(user_train, usernum, itemnum, batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3)


#     # Model
#     model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

#     for name, param in model.named_parameters():
#         try:
#             torch.nn.init.xavier_uniform_(param.data)
#         except:
#             pass # just ignore those failed init layers

#     epoch_start_idx = 1
#     if args.state_dict_path is not None:
#         model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
#         try:
#             print('Loading state_dicts...')
#             model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
#             print('Done')
#             tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
#             epoch_start_idx = int(tail[:tail.find('.')]) + 1
#         except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
#             print('failed loading state_dicts, pls check file path: ')
#             print(args.state_dict_path)
#             print('pdb enabled for your quick check, pls type exit() if you do not need it')
#             import pdb; pdb.set_trace()

#     criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    


# if __name__ == '__main__':
#     config = get_config()
#     main(config)
#%%
preprocessing = Preprocessing_sasrec('/content/drive/MyDrive/brunch/data')
item_num = max(preprocessing.content2num.values())
from torch.utils.data.dataset import random_split
train_dataset = BrunchDataset(preprocessing.train_df, item_num)
train_dataset, val_dataset = random_split(train_dataset, [90, 10])
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
# %%    
from model.sasrec.model.model import SASRec
from model.sasrec.config import get_config
config = get_config()
model = SASRec(item_num, config)
model.to(config.SYSTEM.DEVICE)
criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.BASE_LR, betas=(0.9, 0.98))

from trainer.trainer import Trainer
trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, config)
# %%
trainer.train(config.TRAIN.EPOCH)
# %%
