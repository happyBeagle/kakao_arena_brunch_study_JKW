import collections
import copy
import pickle
import os
import torch
import random
from tqdm import tqdm
import pandas as pd 
import numpy as np
import os 
import sys
from collections import defaultdict  

def generate(model, user_train, config):
    user_list = copy.deepcopy(user_train)

    ## load dictionaries
    article2num, num2article, num2user, user2num = load_mappings('/content/drive/MyDrive/brunch/kakao_arena_brunch_study_JKW/model/sasrec/tmp/mapping')

    ## load dev read list. 중복 제거 위해
    dev_user_read_path = '/content/drive/MyDrive/brunch/kakao_arena_brunch_study_JKW/model/sasrec/tmp/dev_read_list'
    # with open(dev_user_read_path, 'rb') as lf:
    #     dev_read_list = pickle.load(lf)
    dev_read_list = load_pickle(dev_user_read_path)
    
    users = list(user_list.keys())
    predictions = {}
    for u in tqdm(users):
        user_predictions = []
        if user_list[u]:
            seq = np.zeros([config.TRAIN.MAXLEN], dtype=np.int32)
            idx = config.TRAIN.MAXLEN - 1
            # seq[idx] = valid[u][0]
            # seq[idx - 1] = test[u][0]
            # idx -= 2
            for i in reversed(user_list[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(dev_read_list[u])
            # rated.union(set([article2num[i] for i in dev_user_read[num2user[u]]]))
            # rated.add(0)
            logits = -model.predict_([np.array(l) for l in [[u], [seq]]][1])
            logits = logits[0]
            ranks = logits.argsort().argsort()
            ranks = collections.deque(list(ranks))
            while len(user_predictions) <= 100:
                rank = ranks.popleft()
                rank = rank.item()
                if rank not in rated:   
                    user_predictions.append(num2article[rank])
                    rated.add(rank)
                    seq = np.roll(seq, -1)
                    seq[-1] = rank
        else:
            # remain_users.append(num2user[u])
            print(u)

        predictions[num2user[u]] = user_predictions
            
    return predictions

#%%
def beam_search(model, user_train, args, k):
    user_list = copy.deepcopy(user_train)

    ## load dictionaries
    article2num, num2article, num2user, user2num = load_mappings('./mapping')

    ## load dev read list. 중복 제거 위해
    dev_user_read_path = './data/dev_read_list'
    # with open(dev_user_read_path, 'rb') as lf:
    #     dev_read_list = pickle.load(lf)
    dev_read_list = load_pickle(dev_user_read_path)
    
    users = list(user_list.keys())
    predictions = {}
    for u in tqdm(users):
        user_predictions = []
        if user_list[u]:
            seq = np.zeros([args.maxlen], dtype=np.int32)
            idx = args.maxlen - 1
            # seq[idx] = valid[u][0]
            # seq[idx - 1] = test[u][0]
            # idx -= 2
            for i in reversed(user_list[u]):
                seq[idx] = i
                idx -= 1
                if idx == -1: break

            rated = set(dev_read_list[u])
            # rated.union(set([article2num[i] for i in dev_user_read[num2user[u]]]))
            # rated.add(0)

            sequences = [[list(), 0.0]]
            while len(user_predictions) < 0:
                all_candidates = list()
                for i in range(len(sequences)):
                    seq, score = sequences[i]
                    
            logits = -model.predict_(*[np.array(l) for l in [[u], [seq]]])
            logits = logits[0]
            ranks = logits.argsort().argsort()
            ranks = collections.deque(list(ranks))
            while len(user_predictions) <= 100:
                rank = ranks.popleft()
                rank = rank.item()
                if rank not in rated:   
                    user_predictions.append(num2article[rank])
                    rated.add(rank)
                    seq = np.roll(seq, -1)
                    seq[-1] = rank
        else:
            # remain_users.append(num2user[u])
            print(u)

        predictions[num2user[u]] = user_predictions
            
    return predictions

#%%
def load_mappings(dir_path):
    mappings = os.listdir(dir_path)
    mappings = sorted(mappings)
    print(mappings)
    result = []
    for mapping in mappings:
        file_path = os.path.join(dir_path, mapping)
        with open(file_path, 'rb') as lf:
            dict = pickle.load(lf)
        result.append(dict)
    return result
# %%
def load_pickle(file_path):
    with open(file_path, 'rb') as lf:
        result = pickle.load(lf)
    return result
# %%
def load_txt(fname):
    usernum = 0
    itemnum = 0
    user_article_pair = defaultdict(list)
    
    f = open(f'/content/drive/MyDrive/brunch/data/{fname}.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        user_article_pair[u].append(i)
        itemnum = max(i, itemnum)
    
    usernum = len(user_article_pair.keys())

    return user_article_pair, usernum, itemnum

def iterate_data_files(from_dtm, to_dtm):
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    read_root = os.path.join('/content/drive/MyDrive/kaggle/data', 'read')
    for fname in os.listdir(read_root):
        if len(fname) != 21:
            continue 
        if from_dtm != 'None' and from_dtm > fname:
            continue 
        if to_dtm != 'None' and fname > to_dtm:
            continue 
        path = os.path.join(read_root, fname)
        yield path, fname 