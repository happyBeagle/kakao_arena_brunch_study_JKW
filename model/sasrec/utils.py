import collections
import sys
import copy
import pickle
import os
import torch
import random
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Queue
from tqdm import tqdm

# sampler for batch generation
# def random_neq(vocab_list, ts):
#     t = int(random.choice(vocab_list))
#     while t in ts:
#         t = int(random.choice(vocab_list))
#     return t
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t
    
def sample_function(user_train, usernum, itemnum, batch_size, maxlen, result_queue, SEED):
    def sample():

        # user_list = list(user_train.keys())
        # user = random.choice(user_list)  #/////default
        # try: 
        #     while (100 <= len(user_train[user]) <= 1000): 
        #         user = np.random.randint(1, usernum + 1) # 
        # except KeyError: 
        #     print(user)
        user = np.random.randint(1, usernum + 1)
        while (50 <= len(user_train[user]) <= 100): 
            user = np.random.randint(0, usernum)

        
        seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1]
        idx = maxlen - 1

        ts = set(user_train[user])
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(0, itemnum, ts)
            nxt = i
            idx -= 1
            if idx == -1: break

        return (user, seq, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()

# train/val/test data generation
def data_partition(fname):
    user_train = {}
    user_valid = {}
    user_test = {}

    User, usernum, itemnum = load_txt(fname)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
    return [user_train, user_valid, user_test, usernum, itemnum]

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    HT = 0.0
    valid_user = 0.0

    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)

    for u in users:

        if len(train[u]) < 1 or len(test[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break
        rated = set(train[u])
        rated.add(0)
        item_idx = [test[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)
        # item_idx는 rated에 없는 item들만 있음 -> negative sample??
        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user


# evaluate on val set
def evaluate_valid(model, dataset, args):
    [train, valid, test, usernum, itemnum] = copy.deepcopy(dataset)

    NDCG = 0.0
    valid_user = 0.0
    HT = 0.0
    if usernum>10000:
        users = random.sample(range(1, usernum + 1), 10000)
    else:
        users = range(1, usernum + 1)
    for u in users:
        if len(train[u]) < 1 or len(valid[u]) < 1: continue

        seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i
            idx -= 1
            if idx == -1: break

        rated = set(train[u])
        rated.add(0)
        item_idx = [valid[u][0]]
        for _ in range(100):
            t = np.random.randint(1, itemnum + 1)
            while t in rated: t = np.random.randint(1, itemnum + 1)
            item_idx.append(t)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_user += 1

        if rank < 10:
            NDCG += 1 / np.log2(rank + 2)
            HT += 1
        if valid_user % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return NDCG / valid_user, HT / valid_user

def generate(model, user_train, args):
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
import pickle
def load_pickle(file_path):
    with open(file_path, 'rb') as lf:
        result = pickle.load(lf)
    return result
# %%
def load_txt(fname):
    usernum = 0
    itemnum = 0
    user_article_pair = defaultdict(list)
    
    # assume user_article_pair/item index starting from 1
    f = open(f'data/{fname}.txt', 'r')
    for line in f:
        u, i = line.rstrip().split(' ')
        u = int(u)
        i = int(i)
        user_article_pair[u].append(i)
        itemnum = max(i, itemnum)
    
    usernum = len(user_article_pair.keys())

    return user_article_pair, usernum, itemnum

import pandas as pd 
import numpy as np
import os 
import sys
from tqdm import tqdm
from collections import defaultdict  


def check_file(path:str, file_type:str, call_func:str)->bool:
    
    if not os.path.exists(path):
        print(f"[Error] {call_func} : file do not exist ==> file name : {path}")
        return False
    
    f_name, f_type = os.path.splitext(path)

    if f_type != file_type:
        print(f"[Error] {call_func} : file type is not {file_type} ==> file type :{f_type}")
        return False
    return True