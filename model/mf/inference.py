import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  

from tqdm import tqdm

import argparse
import sys
import os
from model.mf.mf import MF
import model.mf.dataprocessing as dp


class Inference():
    '''
    Inference for MF model....
    '''
    def __init__(self, epoch = 100, emb_size = 100, learning_rate = 1e-2, weight_decay = 0.0, if_cuda = False):
        '''
        epoch: int
        emb_size: int
        learning_rate: float
        weight_decay: float
        if_cuda: boolean
        '''
        self.num_epoch = epoch
        self.emb_size = emb_size
        self.learning_rate = learning_rate
        self.if_cuda = if_cuda
        self.weight_decay = weight_decay

    def set_data(self, user_data, item_data, rating_table):
        self.user_data = user_data
        self.item_data = item_data
        self.rating_table = rating_table

    def set_model(self):
        if self.if_cuda:
            model = MF(len(self.user_data), len(self.item_data), self.emb_size).cuda()
        else:
            model = MF(len(self.user_data), len(self.item_data), self.emb_size)

        self.model = model
    
    def test_model(self):
        self.model.eval()

        users = torch.LongTensor(self.rating_table.user_id.values)
        items = torch.LongTensor(self.rating_table.item_id.values)
        ratings = torch.FloatTensor(self.rating_table.rate.values)

        if self.if_cuda:
            users = users.cuda()
            items = items.cuda()
            ratings = ratings.cuda()

        predict = self.model(users, items)
        loss = F.mse_loss(predict, ratings)

        print(f"test loss {loss.item():.3f}")

    def train_model(self):
        print("[Info] train_model : start train.....")
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.model.train()

        for epoch in range(self.num_epoch):
            users = torch.LongTensor(self.rating_table.user_id.values)
            items = torch.LongTensor(self.rating_table.item_id.values)
            ratings = torch.FloatTensor(self.rating_table.rate.values)

            if self.if_cuda:
                users = users.cuda()
                items = items.cuda()
                ratings = ratings.cuda()

            predict = self.model(users, items)
            loss = F.mse_loss(predict, ratings)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 10 == 0:
                print(f"{epoch} / {self.num_epoch} epoch : {loss.item():.3f}")
        
        print("[Info] train model : Done!!")
    
    def set_U(self):
        U =  self.model.users_emb.weight.data
        U = U.cpu().detach().numpy()

        self.U = U

    def set_V(self):
        V = self.model.items_emb.weight.data
        V = V.cpu().detach().numpy()

        self.V =  V   

    def get_recommend_item(self, user_id):
        u = self.U[user_id]
        rate = u @ self.V.transpose()

        predict = []
        for i, p in enumerate(rate):
            predict.append((i, p))

        predict = sorted(predict, key = lambda x : x[1], reverse=True)
        
        return predict

def get_personal_recommend(dp, model, user_id_index, user_id, origin_read, recommend_num = 100):

    recommend_writer_list = model.get_recommend_item(user_id_index)

    content_by_writer = dp.content_by_writer
    writer_list = dp.item_list
    writer_index_to_id = {value : key for key, value in writer_list.items()}

    recommend_list = []
    recommended_count = 0
    
    origin_read = dp.user_read_list[user_id]

    for writer_index, p in recommend_writer_list:
        writer = writer_index_to_id[writer_index]
        if writer not in content_by_writer:
            continue
        temp_recommend = 0
        p = int(p)
        idx = 0
        while temp_recommend < p and idx < len(content_by_writer[writer]):
            content_id = content_by_writer[writer][idx]
            if content_id not in origin_read:
                recommend_list.append(content_id)
                temp_recommend += 1
                recommended_count += 1
            idx += 1
            if recommended_count >= recommend_num:
                break
        if recommended_count >= recommend_num:
            break
    
    count_view_list = dp.count_view_list
    idx = 0
    while recommended_count < recommend_num:
        if count_view_list[idx] in recommend_list or count_view_list[idx] in origin_read:
            idx += 1
            continue
        recommend_list.append(count_view_list[idx])
        idx += 1
        recommended_count += 1

    return recommend_list

def recommend_contents( dir_path:str, 
                        target_user_id_list, 
                        following_rate = 1, 
                        epoch = 100, 
                        emb_size = 100, 
                        learning_rate = 1e-1, 
                        weight_decay = 0.0, 
                        if_cuda = False, 
                        recommend_num = 100):
    
    data_process = dp.DataProcessing(dir_path= dir_path, following_rate = following_rate)

    user_data = data_process.user_list
    item_data = data_process.item_list

    rating_table = data_process.rate_table

    train_model = Inference(epoch=epoch, emb_size=emb_size, learning_rate=learning_rate, weight_decay=weight_decay, if_cuda=if_cuda)
    
    train_model.set_data(user_data, item_data, rating_table)

    train_model.set_model()
    train_model.train_model()
    train_model.test_model()

    train_model.set_U()
    train_model.set_V()

    count = 0
    recommend_total_list = {}
    origin_read_list = data_process.user_read_list

    for user_id_str in target_user_id_list:
        user_id = user_data[user_id_str]
        temp = get_personal_recommend(data_process, train_model, user_id, user_id_str, origin_read_list[user_id_str], recommend_num)
        recommend_total_list[user_id_str] = temp
        
    return recommend_total_list

if __name__=="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir_path", type=str, help="raw data directory path (ex: ../data/)", required=True)
    parser.add_argument("--output_dir_path", type=str, help="set output dir path (ex: ./) ", required=True)
    parser.add_argument("--following_rate", type=float, default=1.0, help="set follow rate")
    parser.add_argument("--if_cuda", type=bool, default=True, help="select True if use cuda")
    
    args = parser.parse_args()

    dir_path = args.data_dir_path
    output_path = args.output_dir_path

    user_list_path = os.path.join(dir_path, "predict/dev.users")
    user_list = []

    with open(user_list_path, 'r') as f:
        for line in f:
            user_list.append(line.strip())

    ans = recommend_contents(dir_path=dir_path, target_user_id_list=user_list, following_rate = args.following_rate, if_cuda=args.if_cuda)

    res = open(os.path.join(output_path, "recommend.txt"), 'w')
    
    for key, val in ans.items():
        temp = key + " "
        temp += " ".join(val)
        temp += "\n"
        res.write(temp)
    res.close()
 
