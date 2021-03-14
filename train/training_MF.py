# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable  

from tqdm import tqdm

if __package__ is None:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from model import model_MF
    import data_process as dp
else:
    from ..model import model_MF
    from . import data_process as dp

# %%
class Train_MF():
    def __init__(self, epoch = 100, emb_size = 100, lr = 1e-2, weight_decay = 0.0, if_cuda = False):
        self.num_epoch = epoch
        self.emb_size = emb_size
        self.learning_rate = lr
        self.if_cuda = if_cuda
        self.weight_decay = weight_decay

    def set_util_data(self, user_data, item_data):
        self.user_data = user_data
        self.item_data = item_data
       
    def set_train_data(self, train_rating):
        self.train_data = train_rating


    def set_test_data(self, test_rating):
        self.test_data = test_rating   


    def set_model(self):
        if self.if_cuda:
            model = model_MF.MF_model(len(self.user_data), len(self.item_data), self.emb_size).cuda()
        else:
            model = model_MF.MF_model(len(self.user_data), len(self.item_data), self.emb_size)

        self.model = model

    
    def test_model(self):
        self.model.eval()

        users = torch.LongTensor(self.test_data.user_id.values)
        items = torch.LongTensor(self.test_data.writer_id.values)
        ratings = torch.FloatTensor(self.test_data.rate.values)

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

        for epoch in tqdm(range(self.num_epoch)):
            users = torch.LongTensor(self.train_data.user_id.values)
            items = torch.LongTensor(self.train_data.writer_id.values)
            ratings = torch.FloatTensor(self.train_data.rate.values)

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
                print(f"{epoch} epoch : {loss.item():.3f}")
        print("[Info] train model : Done!!")
    
    def set_U(self):
        U =  self.model.users_emb.weight.data
        U = U.cpu().detach().numpy()

        print("U : ")
        print(U)
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


def get_personal_recommand(dp, model, user_id,user_id_str, recommand_num = 100):
    recommand_writer_list = model.get_recommend_item(user_id)
    letter_by_writer = dp.letter_by_writer
    user_origin_read = dp.get_user_origin_read(user_id_str)
    index_to_writer = dp.index_to_writer

    recommand_list = []
    recommand_count = 0
    for writer_index, p in recommand_writer_list:
        writer = index_to_writer[writer_index]
        if writer not in letter_by_writer:
            continue
        for letter in letter_by_writer[writer]:
            if letter in user_origin_read:
                continue
            recommand_list.append(letter)
            recommand_count += 1

            if recommand_count >= recommand_num:
                break
        if recommand_count >= recommand_num:
            break

    return recommand_list




def recommand_letters(dir_path:str, split_read_data:int, target_user_id_list,  epoch = 100, emb_size = 100, lr = 1e-1, weight_decay = 0.0, if_cuda = False, recommand_num = 100, weight_followee=1):
    data_process = dp.Data_processing(dir_path)

    data_process.set_MF_model_data(weight_followee = weight_followee)

    user_data = data_process.user_to_index
    item_data = data_process.writer_to_index

    rating_train_data = data_process.train_rating_table
    rating_test_data = data_process.test_rating_table


    train_MF = Train_MF(epoch=epoch, emb_size=emb_size, lr=lr, weight_decay=weight_decay, if_cuda=if_cuda)
    
    train_MF.set_util_data(user_data, item_data)
    train_MF.set_train_data(rating_train_data)
    train_MF.set_test_data(rating_test_data)

    train_MF.set_model()
    train_MF.train_model()
    train_MF.test_model()

    train_MF.set_U()
    train_MF.set_V()

    count = 0
    recommand_total_list = {}
    for user_id_str in target_user_id_list:
        user_id = data_process.user_to_index[user_id_str]
        temp = get_personal_recommand(data_process, train_MF, user_id,user_id_str, recommand_num)
        if count < 2 :
            print(user_id_str)
            print(temp)
            count += 1
        recommand_total_list[user_id_str] = temp
        
    return recommand_total_list


# %%
dir_path = "../../data/"

user_list_path = os.path.join(dir_path, "predict/dev.users")
user_list = []

with open(user_list_path, 'r') as f:
    for line in f:
        user_list.append(line.strip())

ans = recommand_letters(dir_path=dir_path, split_read_data=20190222,target_user_id_list=user_list, if_cuda=True)

print(ans[user_list[0]])
# %%
