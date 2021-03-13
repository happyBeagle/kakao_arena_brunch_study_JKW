import torch
import torch.nn as nn
import torch.nn.functional as F
from model import model_MF
import data_process as dp
from torch.autograd import Variable  

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

    
    def test_loss(self):
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
        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate, weight_decay = self.weight_decay)
        self.model.train()

        for epoch in range(self.num_epoch):
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
    
    
    def get_U(self):
        U =  self.model.users_emb.weight.data
        U = U.cpu().detach().numpy()

        return U

    def get_V(self):
        V = self.model.items_emb.weight.data
        V = V.cpu().detach().numpy()

        return V   


    def get_recommend_item(self, user_id):
        u = self.U[user_id]
        rate = u @ self.V.transpose()

        predict = []
        for i, p in enumerate(R):
            predict.append((i, p))

        predict = sorted(predict, key = lambda x : x[1], reverse=True)
        
        return predict


def get_personal_recommand(dp, model, user_id, recommand_num = 100):
    recommand_writer_list = model.get_recommend_item(user_id)
    letter_by_writer = dp.letter_by_writer
    user_origin_read = dp.get_user_origin_read(user_id)

    recommand_list = []
    recommand_count = 0
    for writer in recommand_list:
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


def recommand_letters(dir_path:str, split_read_data:int, target_user_id_list,  epoch = 100, emb_size = 100, lr = 1e-2, weight_decay = 0.0, if_cuda = False, recommand_num = 100):
    
    data_process = dp.Data_processing(dir_path)

    data_process.set_MF_model_data(weight_followee = 2)

    user_data = data_process.user_to_index
    item_data = data_process.writer_to_index

    rating_train_data = data_process.train_rating_table
    rating_test_data = data_process.test_rating_table


    train_MF = Train_MF(epoch=epoch, emb_size=emb_size, learning_rate=lr, weight_decay=weight_decay, if_cuda=if_cuda)
    
    train_MF.set_util_data(user_data, item_data)
    train_MF.set_train_data(rating_train_data)
    train_MF.sef_test_data(rating_test_data)

    train_MF.set_model()
    train_MF.train_model()
    train_MF.test_model()

    recommand_total_list = []
    for user_id in target_user_id_list:
        temp_recommand = [user_id]
        temp_recommand.extend(get_personal_recommand(dp, model, user_id, recommand_num))
        recommand_total_list.append(temp_recommand)

    return recommand_total_list


if __name__=="__main__":
