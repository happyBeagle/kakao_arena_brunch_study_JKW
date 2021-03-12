import torch
import torch.nn as nn
import torch.nn.functional as F
import model_MF
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


    
def recommand_letters():
    
    train_MF = Train_MF(if_cuda=True)
    #TODO : 함수 채우기
         