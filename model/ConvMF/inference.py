import sys
import os 
import numpy as np 
import torch, gc
from model.ConvMF.dataprocessing import DataProcessing
from model.ConvMF.convmf import ConvMF
from preprocessing.readrawdata import ReadRawData

class Inference:
    def __init__(self, rate_table, vocab_size, if_cuda, cnn_input, cnn_epoch = 20, cnn_lr = 1e-4,
                 lambda_u=1, lambda_v=100, dimension=50, dropout_rate=0.2, emb_dim=200, max_len=50, num_kernel_per_ws=100):
        
        self.__convMF  = ConvMF(rate_table, vocab_size, if_cuda, cnn_input,
                                cnn_epoch=cnn_epoch, cnn_lr=cnn_lr,
                                lambda_u=lambda_u, lambda_v=lambda_v, dimension=dimension,
                                dropout_rate=dropout_rate, emb_dim=emb_dim, max_len=max_len, num_kernel_per_ws=num_kernel_per_ws)
        
    def train(self, max_epochs, train_user, train_item, valid_user, test_user):
        self.__convMF.train(max_epochs, train_user, train_item, valid_user, test_user)
    
    def get_personal_recommend_list(self, user_id):
        u = self.__convMF.U[user_id]
        content_predict_list = u@self.__convMF.V.T
        content_predict_list = content_predict_list.reshape(-1)
        content_list = []
        for i, p in enumerate(content_predict_list):
            content_list.append((i, p))
        
        content_list = sorted(content_list, key=lambda x: x[1], reverse=True)
        return content_list

def recommend_contents(dir_path):
    dp = DataProcessing(dir_path =dir_path)

    item_list = dp.item_list
    user_list = dp.user_list

    train_user = dp.train_user
    train_item = dp.train_item
    valid_user = dp.valid_user
    test_user = dp.test_user

    vocab_size = dp.vocab_size
    rate_table = dp.rate_table

    recommend_user_list = dp.recommend_user_list
    already_read_list = dp.already_read_list
    content_view_list = dp.content_view_list
    cnn_x = dp.x_sequence
    
    del dp

    input_array = np.full((len(cnn_x), 80), vocab_size)

    for i in range(len(cnn_x)):
        for j in range(len(cnn_x[i])):
            input_array[i][j] = cnn_x[i][j]

    inference = Inference(rate_table, vocab_size, True, input_array, max_len=80)
    inference.train(100, train_user, train_item, valid_user, test_user)

    index_to_item_list = {val:key for key, val in item_list.items()}
    recommend_list = {}

    for user_id in recommend_user_list:
        temp_list = []
        if user_id in user_list:
            count = 0
            recommend_temp_list = inference.get_personal_recommend_list(user_list[user_id])
            for item_index, _ in recommend_temp_list:
                item_id = index_to_item_list[item_index]
                if item_id in already_read_list[user_id]:
                    continue
                temp_list.append(item_id)
                count += 1
                if count >= 100:
                    break
        else:
            count = 0
            for key, val in content_view_list:
                temp_list.append(key)
                count += 1
                if count >= 100:
                    break
        recommend_list[user_id] = temp_list

    return recommend_list

