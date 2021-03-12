import pandas as pd 
import numpy as np
import os 
import sys  

from tqdm import tqdm
from collections import defaultdict  

import utils

class Data_processing():
    def get_meta_data(self, file_path : str):
        current_func_name = sys._getframe.f_code.co_name

        if not utils.check_file(file_path, ".json", current_func_name):
            return
        
        meta_data = pd.read_json(file_path, lines=True)

        self.meta_data = meta_data

        return meta_data  


    def get_user_data(self, file_path : str):
        current_func_name = sys._getframe.f_code.co_name

        if not utils.check_file(file_path, ".json", current_func_name):
            return

        user_data = pd.read_json(file_path, lines=True)

        self.user_data = user_data  

        return user_data


    def preprocess_read_data(self, file_path : str):
        ###################################
        # save read file as pickle
        # input:
        #   file_path : path of read directory
        ###############################

        current_func_name = sys._getframe.f_code.co_name

        if not os.path.exists(file_path):
            print(f"[Error] {current_func_name} : file is not exist ==> file path :{file_path}")
            return False
        
        # delete .2019010120_2019010121.un~
        # get file list in read directory
        read_file_list = os.listdir(file_path)  
        read_list = []  # file_name : file name
                        # user_id : user id
                        # content_id : written letter id

        for file_name in read_file_list:
            tmp_path = os.path.join(file_path, file_name)
            with open(tmp_path, 'r') as f:
                for line in f:
                    line_list = line.split()
                    read_list.append({"file_name":file_name,"user_id":line_list[0],"content_id":line_list[1:]})

        read_list = pd.DataFrame(read_list)
        read_list.to_pickle(file_path + ".pkl")

        return True


    def split_read_data(self, read_data, split_date: int):
        read_data["tmp_file_name"] = read_data["file_name"].apply(lambda x : int(x[:8]))

        read_train_data = read_data[read_data["tmp_file_name"] < split_date]
        read_test_data = read_data[read_data["tmp_file_name"] >= split_date]

        return read_train_data, read_test_data


    def get_read_data(self, file_path : str, split_date: int):
        #####################################
        # load read data
        # input :
        #   file_path : read pkl data
        #####################################
        current_func_name = sys._getframe.f_code.co_name

        if not utils.check_file(file_path, ".pkl", current_func_name):
           if not self.preprocess_read_data(file_path):
               return
            
        read_data = pd.read_pickle(file_path)
        
        read_train_data, read_test_data = self.split_read_data(read_data, split_date)

        self.read_train_data = read_train_data
        self.read_test_data = read_test_data

        return read_train_data, read_test_data

        
    def get_writer_by_letter(self):
        #################################
        # return:
        #   dict(str, list[str]) --> key : writer, value : list of letter id
        #################################
        
        writer_data = {}
        
        meta_data = self.meta_data
        
        for i, data in meta_data.iterrows():
            writer = data.user_id 
            letter = data.id 
            if writer not in writer_data:
                writer_data[writer] = []
            writer_data[writer].append(letter)
        
        return writer_data


    def get_following_list(self):
        following_list = {}

        user_data = self.user_data

        for i, data in user_data.iterrows():
            user_id = data['id']
            followings = data['following_list']
            following_list[user_id] = followings

        return following_list


    def get_rating_table(self, is_train: bool, weight_followee: int):

        user_read_list = defaultdict(int)
        userToIndex = {}
        writerToIndex = {}

        if is_train :
            read_data = self.read_train_data
        else:
            read_data = self.read_test_data


        following_list = get_following_list()

        user_idx = 0
        writer_idx = 0
        
        for idx, data in read_data.iterrows():
            user_id = data["user_id"]
            content_id_list = data["content_id"]
            if user_id not in userToIndex:
                userToIndex[user_id] = user_idx
                user_idx += 1
            user_id = userToIndex[user_id]
            for content_id in content_id_list:
                writer = content_id[:content_id.find("_")]
                if writer not in writerToIndex:
                    writerToIndex[writer] = writer_idx
                    writer_idx += 1
                writer = writerToIndex[writer]
                key = (user_id, writer)
                if writer in following_list[user_id]:
                    user_read_list[key] += weight_followee
                else:
                    user_read_list[key] += 1 

        user_list = []
        writer_list = []
        read_count_list = []

        for key, val in user_read_list.items():
            user_id = key[0]
            writer_id = key[1]
            read_count = val

            user_list.append(user_id)
            writer_list.append(writer_id)
            read_count_list.append(read_count)

        rating_data = {"user_id": user_list, "writer_id": writer_list, "rate": read_count_list}
        rating_data = pd.DataFrame(rating_data)

        return rating_data