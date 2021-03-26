# %%
import pandas as pd
import numpy as np
from tqdm import tqdm
import sys
import os
from preprocessing.readrawdata import ReadRawData
from model.ConvMF.tokenizer import Tokenizer
from scipy.sparse.csr import csr_matrix
import random
import utils
# %%
class DataProcessing:
    def __init__(self, dir_path = "../data", save_path="./model/ConvMF/data", do_split_data=True, following_rate = 1, ratio = 0.2):
        self.__read_data = ReadRawData(dir_path)
        self.__item_list, self.__item_index = self.__get_info_item()
        self.__user_list, self.__user_index = self.__get_read_user()
        # set R data
        self.__following_list = self.__get_following_list()
        self.__rate_table = self.__get_rate_table(following_rate)
        if do_split_data:
            self.__train, self.__valid, self.__test = self.__split_data(ratio)
            self.__save_train_valid_test(save_path, self.__train, "train")
            self.__save_train_valid_test(save_path, self.__valid, "valid")
            self.__save_train_valid_test(save_path, self.__test, "test")
        
        self.__train_user = self.load_file(os.path.join(save_path, "user_train.dat"))
        self.__train_item = self.load_file(os.path.join(save_path, "item_train.dat"))
        self.__valid_user = self.load_file(os.path.join(save_path, "user_valid.dat"))
        self.__test_user = self.load_file(os.path.join(save_path, "user_test.dat"))
        # set D
        self.__content_info_list, self.__max_len = self.__get_content_data()
        self.__x_sequence, self.__vocab_size = self.__process_content_data() 

    def __get_following_list(self):
        print("[Info] DataProcessing : get following list....", end="")
        users = self.__read_data.users
        following_list = {}
        for i, data in users.iterrows():
            user_id = data.id
            following_list[user_id] = data["following_list"]
        
        print("Done!!!")
        return following_list

    def __get_info_item(self):
        print("[Info] DataProcessing : get info item...", end="")
        read = self.__read_data.read


        item_list = {}
        item_index = 0
        metadata_content = set(self.__read_data.metadata.id)
 
        for i, data in read.iterrows():
            for content_id in data.content_id:
                if content_id in metadata_content:
                    if content_id not in item_list:
                        item_list[content_id] = item_index
                        item_index += 1
        print("Done!!!")
        return item_list, item_index

    def __get_read_user(self):
        print("[Info] DataProcessing : get read user...", end="")
        read = self.__read_data.read
        item_list = self.__item_list

        read_count = {}

        for i, data in read.iterrows():
            user_id = data.user_id

            if user_id not in read_count:
                read_count[user_id] = 0
            
            for content_id in data.content_id:
                if content_id in item_list:
                    read_count[user_id] += 1
        
        user_list = {}
        user_index = 0

        for user_id, count in read_count.items():
            if count > 0:
                user_list[user_id] = user_index
                user_index += 1

        
        print("Done!!!")
        return user_list, user_index

    def __get_rate_table(self, following_rate):
        print("[Info] DataProcessing : get rate table....")
        read = self.__read_data.read
        user_list = self.__user_list
        item_list = self.__item_list

        print("[+] count number of views......")
        read_count = {}
        for i, data in tqdm(read.iterrows(), total=read.shape[0]):
            user_id = data.user_id
            content_list = data.content_id
            for content_id in content_list:
                if content_id in item_list:
                    key = (user_id, content_id)
                    if key not in read_count:
                        read_count[key] = 0
                    read_count[key] += 1
        
        following_list = self.__following_list
        user_array = []
        item_array = []
        rate_array = []
        print("[+] setting rate table......")

        for key, val in tqdm(read_count.items()):
            user_id = key[0]
            writer_id = key[1]
            rate = val

            if user_id not in user_list:
                user_list[user_id] = self.__user_index
                self.__user_index += 1

            user_array.append(user_list[user_id])
            item_array.append(item_list[writer_id])

            if user_id in following_list and writer_id in following_list[user_id]:
                rate_array.append(rate * following_rate)
            else:
                rate_array.append(rate)

        rate_table = csr_matrix((rate_array, (user_array, item_array)))

        print("Done!!!")
        return rate_table

    def __split_data(self, ratio):
        R = self.__rate_table
        train = []
        print("[Info] DataProcessing : split data...")
        print("[+] DataProcessing : processing train data.....")
        for i in tqdm(range(R.shape[0])):
            user_rating = R[i].nonzero()[1]
            if len(user_rating) == 0:
                continue
            np.random.shuffle(user_rating)
            train.append((i, user_rating[0]))
        
        remain_item = set(range(R.shape[1])) - set(list(zip(*train))[1])
        
        R_csc = R.tocsc().T
        for j in tqdm(remain_item):
            item_rating = R_csc[j].nonzero()[1]
            if len(item_rating) == 0:
                continue
            np.random.shuffle(item_rating)
            train.append((item_rating[0], j))
        
        rating_list = set(zip(R.nonzero()[0], R.nonzero()[1]))
        total_size = len(rating_list)
        remain_rating_list = list(rating_list - set(train))
        random.shuffle(remain_rating_list)

        num_addition = int((1 - ratio) * total_size) - len(train)

        if num_addition < 0:
            print("[-] DataProcessing : num addition is invalid....")
            sys.exit()
        
        train.extend(remain_rating_list[:num_addition])
        temp = remain_rating_list[num_addition:]
        random.shuffle(temp)
        valid = temp[::2]
        test = temp[1::2]

        trainset_u_idx, trainset_i_idx = zip(*train)
        trainset_u_idx = set(trainset_u_idx)
        trainset_i_idx = set(trainset_i_idx)

        if len(trainset_u_idx) != R.shape[0] or len(trainset_i_idx) != R.shape[1]:
            print("[-] DataProcessing : check the data....")
            sys.exit()

        return train, valid, test

    def __save_train_valid_test(self, target_dir, data, data_name):
        print("[Info] DataProcessing : save train valid test...")
        R = self.__rate_table
        R_lil = self.__rate_table.tolil()

        user_ratings = {}
        item_ratings = {}

        for i, j in tqdm(data):
            if i in user_ratings:
                user_ratings[i].append(j)
            else:
                user_ratings[i] = [j]

            if j in item_ratings:
                item_ratings[j].append(i)
            else:
                item_ratings[j] = [i]

        print(f"[+] Dataprocessing : save train valid test -> user_{data_name}.dat")
        format_user = []
        for i in tqdm(range(R.shape[0])):
            if i in user_ratings:
                format_list = [str(len(user_ratings[i]))]
                format_list.extend(["%d::%.1f"%(j, R_lil[i,j]) for j in sorted(user_ratings[i])])
                format_user.append(" ".join(format_list))
            else:
                format_user.append("0")
        
        user_file = open(os.path.join(target_dir, f"user_{data_name}.dat"), 'w')
        user_file.write("\n".join(format_user)) 
        user_file.close()
        print(f"[+] Dataprocessing : save train valid test -> item_{data_name}.dat")
        format_item = []
        for j in tqdm(range(R.shape[1])):
            if j in item_ratings:
                format_list = [str(len(item_ratings[j]))]
                format_list.extend(["%d::%.1f"%(i, R_lil[i, j]) for i in sorted(item_ratings[j])])
                format_item.append(" ".join(format_list))
            else:
                format_item.append("0")
        item_file = open(os.path.join(target_dir, f"item_{data_name}.dat"), 'w')
        item_file.write("\n".join(format_item))
        item_file.close()
        print("Done!!!")
    
    def load_file(self, target_path):
        print("[Info] DataProcessing : load file...")
        result = []
        rating = open(target_path, 'r')

        index_list = []
        rating_list = []

        for line in rating:
            temp = line.split()
            num_rating = int(temp[0])
            if num_rating > 0:
                tmp_i, tmp_r = zip(*(data.split("::") for data in temp[1:]))
                index_list.append(np.array(tmp_i, dtype=int))
                rating_list.append(np.array(tmp_r, dtype=float))
            else:
                index_list.append(np.array([], dtype=int))
                rating_list.append(np.array([], dtype=float))

        result.append(index_list)
        result.append(rating_list)

        return result

    def __get_content_data(self):
        print("[Info] DataProcessing : get content data...")
        metadata = self.__read_data.metadata
        item_list = self.__item_list

        content_info = {}
        max_len = 0
        for i, data in metadata.iterrows():
            if data.id not in item_list:
                continue
            content = data.title + " " + data.sub_title
            content_info[item_list[data.id]] = content
            max_len = max(max_len, len(content))

        print("Done!!!")
        return content_info, max_len

    def __process_content_data(self):
        print("[Info] DataProcessing : process content data...")
        content_info_list = self.__content_info_list

        tokenizer = Tokenizer()
        x_sequence = []
        for content_id, content_info in content_info_list.items():
            temp_list = tokenizer.get_token_ids(content_info)
            x_sequence.append(temp_list)

        return x_sequence, tokenizer.vocab_size

    @property
    def rate_table(self):
        return self.__rate_table
    
    @property
    def train_user(self):
        return self.__train_user
    
    @property
    def train_item(self):
        return self.__train_item

    @property
    def valid_user(self):
        return self.__valid_user
        
    @property
    def test_user(self):
        return self.__test_user
      
    @property
    def max_len(self):
        return self.__max_len

    @property
    def x_sequence(self):
        return self.__x_sequence

    @property
    def vocab_size(self):
        return self.__vocab_size


# %%
