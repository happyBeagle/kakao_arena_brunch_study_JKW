# %%
import pandas as pd

import sys
import os
from tqdm.notebook import tqdm
from preprocessing.readrawdata import ReadRawData
# %%
class DataProcessing:
    '''
    data processing class for MF model.
    '''
    def __init__(self, dir_path = "../../../data", following_rate = 1):
        '''
        dir_path: directory path about raw data files directory
        following_rate: you can weight rate for the authors followed by user
        '''
        self.__read_data = ReadRawData(dir_path)

        self.__following_list = self.__get_following_list()
        self.__user_list, self.__user_index = self.__get_user_list()
        self.__item_list, self.__item_index = self.__get_item_list()
        self.__rate_table = self.__get_rate_table(following_rate)
        self.__content_by_writer = self.__get_content_by_writer()
        self.__user_read_list = self.__get_user_read_list()
        self.__count_view_list = self.__get_count_view_list()

    def __get_following_list(self):
        print("[Info] DataProcessing : get following list....", end="")
        users = self.__read_data.users
        following_list = {}
        for i, data in users.iterrows():
            user_id = data.id
            following_list[user_id] = data["following_list"]
        
        print("Done!!!")
        return following_list

    def __get_user_list(self):
        print("[Info] DataProcessing : get user list....", end="")
        users = self.__read_data.users

        user_list = {}
        user_index = 0

        for i, data in users.iterrows():
            if data.id in user_list:
                continue
            user_list[data.id] = user_index
            user_index += 1

        print("Done!!!")        
        return user_list, user_index

    def __get_item_list(self):
        print("[Info] DataProcessing : get item list....", end="")
        metadata = self.__read_data.metadata
        item_list = {}
        item_index = 0
        for i, data in metadata.iterrows():
            if data.user_id in item_list:
                continue
            item_list[data.user_id] = item_index
            item_index += 1
        print("Done!!!")
        return item_list, item_index   

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
                writer_id = content_id[:content_id.find("_")]
                key = (user_id, writer_id)
                if key not in read_count:
                    read_count[key] = 0
                read_count[key] += 1
        
        following_list = self.__following_list
        user_array = []
        item_array = []
        rate_array = []
        print("[+] setting rating table......")

        for key, val in tqdm(read_count.items()):
            user_id = key[0]
            writer_id = key[1]
            rate = val

            if user_id not in user_list:
                user_list[user_id] = self.__user_index
                self.__user_index += 1

            if writer_id not in item_list:
                item_list[writer_id] = self.__item_index
                self.__item_index += 1

            user_array.append(user_list[user_id])
            item_array.append(item_list[writer_id])

            if user_id in following_list and writer_id in following_list[user_id]:
                rate_array.append(rate * following_rate)
            else:
                rate_array.append(rate)

        rate_table = {"user_id": user_array, "item_id": item_array, "rate": rate}
        rate_table = pd.DataFrame(rate_table)
        
        self.__user_list = user_list
        self.__item_list = item_list
        print("Done!!!")
        return rate_table

    def __get_content_by_writer(self):
        print("[Info] DataProcessing : get content by writer....", end="")
        metadata = self.__read_data.metadata
        
        content_by_writer = {}
        for i, data in metadata.iterrows():
            content_id = data.id
            user_id = data.user_id
            if user_id not in content_by_writer:
                content_by_writer[user_id] = []
            content_by_writer[user_id].append(content_id)

        print("Done!!!")
        return content_by_writer

    def __get_user_read_list(self):
        print("[Info] DataProcessing : get user read list....")
        read_data = self.__read_data.read

        user_read_list = {}
        for i, data in tqdm(read_data.iterrows(), total=read_data.shape[0]):
            user_id = data.user_id
            if user_id not in user_read_list:
                user_read_list[user_id] = set()
            content_list = data.content_id
            for content_id in content_list:
                user_read_list[user_id].add(content_id)
        print("Done!!!")
        return user_read_list

    def __get_count_view_list(self):
        print("[Info] DataProcessing : get count view list....")
        read_data = self.__read_data.read

        count_view_list = {}
        for i, data in tqdm(read_data.iterrows(), total=read_data.shape[0]):
            content_list = data.content_id
            for content_id in content_list:
                if content_id not in count_view_list:
                    count_view_list[content_id] = 0
            count_view_list[content_id] += 1
        
        count_view_list = sorted(count_view_list, key=lambda x : x[1], reverse=True)
        print("Done!!!")
        return count_view_list

    @property
    def following_list(self):
        return self.__following_list

    @property
    def user_list(self):
        return self.__user_list

    @property
    def item_list(self):
        return self.__item_list

    @property
    def rate_table(self):
        return self.__rate_table
    
    @property
    def content_by_writer(self):
        return self.__content_by_writer

    @property
    def user_read_list(self):
        return self.__user_read_list
    
    @property
    def count_view_list(self):
        return self.__count_view_list
# %%
