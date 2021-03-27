# %%
import pandas as pd 
import numpy as np
import os 
import sys  

from tqdm import tqdm
from collections import defaultdict  

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import utils
from preprocessing.singleton import Singleton

# %%
class ReadRawData(metaclass=Singleton):
    def __init__(self, dir_path):
        if not os.path.exists(dir_path):
            print("[Error] Preprocessing : dir_path is not exist...")
            return
        self.__dir_path = dir_path
        self.__metadata = self.get_metadata()
        self.__users = self.get_users()
        self.__magazine = self.get_magazine()
        self.__raed = self.get_read()
        self.__dev_users = self.get_dev_users_data()
        self.__test_users = self.get_test_users_data()

    def get_metadata(self):
        print("[Info] Preprocessing : get metadata...", end="")
        file_path = os.path.join(self.__dir_path,  "metadata.json")
        if not utils.check_file(file_path, ".json", "Preprocessing : get_metadata"):
            return
        metadata = pd.read_json(file_path, lines=True)
        print("Done!!")
        return metadata

    def get_users(self):
        print("[Info] Preprocessing : get users...", end="")
        file_path = os.path.join(self.__dir_path, "users.json")
        if not utils.check_file(file_path, ".json", "Preprocessing : get_users"):
            return
        users = pd.read_json(file_path, lines=True)
        print("Done!!")
        return users

    def get_magazine(self):
        print("[Info] Preprocessing : get magazine...", end="")
        file_path = os.path.join(self.__dir_path, "magazine.json")
        if not utils.check_file(file_path, ".json", "Preprocessing : get_magazine"):
            return
        magazine = pd.read_json(file_path, lines=True)
        print("Done!!")
        return magazine

    def preprocessing_read(self):
        print("[Info] Preprocessing : preprocessing_read Start...")
        file_path = os.path.join(self.__dir_path, "read/")

        if not os.path.exists(file_path):
            print(f"[Error] Preprocessing : preprocessing_read...file is not exist")
            return False
        
        # delete .2019010120_2019010121.un~
        # get file list in read directory
        read_file_list = os.listdir(file_path)  
        read_list = []  # file_name : file name
                        # user_id : user id
                        # content_id : written letter id

        for file_name in tqdm(read_file_list):
            tmp_path = os.path.join(file_path, file_name)
            with open(tmp_path, 'r') as f:
                for line in f:
                    line_list = line.split()
                    read_list.append({"file_name":file_name,"user_id":line_list[0],"content_id":line_list[1:]})
                    
        read_list = pd.DataFrame(read_list)
        read_list.to_pickle(file_path[:-1] + ".pkl")

        print("[Info] Preprocessing : preprocess_read_data...Done!!!")
    
        return True

    def get_read(self):
        print("[Info] Preprocessing : get read...", end="")
        file_path = os.path.join(self.__dir_path, "read.pkl")
        if not utils.check_file(file_path, ".pkl", "Preprocessing : get_read"):
           if not self.preprocessing_read():
               return
            
        read = pd.read_pickle(file_path)
        print("Done!!")
        return read

    def get_dev_users_data(self):
        # print("[Info] Preprocessing : get dev users data...", end="")
        users_list = []
        file_path = os.path.join(self.__dir_path, "predict", 'dev.users')
        with open(file_path, 'r') as f:
            for line in f:
                users_list.append(line.strip())
        # print("Done!!")
        return users_list

    def get_test_users_data(self):
        # print("[Info] Preprocessing : get test users data...", end="")
        users_list = []
        file_path = os.path.join(self.__dir_path, "predict", 'test.users')
        with open(file_path, 'r') as f:
            for line in f:
                users_list.append(line.strip())
        # print("Done!!")
        return users_list

    @property
    def metadata(self):
        return self.__metadata

    @property
    def users(self):
        return self.__users

    @property
    def magazine(self):
        return self.__magazine

    @property
    def read(self):
        return self.__read

    @property
    def dev_users(self):
        return self.__dev_users

    @property
    def test_users(self):
        return self.__test_users
# %%
