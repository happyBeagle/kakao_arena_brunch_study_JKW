# %%
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from preprocessing.readrawdata import ReadRawData
from collections import defaultdict, Counter

# %%
class Preprocessing_user2vec():
    def __init__(self, dir_path):
        if not os.path.exists(dir_path):
            print("[Error] Preprocessing : dir_path is not exist...")
            return
        self.__dir_path = dir_path
        self.read_raw_data = ReadRawData(dir_path)

    def user_read_contents(self):
        # print("[Info] Preprocessing : user read contents...")
        user_read_list = {i : [] for i in self.read_raw_data.get_dev_users_data()}
        for file_name in os.listdir(self.__dir_path + '/read'):
            tmp_path = os.path.join(self.__dir_path + '/read', file_name)
            with open(tmp_path, 'r') as f:
                for line in f:
                    line_list = line.split()
                    try:
                        user_read_list[line_list[0]].extend(line_list[1:])
                    except:
                        pass
        # print("Done!!")
        return user_read_list

    def writer_write_contents(self):
        # print("[Info] Preprocessing : writer write contents...")
        writer_write_list = defaultdict(list)
        for file_name in os.listdir(self.__dir_path + '/contents'):
            print('.',end='')
            file_path = os.path.join(self.__dir_path, 'contents', file_name)
            for line in open(file_path,'r'):
                tmp = json.loads(line)['id']
                writer_write_list[tmp.split('_')[0]].append(tmp)
        # print("Done!!")
        return writer_write_list

    def user_read_contents_dup(self):
        # print("[Info] Preprocessing : user read contents...")
        user_read_list_dup = {i : [] for i in self.read_raw_data.get_dev_users_data()}
        for file_name in os.listdir(self.__dir_path + '/read'):
            if int(file_name[:8]) >= 20190222:
                tmp_path = os.path.join(self.__dir_path + '/read', file_name)
                with open(tmp_path, 'r') as f:
                    for line in f:
                        line_list = line.split()
                        try:
                            user_read_list_dup[line_list[0]].extend(line_list[1:])
                        except:
                            pass
        # print("Done!!")
        return user_read_list_dup

    def writer_write_contents_dup(self):
        meta_data = self.read_raw_data.get_metadata()
        meta_data = meta_data[meta_data['reg_ts'] >= 1550761200000]
        meta_data = meta_data[meta_data['reg_ts'] < 1552575600000]
        meta_data = meta_data.sort_values(by=['reg_ts'], ascending=True)
        writer_write_list_dup = defaultdict(list)
        for num in range(len(meta_data)):
            writer_write_list_dup[meta_data.iloc[num, 1]].append(meta_data.iloc[num, -1])
        return writer_write_list_dup

    def cold_start(self):
        view = []
        for file_name in os.listdir(self.__dir_path + '/read'):
            if int(file_name[:8]) >= 20190222:
                tmp_path = os.path.join(self.__dir_path + '/read', file_name)
                with open(tmp_path, 'r') as f:
                    for line in f:
                        line_list = line.split()[1:]
                        view.extend(line_list)
        top100 = []
        for content in Counter(view).most_common(102):
            if content[0].split('_')[0] != '@brunch':
                top100.append(content[0])
        return top100