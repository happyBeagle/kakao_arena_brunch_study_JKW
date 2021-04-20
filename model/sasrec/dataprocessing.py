#%%
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
# from readrawdata import ReadRawData
from preprocessing.readrawdata import ReadRawData
# %%
class Preprocessing_sasrec():
    def __init__(self, dir_path):
        self.read_raw_data = ReadRawData(dir_path)

        self.read_df = self._convert_read_data()
        
        self.user2num, self.content2num, self.num2user, self.num2content = \
            self._make_save_mappings('/content/drive/MyDrive/brunch/kakao_arena_brunch_study_JKW/model/sasrec/tmp')
        
        self.dev_users, self.test_users = self._load_test_users()

        self.train_df, self.dev_df, self.test_df = self._divide_read_df()

        self.train_df = self._preprocess_train()

        self._save_dev_read_list()

        # self.read_dev_df, self.unread_dev_df = self._divide_dev_df()

        self._save_dev_dfs()

    def _convert_read_data(self):
        f1 = lambda x: x[:10]
        self.read_raw_data.read['file_name'] = self.read_raw_data.read['file_name'].map(f1)

        data = []
        for _, row in tqdm(self.read_raw_data.read.iterrows()):
            for content in row.content_id:
                data.append([row.file_name, row.user_id, content])

        read_df = pd.DataFrame(data)
        read_df.columns = ['date', 'user_id', 'content_id']
        return read_df
    
    def _make_save_mappings(self, dir_path):
        user2num = {id: i+1 for i, id in enumerate(set(self.read_df['user_id']))}
        self.read_df['user_num'] = self.read_df.user_id.map(user2num)
        content2num = {id: i+1 for i, id in enumerate(set(self.read_df['content_id']))}
        self.read_df['content_num'] = self.read_df.content_id.map(content2num)   

        num2user = {v: k for k,v in user2num.items()}
        num2content = {v: k for k,v in content2num.items()}
        mappings = {'user2num': user2num,
                    'content2num': content2num,
                    'num2user': num2user,
                    'num2content': num2content}
        for k, v in mappings.items():
            file_path = os.path.join(dir_path, k)
            with open(file_path, 'wb') as lf:
                pickle.dump(v, lf)
        
        return mappings.values()
    
    def _load_test_users(self):
        # Load dev users
        # dev_user_path = '/content/drive/MyDrive/kaggle/data/predict/dev.users'
        # dev_user_data = []
        # with open(dev_user_path, 'r') as f:
        #     for line in f:
        #         line = line.strip()
        #         dev_user_data.append(line)

        # # Load dev users
        # test_user_path = '/content/drive/MyDrive/kaggle/data/predict/test.users'
        # test_user_data = []

        # with open(test_user_path, 'r') as f:
        #     for line in f:
        #         line = line.strip()
        #         test_user_data.append(line)
        test_user_data = self.read_raw_data.test_users
        dev_user_data = self.read_raw_data.dev_users
        self.nn_dev_user = [] # cold-start / 아예 읽은 기록이 없음
        def dev_id2num(id):
            try:
                return self.user2num[id]
            except KeyError:
                self.nn_dev_user.append(id)
        
        nn_test_user = []
        def test_id2num(id):
            try:
                return self.user2num[id]
            except KeyError:
                nn_test_user.append(id)

        dev_users = [dev_id2num(id) for id in dev_user_data if dev_id2num(id) is not None]
        test_users = [test_id2num(id) for id in test_user_data if test_id2num(id) is not None]
            
        return dev_users, test_users
    
    def _divide_read_df(self):
        # read_df 를 train_df, dev_df, test_df로 나눈다. 
        read_df = self.read_df[['user_num', 'date', 'content_num']]

        dev_df = read_df[read_df.user_num.isin(self.dev_users)]
        test_df = read_df[read_df.user_num.isin(self.test_users)]
        train_df = read_df[~read_df.user_num.isin(self.dev_users + self.test_users)]

        return train_df, dev_df, test_df
    
    def _save_dev_read_list(self):
        # dev.user 들이 어떤 글을 읽었는지를 dictionary 형태로 저장. 추천할때 중복글 제거하기 위해.
        dev_read_list = {}
        for user_num, group_data in self.dev_df[['user_num', 'content_num']].groupby('user_num'):
            dev_read_list[user_num] = list(group_data.content_num.unique())

        with open('/content/drive/MyDrive/brunch/kakao_arena_brunch_study_JKW/model/sasrec/tmp/dev_read_list', 'wb') as lf:
            pickle.dump(dev_read_list, lf)

    def _divide_dev_df(self):
        # dev user를 중복기간전에 읽은 기록 유무를 기준으로 나눈다. -> 있다면 transformer로 추천하고 없으면 유사도를 이용해서 추천.
        dev_count = self.dev_df.groupby(['user_num','date']).count()
        dev_count = dev_count.reset_index(level='date')
        dev_count['date'] = pd.to_numeric(dev_count['date'])

        df_q = dev_count.query("date < 2019022200")

        read_dev_users = set(df_q.index)
        unread_dev_users = set(self.dev_df.user_num.unique()) - read_dev_users

        # 결과적으로 dev.user를 세 종류로 나눴다.
        # read_dev_users -> transformer이용
        # unread_dev_users -> 과거에 읽은 기록은 없고 중복기간에만 읽은 기록이 있다. 유사도 이용해서 추천.          구독자..     
        # nn_dev_users -> 아예 기록이 없음. 그냥 많이 읽은거 추천.                                         22일 이후 기록으로만. 
        return read_dev_users, unread_dev_users
    
    def _preprocess_train(self):
        train_df = self.train_df.groupby('user_num').apply(lambda row: row['content_num'].values)
        def remove_dup_content(row):
            cmp_list = np.roll(row, 1)
            sub = row - cmp_list 
            row = np.delete(row, np.where(sub == 0)[0], axis=0)
            return row
        train_df = train_df.apply(lambda x: remove_dup_content(x))
        train_df = train_df[train_df.apply(lambda x: len(x)) > 10]
        return train_df

    def _save_dev_dfs(self):
        read_dev_users, unread_dev_users = self._divide_dev_df()
        # 예측 1. transformer를 이용한 예측을 위한 data
        dev1_df = self.dev_df[self.dev_df.user_num.isin(read_dev_users)]
        dev1_df = dev1_df.sort_values(by=['user_num', 'date'])
        dev1_df[['user_num', 'content_num']].to_csv('/content/drive/MyDrive/brunch/data/dev1.txt', sep=' ', index=False, header=False)

        # 예측 2. 중복기간내에 읽은 글과 유사한 글 추천을 위한 data
        dev2_df = self.dev_df[self.dev_df.user_num.isin(unread_dev_users)]
        dev2_df = dev2_df.sort_values(by=['user_num', 'date'])
        dev2_df[['user_num', 'content_num']].to_csv('/content/drive/MyDrive/brunch/data/dev2.txt', sep=' ', index=False, header=False)
        
        # 예측 3. 기록이 아예 없어서 단순 많이 읽은 글 추천. list 형태로 아이디만 저장
        with open('/content/drive/MyDrive/brunch/data/dev3', 'wb') as lf:
            pickle.dump(self.nn_dev_user, lf)

Preprocessing_sasrec('/content/drive/MyDrive/brunch/data')