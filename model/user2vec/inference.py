# %%
import pickle
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from preprocessing.readrawdata import ReadRawData
from dataprocessing import Preprocessing_user2vec
from model.user2vec.user2vec import User2Vec
import utils

from tqdm import tqdm
import numpy as np

# %%
def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

class Inference():
    def __init__(self, dir_path):
        if not os.path.exists(dir_path):
            print("[Error] Preprocessing : dir_path is not exist...")
            return
        self.__dir_path = dir_path
        self.read_raw_data = ReadRawData(dir_path)
        self.preprocessing_user2vec = Preprocessing_user2vec(dir_path)
        self.user2vec = User2Vec(dir_path)

        file_path = os.path.join(os.getcwd(),'users2vec.pickle')
        if not utils.check_file(file_path, ".pickle", "Training : get_users2vec..."):
            self.user2vec.get_users2vec()
        with open(file_path, 'rb') as f:
            self.user_vector = pickle.load(f)

        file_path = os.path.join(os.getcwd(), 'writers2vec.pickle')
        if not utils.check_file(file_path, ".pickle", "Training : get_writers2vec"):
            self.user2vec.get_writers2vec()
        with open(file_path, 'rb') as f:
            self.writer_vector = pickle.load(f)

        self.writer_write_lists_dup = self.preprocessing_user2vec.writer_write_contents_dup()
        self.user_read_lists_dup = self.preprocessing_user2vec.user_read_contents_dup()

    def find_new_articles(self,viewer):
        recommends_writer = []

        try:
            viewer_following_list = self.read_raw_data.users[self.read_raw_data.users['id'] == viewer]['following_list'].to_list()[0]
            recommends_writer.extend(viewer_following_list)
        except:
            pass

        if sum(self.user_vector[viewer]) != 0:
            writer_cosine_similarity = []
            for writer in self.writer_vector.keys():
                writer_cosine_similarity.append((writer,cosine_similarity(self.user_vector[viewer],self.writer_vector[writer])))
            writer_cosine_similarity = [i[0] for i in sorted(writer_cosine_similarity, key=lambda x: x[1], reverse=True)]
            for i in recommends_writer:
                try:
                    writer_cosine_similarity.remove(i)
                except:
                    pass
            recommends_writer.extend(writer_cosine_similarity)

        recommends = []

        for writer in recommends_writer:
            tmp = self.writer_write_lists_dup[writer]
            for i in self.user_read_lists_dup[viewer]:
                try:
                    tmp.remove(i)
                except:
                    pass
            recommends.extend(tmp)
            if len(recommends) >= 100:
                break

        if len(recommends) >= 100:
            recommends = recommends[:100]
        else:
            recommends += self.preprocessing_user2vec.cold_start()[:100 - len(recommends)]

        return recommends

    def user_recommend(self):
        f = open("../../recommend/recommend.txt", 'w')
        for id in tqdm(self.read_raw_data.get_dev_users_data()):
            f.write('%s %s\n' % (id, ' '.join(self.find_new_articles(id))))
        f.close()

# %%
dir_path = "../../data/"

tmp = Inference(dir_path)
tmp.user_recommend()