# %%
import json
import pickle

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from preprocessing.Preprocessing_user2vec import Preprocessing_user2vec
import utils

import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

# %%
class Doc2VecCorpus:
    def __init__(self, fname):
        self.fname = fname
    def __iter__(self):
        for file_name in os.listdir(self.fname):
            for line in open(self.fname+'/'+file_name, 'r'):
                tmp = json.loads(line)
                yield TaggedDocument(
                    words = [j for i in tmp["morphs"] for j in i],
                    tags = [tmp["id"]])

class User2Vec():
    def __init__(self, dir_path):
        if not os.path.exists(dir_path):
            print("[Error] Preprocessing : dir_path is not exist...")
            return
        self.__dir_path = dir_path
        self.preprocessing_user2vec = Preprocessing_user2vec(dir_path)

    def doc2vec(self):
        print("[Info] Training : doc2vec...", end="")
        file_path = os.path.join(os.getcwd(), "model.doc2vec")
        if not utils.check_file(file_path, ".doc2vec", "Training : doc2vec"):
            doc2vec_corpus = Doc2VecCorpus(self.__dir_path + 'contents')
            doc2vec_model = Doc2Vec(doc2vec_corpus)
            doc2vec_model.save(file_path)
            return doc2vec_model
        doc2vec_model = Doc2Vec.load(file_path)
        print("Done!!")
        return doc2vec_model

    def get_users2vec(self):
        print("[Info] Training : get_users2vec...", end="")
        file_path = os.path.join(os.getcwd(),'users2vec.pickle')
        doc2vec_model = Doc2Vec.load(os.path.join(os.getcwd(), "model.doc2vec"))
        user_read_list = self.preprocessing_user2vec.user_read_contents()
        user_vector = {}
        for user, read_list in user_read_list.items():
            vector = np.zeros(100)
            cnt = 0
            for contents in read_list:
                try:
                    vector += doc2vec_model.docvecs[contents]
                    cnt += 1
                except:
                    pass
            if cnt != 0:
                vector /= cnt
            user_vector[user] = vector
        with open(file_path, 'wb') as f:
            pickle.dump(user_vector, f, pickle.HIGHEST_PROTOCOL)
        print("Done!!")
        return user_vector

    def get_writers2vec(self):
        print("[Info] Training : get_writers2vec", end="")
        file_path = os.path.join(os.getcwd(),'writers2vec.pickle')
        doc2vec_model = Doc2Vec.load(os.path.join(os.getcwd(), "model.doc2vec"))
        writer_read_list = self.preprocessing_user2vec.writer_write_contents()
        writer_vector = {}
        for writer, read_list in writer_read_list.items():
            vector = np.zeros(100)
            cnt = 0
            for contents in read_list:
                try:
                    vector += doc2vec_model.docvecs[contents]
                    cnt += 1
                except:
                    pass
            if cnt != 0:
                vector /= cnt
            writer_vector[writer] = vector
        with open(file_path, 'wb') as f:
            pickle.dump(writer_vector, f, pickle.HIGHEST_PROTOCOL)
        print("Done!!")
        return writer_vector
