import os
import json
import numpy as np
from tqdm.auto import tqdm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
import preprocessing

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

def user2vec(user_file_path):
    try:
        doc2vec_model = Doc2Vec.load(model_save_path + "/model.doc2vec")
    except:
        doc2vec_corpus = Doc2VecCorpus(contents_file_path)
        doc2vec_model = Doc2Vec(doc2vec_corpus)
        doc2vec_model.save(model_save_path + "/model.doc2vec")

    reader_read_list = preprocessing.reader_read_contents(user_file_path,read_file_path)
    reader_vector = {}
    for reader, read_list in tqdm(reader_read_list.items()):
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
        reader_vector[reader] = vector

    writer_write_list = preprocessing.writer_write_contents(contents_file_path)
    writer_vector = {}
    for writer, write_list in tqdm(writer_write_list.items()):
        vector = np.zeros(100)
        cnt = 0
        for contents in write_list:
            try:
                vector += doc2vec_model.docvecs[contents]
                cnt += 1
            except:
                pass
        if cnt != 0:
            vector /= cnt
        writer_vector[writer] = vector

    return reader_vector, writer_vector

if __name__ == '__main__':
    os.chdir('../../')
    contents_file_path = 'data/contents'
    dev_user_file_path = 'data/predict/dev.users'
    test_user_file_path = 'data/predict/test.users'
    read_file_path = 'data/read'
    model_save_path = 'model/user2vec'
