import os
import numpy as np
from tqdm.auto import tqdm
import user2vec
import preprocessing

def cosine_similarity(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def find_new_articles(viewer):
    recommends = []

    reader_vector, writer_vector = user2vec(dev_user_file_path)

    if sum(reader_vector[viewer]) == 0:
        recommends = top100
    else:
        writer_cosine_similarity = []

        for writer in writer_vector.keys():
            writer_cosine_similarity.append((writer,cosine_similarity(reader_vector[viewer],writer_vector[writer])))

        writer_cosine_similarity = sorted(writer_cosine_similarity, key=lambda x : x[1], reverse=True)

        for writer_cosine in writer_cosine_similarity:
            recommends.extend(writer_write_list_dup[writer_cosine[0]])
            if len(recommends) >= 100:
                break

        recommends = recommends[:100]

    f.write('%s %s\n' % (viewer, ' '.join(recommends)))


if __name__ == '__main__':
    os.chdir('../../')
    contents_file_path = 'data/contents'
    dev_user_file_path = 'data/predict/dev.users'
    test_user_file_path = 'data/predict/test.users'
    read_file_path = 'data/read'
    meta_file_path = 'data/metadata.json'
    recommend_file_path = 'recommend'

    writer_write_list_dup = preprocessing.writer_write_contents_dup(meta_file_path)

    top100 = preprocessing.view_top_100(read_file_path)

    f = open(recommend_file_path + '/recommend.txt', 'w')

    for id in tqdm(preprocessing.users_data(dev_user_file_path)):
        find_new_articles(id)

    f.close()