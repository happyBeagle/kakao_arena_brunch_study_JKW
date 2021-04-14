user_train, usernum, itemnum = load_txt(args.dataset)

num_batch = len(user_train) // args.batch_size # tail? + ((len(user_train) % args.batch_size) != 0)

cc = 0.0
for u in user_train:
    cc += len(user_train[u])
print('average sequence length: %.2f' % (cc / len(user_train)))

f = open(os.path.join(args.dataset + '_' + args.train_dir, 'log.txt'), 'w')

model = SASRec(usernum, itemnum, args).to(args.device) # no ReLU activation in original SASRec implementation?

epoch_start_idx = 1
if args.state_dict_path is not None:
    model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
    try:
        print('Loading state_dicts...')
        model.load_state_dict(torch.load(args.state_dict_path, map_location=torch.device(args.device)))
        print('Done')
        tail = args.state_dict_path[args.state_dict_path.find('epoch=') + 6:]
        epoch_start_idx = int(tail[:tail.find('.')]) + 1
    except: # in case your pytorch version is not 1.6 etc., pls debug by pdb if load weights failed
        print('failed loading state_dicts, pls check file path: ')
        print(args.state_dict_path)
        print('pdb enabled for your quick check, pls type exit() if you do not need it')
        import pdb; pdb.set_trace()

    model.eval()
    predictions = generate(model, user_train, args)

    with open('./r_1', 'wb') as f:
        pickle.dump(predictions, f)


#%%
from utils import load_mappings
import torch 
import pandas as pd
import numpy as np
article2num, num2article, num2user, user2num = load_mappings('./mapping')

def load_transformer(usernum, itemnum):
    pass
# dev1 
#%%
import pickle

with open('./r_1', 'rb') as f:
    r_1 = pickle.load(f)
#%%
dev_read_list = {}
dev_user_read_path = './data/dev_read_list'
with open(dev_user_read_path, 'rb') as f:
    dev_read_list_ = pickle.load(f)

for k, v in dev_read_list_.items():
    dev_read_list[num2user[k]] = [num2article[article] for article in v]

#%%
# dev2
import gensim
model_256 = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/MyDrive/kaggle/train_default/w2v_256')
# model_512 = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/MyDrive/kaggle/train_default/w2v_512')
# model_128 = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/MyDrive/kaggle/train_default/w2v_128')

# %%
# dev1_df = pd.read_csv('/content/drive/MyDrive/kaggle/data/dev1.txt', sep=' ', header=None)
# dev1_df.rename(columns={0:'user_id', 1:'article_id'}, inplace=True)

dev2_df = pd.read_csv('/content/drive/MyDrive/kaggle/data/dev2.txt', sep=' ', header=None)
dev2_df.rename(columns={0:'user_id', 1:'article_id'}, inplace=True)

#%%
# def predict(model, dim, user_df):
#     prediction_t = {}
#     dev_user_group = user_df.groupby(['user_id'])
#     for u, group_data in dev_user_group:
#         articles = list(group_data.article_id.unique())
#         vector = np.zeros((dim, ))
#         i = 0
#         for article in articles:
#             try:
#                 vector += model.wv.get_vector(str(article))
#             except KeyError:
#                 i += 1
#             vector = vector / (len(articles) + i)
#         top_300 = model.wv.similar_by_vector(vector, topn=300) 
#         prediction = [i[0] for i in top_300 if int(i[0]) not in articles][:100]
#         user = num2user[u]
#         prediction_t[user] = [num2article[int(i)] for i in prediction if i != '505841']
#     return prediction_t

# #%%
# prediction_256_1 = predict(model_256, 256, dev1_df)
# prediction_512_1 = predict(model_512, 512, dev1_df)

# #%%
# prediction_256_2 = predict(model_256, 256, dev2_df)
# prediction_512_2 = predict(model_512, 512, dev2_df)


# #%%
# from tqdm import tqdm
# def predict_n(n, model_w, user_df):
#     predictions = {}
#     dev_user_group = user_df.groupby(['user_id'])
#     for u, group_data in tqdm(dev_user_group):
#         articles = list(group_data.article_id.unique())
#         check_list = set(articles)
#         prediction = []
#         i = 0
#         while len(prediction) < 100:
#             for article in articles[::-1]:
#                 top_p = model_w.wv.most_similar(str(article), topn=(i+1)*10)[i:i+10] 
#                 # print(top_p)
                
#                 top_p = [j[0] for j in top_p if int(j[0]) not in check_list]
#                 prediction.extend(top_p)
#                 check_list.update(set(top_p))
#             i += 1
#                 # print(prediction)
#         predictions[num2user[u]] = prediction[:100]
#     return predictions

#%%
#%%
from tqdm import tqdm
def predict_n(n, model_w, user_df):
    predictions = {}
    dev_user_group = user_df.groupby(['user_id'])
    for u, group_data in tqdm(dev_user_group):
        articles = list(group_data.article_id.unique())
        check_list = set(articles)
        num_articles = len(articles)
        p = n // num_articles
        prediction = []
        prev = 0
        now = p
        
        while len(prediction) < 100:
            now += 3
            # print(prev, now)
            for article in articles[::-1]:
                top_p = model_w.wv.most_similar(str(article), topn=now)[prev:now] 
                top_p_ = [j[0] for j in top_p if int(j[0]) not in check_list]
                test = [j[0] for j in top_p if int(j[0]) in check_list]
                if test:
                    print(test)
                prediction.extend(top_p_)
                check_list.update(set(top_p_))
                if len(set(prediction)) > 100:
                    break
            prev = now
        
        prediction = list(set(prediction))    
        predictions[num2user[u]] = [num2article[int(i)] for i in prediction[:100]]
    return predictions
#%%
# r1 = predict_n(100, model_128, dev1_df)
r2 = predict_n(100, model_256, dev2_df)
#%%
# r 분석
from collections import Counter
for k,v in r2.items():
    a_list = sorted(v)
    a_list2 = list(map(lambda x: x.split('_')[0][1:], a_list))
    # from collections import Counter 
    
    print(Counter(a_list2).most_common(1))
    print()
#%% 
def similar2article(article, model):
    article_list = model.most_similar(str(article2num[article]), topn=500)
    # print(article_list)
    article_list = [num2article[int(article[0])] for article in article_list]
    # article_list = list(map(lambda x: x.split('_')[0][1:], article_list))
    # print(Counter(article_list).most_common())
    # print(len(article_list))
    return article_list

#%%
import pandas as pd
import matplotlib as plt
import seaborn as sns

metadata_df = pd.read_json('/content/drive/MyDrive/kaggle/data/metadata.json', lines=True)


#%%
def similar_keyword(article, model):
    
    article_list = similar2article(article, model)
    # print(article_list)
    a_list2 = list(map(lambda x: x.split('_')[0][1:], article_list))
    # from collections import Counter 
    
    print(Counter(a_list2).most_common()[:2])
    print(metadata_df[metadata_df.id == article].keyword_list)
    print(metadata_df[metadata_df.id.isin(article_list)][['id','keyword_list']])
#%%   

#%%
# dev3
import os, sys
import tqdm

def iterate_data_files(from_dtm, to_dtm):
    from_dtm, to_dtm = map(str, [from_dtm, to_dtm])
    read_root = os.path.join('/content/drive/MyDrive/kaggle/data', 'read')
    for fname in os.listdir(read_root):
        if len(fname) != 21:
            continue 
        if from_dtm != 'None' and from_dtm > fname:
            continue 
        if to_dtm != 'None' and fname > to_dtm:
            continue 
        path = os.path.join(read_root, fname)
        yield path, fname 

data = []

files = sorted([path for path, _ in iterate_data_files('2019022200', '2019030100')])

for path in tqdm.tqdm(files, mininterval=1):
    for line in open(path):
        tokens = line.strip().split()
        read_datetime = path[-21:-11]
        user_id = tokens[0]
        reads = tokens[1:]
        for item in reads:
            data.append([read_datetime, user_id, item])
# %%
import pandas as pd 

read_df = pd.DataFrame(data)
read_df.columns = ['date', 'user_id', 'article_id']
# %%
most_visited_article = read_df.groupby('article_id').count().user_id.sort_values(ascending=False).index[:100]
# %%
import pickle
with open('./data/dev3', 'rb') as f:
    dev3_users = pickle.load(f)
# %%
r3 = {}
for user in dev3_users:
    r3[user] = list(most_visited_article)
# %%
new_r_1.update(r3)
new_r_1.update(r2)
# %%
#순서 맞추기
# Load dev userss
dev_user_path = '/content/drive/MyDrive/kaggle/data/predict/dev.users'
dev_user_data = []

with open(dev_user_path, 'r') as f:
    for line in f:
        line = line.strip()
        dev_user_data.append(line)
# %%
with open('./recommend.txt', 'w') as wf:
    for user in dev_user_data:
        prediction = ' '.join(new_r_1[user])
        wf.write(f'{user} {prediction}\n') 
# %%
