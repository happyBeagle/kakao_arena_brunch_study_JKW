import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.sasrec.utils import generate, load_txt
from config import get_config
from model.model import SASRec
from utils import iterate_data_files
# python3 main.py --device=cuda --dataset=dev1 --train_dir=default_ --state_dict_path='/content/drive/MyDrive/kaggle/train_default/SASRec.epoch=70.lr=0.001.layer=2.head=1.hidden=512.maxlen=200.pth' --inference_only=true --maxlen=200
import pandas as pd 
from tqdm import tqdm
import pickle
import torch

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

def inference_dev1(config):
    user_train, usernum, itemnum = load_txt('dev1')
    model = SASRec(505840, config).to(config.SYSTEM.DEVICE) # no ReLU activation in original SASRec implementation?
    model.load_state_dict(torch.load(config.PATH.TEST)['model_state_dict'])
    predictions = generate(model, user_train, config)
    with open('./r_1', 'wb') as f:
        pickle.dump(predictions, f)
    print(predictions)
    return predictions

def inference_dev2(config):
    import gensim
    model_256 = gensim.models.KeyedVectors.load_word2vec_format('/content/drive/MyDrive/kaggle/train_default/w2v_256')
    dev2_df = pd.read_csv('/content/drive/MyDrive/kaggle/data/dev2.txt', sep=' ', header=None)
    dev2_df.rename(columns={0:'user_id', 1:'content_id'}, inplace=True)
    return predict_n(100, model_256, dev2_df)

def inference_dev3(config):

    data = []

    files = sorted([path for path, _ in iterate_data_files('2019022200', '2019030100')])

    for path in tqdm(files, mininterval=1):
        for line in open(path):
            tokens = line.strip().split()
            read_datetime = path[-21:-11]
            user_id = tokens[0]
            reads = tokens[1:]
            for item in reads:
                data.append([read_datetime, user_id, item])

    read_df = pd.DataFrame(data)
    read_df.columns = ['date', 'user_id', 'article_id']
    most_visited_article = read_df.groupby('article_id').count().user_id.sort_values(ascending=False).index[:100]
    with open('./data/dev3', 'rb') as f:
        dev3_users = pickle.load(f)
    r3 = {}
    for user in dev3_users:
        r3[user] = list(most_visited_article)
    return r3

def save_predictions(prediction):
    # 순서 맞춰서 저장
    dev_user_path = '/content/drive/MyDrive/kaggle/data/predict/dev.users'
    dev_user_data = []

    with open(dev_user_path, 'r') as f:
        for line in f:
            line = line.strip()
            dev_user_data.append(line)
        with open('./recommend.txt', 'w') as wf:
            for user in dev_user_data:
                c = ' '.join(prediction[user])
                wf.write(f'{user} {c}\n') 

def main(config):
    predictions_1 = inference_dev1(config)
    # predictions_2 = inference_dev2(config)
    # predictions_3 = inference_dev3(config)
    # predictions_1.update(predictions_2)
    # predictions_1.update(predictions_3)
    save_predictions(predictions_1)

if __name__ == '__main__':
    config = get_config()
    main(config)