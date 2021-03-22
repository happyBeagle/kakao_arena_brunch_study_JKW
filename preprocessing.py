import os
import json
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict, Counter

def users_data(file_path):
    users_list = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            users_list.append(line)
    return users_list

def reader_read_contents(user_file_path,read_file_path):
    reader_read_list = {i : [] for i in users_data(user_file_path)}
    for file_name in tqdm(os.listdir(read_file_path)):
        tmp_path = os.path.join(read_file_path, file_name)
        with open(tmp_path, 'r') as f:
            for line in f:
                line_list = line.split()
                try:
                    reader_read_list[line_list[0]].extend(line_list[1:])
                except:
                    pass
    return reader_read_list

def writer_write_contents(contents_file_path):
    writer_write_list = defaultdict(list)
    for file_name in tqdm(os.listdir(contents_file_path)):
        for line in open(contents_file_path + '/' + file_name, 'r'):
            tmp = json.loads(line)['id']
            writer_write_list[tmp.split('_')[0]].append(tmp)
    return writer_write_list

def reader_read_writer_dup(user_file_path,read_file_path):
    reader_read_list = {i : [] for i in users_data(user_file_path)}
    for file_name in tqdm(os.listdir(read_file_path)):
        if int(file_name[:8]) >= 20190222:
            tmp_path = os.path.join(read_file_path, file_name)
            with open(tmp_path, 'r') as f:
                for line in f:
                    line_list = line.split()
                    try:
                        reader_read_list[line_list[0]].extend([i.split('_')[0] for i in line_list[1:]])
                    except:
                        pass
    for reader in reader_read_list.keys():
        reader_read_list[reader] = list(set(reader_read_list[reader]))
    return reader_read_list

def writer_write_contents_dup(meta_file_path):
    meta_data = pd.read_json(meta_file_path, lines=True)
    meta_data = meta_data[meta_data['reg_ts'] >= 1550761200000]
    meta_data = meta_data.sort_values(by=['reg_ts'], ascending=True)
    writer_write_list_dup = defaultdict(list)
    for num in range(len(meta_data)):
        writer_write_list_dup[meta_data.iloc[num,1]].append(meta_data.iloc[num,-1])
    return writer_write_list_dup

def view_top_100(read_file_path):
    view = []
    for file_name in tqdm(os.listdir(read_file_path)):
        tmp_path = os.path.join(read_file_path, file_name)
        with open(tmp_path, 'r') as f:
            for line in f:
                line_list = line.split()[1:]
                view.extend(line_list)
    top100 = []
    for content in Counter(view).most_common(116):
        if content[0].split('_')[0] != '@brunch':
            top100.append(content[0])
    return top100

if __name__ == '__main__':
    contents_file_path = 'data/contents'
    dev_user_file_path = 'data/predict/dev.users'
    test_user_file_path = 'data/predict/test.users'
    read_file_path = 'data/read'
    meta_file_path = 'data/metadata.json'
