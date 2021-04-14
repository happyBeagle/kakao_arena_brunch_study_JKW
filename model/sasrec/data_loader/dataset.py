import random
import torch
class BrunchDataset(torch.utils.data.Dataset):
    def __init__(self, df, maximum_content_num, max_len=100):
        self.user_nums = []
        self.df = df
        self.maximum_content_num = maximum_content_num
        self.max_len = max_len
        for user_num in df.index.values:
            self.user_nums.append(user_num)
    
    def __len__(self):
        return len(self.user_nums)
    
    def __getitem__(self, idx):
        user_num = self.user_nums[idx]
        content_list = self.df.loc[user_num]
        seq_len = len(content_list)

        neg = random.sample(set(range(0, self.maximum_content_num+1)) - set(content_list), seq_len)

        content_list_ = torch.as_tensor(content_list, dtype=int)
        neg_ = torch.as_tensor(neg, dtype=int)

        content_list = torch.zeros(self.max_len, dtype=int)
        pos = torch.zeros(self.max_len, dtype=int)
        neg = torch.zeros(self.max_len, dtype=int)
        
        src_mask = torch.ones(self.max_len, dtype=bool)
        src_mask[-seq_len+1:] = False 

        if seq_len > self.max_len:
            content_list[:] = content_list_[-self.max_len-1:-1]
            pos[:] = content_list_[-self.max_len:]
            neg[:] = neg_[-self.max_len:]
        else:
            content_list[-seq_len+1:] = content_list_[:-1]
            pos[-seq_len+1:] = content_list_[1:]
            neg[-seq_len+1:] = neg_[1:]
        return content_list, pos, neg, src_mask
