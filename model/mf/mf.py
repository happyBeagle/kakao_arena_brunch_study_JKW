import torch
import torch.nn as nn

class MF(nn.Module):
    def __init__(self, num_users, num_items, emb_size = 100):
        super(MF, self).__init__()
        self.users_emb = nn.Embedding(num_users, emb_size)
        self.items_emb = nn.Embedding(num_items, emb_size)
        self.users_emb.weight.data.uniform_(0, 0.5)
        self.items_emb.weight.data.uniform_(0, 0.5)

    def forward(self, u, v):
        u = self.users_emb(u)
        v = self.items_emb(v)

        return (u*v).sum(1)