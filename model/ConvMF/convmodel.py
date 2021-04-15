import torch
import torch.nn as nn
import torch.nn.functional as F  
import torch.optim as optim

import numpy as np 

from torch.autograd import Variable

import gc
class CNN(nn.Module):
    def __init__(self, output_dimension, vocab_size, emb_dimension, dropout_rate, max_len, n_filters, if_cuda):
        super(CNN, self).__init__()
        gc.collect()
        torch.cuda.empty_cache()

        self.max_len = max_len
        self.emb_dimension = emb_dimension
        self.if_cuda = if_cuda
        self.vanila_dimension = 2 * n_filters
        self.projection_dimension = output_dimension
        self.qual_conv_set = {}

        self.embedding = nn.Embedding(vocab_size + 1, emb_dimension)

        self.conv1 = nn.Sequential(
            nn.Conv1d(emb_dimension, n_filters, kernel_size = 3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 3 + 1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(emb_dimension, n_filters, kernel_size = 4),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 4 + 1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(emb_dimension, n_filters, kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=max_len - 5 + 1)
        )

        self.layer = nn.Linear(n_filters * 3, self.vanila_dimension)
        self.dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(self.vanila_dimension, self.projection_dimension)
    
    def forward(self, input):
        size = len(input)
        embed = self.embedding(input)

        embed = embed.view([len(embed), self.emb_dimension, -1])

        x = self.conv1(embed)
        y = self.conv2(embed)
        z = self.conv3(embed)

        flatten = torch.cat((x.view(size, -1), y.view(size, -1), z.view(size, -1)), 1)

        out = F.tanh(self.layer(flatten))
        out = self.dropout(out)
        out = F.tanh(self.output_layer(out))

        return out

class TrainCNN():
    def __init__(self, epochs, learning_rate, output_dimension, vocab_size, emb_dimension,
                 dropout_rate, max_len, n_filters, if_cuda):
        self.model = CNN(output_dimension=output_dimension,
                         vocab_size=vocab_size,
                         emb_dimension=emb_dimension,
                         dropout_rate=dropout_rate,
                         max_len=max_len,
                         n_filters=n_filters,
                         if_cuda=if_cuda)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.if_cuda = if_cuda

        if if_cuda:
            self.model = self.model.cuda()

    def train(self, batch_size, X_train, V):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            n_batch =  len(X_train) // batch_size
            for i in range(n_batch + 1):
                begin_idx, end_idx = i * batch_size, (i + 1) * batch_size
                
                if i < n_batch:
                    x = X_train[begin_idx:end_idx]
                    y = V[begin_idx:end_idx]
                else:
                    x = X_train[begin_idx:]
                    y = V[begin_idx:]

                x = Variable(torch.from_numpy(x.astype('int64')).long())
                y = Variable(torch.from_numpy(y))

                if self.if_cuda:
                    x, y = x.cuda(), y.cuda()

                optimizer.zero_grad()

                predict = self.model(x)

                loss = F.mse_loss(predict, y)
                loss.backward()
                optimizer.step()

    def get_projection(self, batch_size, X_train):
        output_list = np.array([], dtype="float")
        with torch.no_grad():
            n_batch =  len(X_train) // batch_size
            for i in range(n_batch + 1):
                begin_idx, end_idx = i * batch_size, (i + 1) * batch_size
                
                if i < n_batch:
                    x = X_train[begin_idx:end_idx]
                else:
                    x = X_train[begin_idx:]
                
                x = Variable(torch.from_numpy(x.astype('int64')).long())
                
                if self.if_cuda:
                    x = x.cuda()
                predict = self.model(x)
                if len(output_list) == 0:
                    output_list = predict.cpu().numpy()
                else:
                    output_list = np.concatenate((output_list, predict.cpu().numpy()), axis=0)
        return output_list