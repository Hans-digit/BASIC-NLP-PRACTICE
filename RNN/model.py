import torch
import time
import torch.nn.functional as F
import numpy as np


#batch_size     : size of mini batch
#text_size      : size of word of text
#input_size     : size of RNN input data
#hidden_size    : size of RNN hidden layer data
#num_layers     : depth of RNN

class RNN_imdb(torch.nn.Module):
    def __init__(self, vocabulary_len, batch_size, text_size, input_size, hidden_size, num_layers, bidirectional=False):
        super(RNN_imdb, self).__init__()
        self.vocabulary_len = vocabulary_len
        self.batch_size = batch_size
        self.text_size = text_size
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(vocabulary_len+1, input_size, padding_idx= vocabulary_len)
        self.RNN = torch.nn.RNN(input_size = input_size,
                                hidden_size = hidden_size,
                                num_layers = num_layers,
                                bidirectional = bidirectional)
        if bidirectional == True:
            self.Linear1 = torch.nn.Linear(in_features = hidden_size * 2, out_features = 2)
        else:
            self.Linear1 = torch.nn.Linear(in_features=hidden_size, out_features=2)
        self.Linear2 = torch.nn.Linear(in_features = text_size * 2,
                                       out_features = 2)


    def forward(self, context):
        emb_context = self.embedding(context)
        rnn_hidden_state = self.RNN(emb_context)[0]
        linear_text = self.Linear1(rnn_hidden_state)
        linear_text = linear_text.view([self.batch_size, self.text_size*2])
        linear_text = self.Linear2(linear_text)
        linear_text = F.softmax(linear_text, dim=1)
        return linear_text




