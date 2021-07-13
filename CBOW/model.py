#Using CBOW, train W2V
import torch
import time
import torch.nn.functional as F
import numpy as np

class CBOW(torch.nn.Module):
    def __init__(self, D_dim, W_dim):
        super(CBOW, self).__init__()
        self.dict_len = D_dim
        self.Embedding = torch.nn.Embedding(D_dim+1, W_dim, padding_idx = D_dim)
        self.Linear = torch.nn.Linear(W_dim, D_dim)
        # self.activation_function = torch.nn.LogSoftmax(dim = 1)

    def forward(self, context_word):
        emb_context = self.Embedding(context_word)
        emb_context = torch.sum(emb_context, dim = 1)/4
        emb_context = self.Linear(emb_context)
        # emb_context = self.activation_function(emb_context)

        return emb_context
