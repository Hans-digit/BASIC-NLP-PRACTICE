import torch
import time
import torch.nn.functional as F
import numpy as np


#ELMO code with depth 4
class ELMO(torch.nn.Module):
    def __init__(self, Dic_dim, Emb_dim):
        super(ELMO, self).__init__()
        self.dict_len = Dic_dim
        self.Embedding = torch.nn.Embedding(Dic_dim+1, Emb_dim, padding_idx = Dic_dim)
        #first start linear layer
        self.StartLinear = torch.nn.Linear(Emb_dim, Emb_dim)
        #upword linear layers
        self.Linear1 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.Linear2 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.Linear3 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.Linear4 = torch.nn.Linear(Emb_dim, Emb_dim)
        #horizontal linear layers
        self.HLinear1 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.HLinear2 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.HLinear3 = torch.nn.Linear(Emb_dim, Emb_dim)
        self.HLinear4 = torch.nn.Linear(Emb_dim, Emb_dim)
        #output linear layer
        self.OutLinear = torch.nn.Linear(Emb_dim, Dic_dim)
        # self.activation_function = torch.nn.LogSoftmax(dim = 1)

    def forward(self, input_tensor, output_tensor):
        emb_tensor = self.Embedding(input_tensor)

        #first hidden state
        hidden_tensor1 = self.StartLinear(emb_tensor)

        ##hidden state modify
        hidden_tensor1 = self.

        #..

        #fourth hidden state



