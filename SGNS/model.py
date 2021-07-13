import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F


#Word to Vec With Negative Sampling
class SGNS(torch.nn.Module):
    def __init__(self, D_dim, W_dim):
        super(SGNS, self).__init__()
        self.dict_len = D_dim
        #first embedding layer, for center word
        self.embedding1 = torch.nn.Embedding(D_dim, W_dim)
        #second embedding layer, for neighbor word
        self.embedding2 = torch.nn.Embedding(D_dim, W_dim)

    def forward(self, target_tensor, context_tensor, rand_tensor):
        projection = self.embedding1(target_tensor)
        projection2 = self.embedding2(context_tensor)
        emb_product = torch.mul(projection, projection2)
        emb_product = torch.sum(emb_product, dim=1)
        out = torch.sum(F.logsigmoid(emb_product))

        projection3 = self.embedding2(rand_tensor)
        emb_product2 = torch.mul(projection, projection3)
        emb_product2 = torch.sum(emb_product2, dim=1)
        out += torch.sum(F.logsigmoid(-emb_product2))
        return -out

