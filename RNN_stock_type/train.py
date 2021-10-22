import pandas as pd
import re
from nltk import word_tokenize
from collections import Counter
import itertools
import random
from tqdm import tqdm
import torch
import numpy as np
from torch import optim as optim
from model import RNN_type
import time

r = open('/home/james/data/PycharmProjects/BASIC-NLP-PRACTICE/RNN_stock_type/data/type_data.csv',mode = 'rt')
data_origin = r.readlines()
data = []
number_dic = {}
for _ in range(len(data_origin)):
    temp_data = data_origin[_].lower().replace('\n','')
    temp_data = temp_data.split('|')
    if temp_data[0] in number_dic.keys():
        if number_dic[temp_data[0]] > 1000:
            pass
        else:
            data.append(temp_data)
            number_dic[temp_data[0]] += 1
    else:
        number_dic[temp_data[0]] = 1
        data.append(temp_data)
print(number_dic)
corp_list = [i[1] for i in data]
type_list = [i[0] for i in data]
result = [word_tokenize(sentence) for sentence in corp_list]
#max sentence len = 34
#mean sentence len = 6.69
#define sentence len to 34

word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 15}
print(word_count)
print(len(word_count))
time.sleep(10)
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)
word_to_index = {w: idx for idx, w in enumerate(vocabulary)}
index_to_word = {idx: w for idx, w in enumerate(vocabulary)}

# for i in range(len(result)):
#     temp_list = []
#     for j in result[i]:
#         if j in vocabulary:
#             temp_list.append(j)
#     result[i] = temp_list

#max sentence len 29
#mean sentence len = 5.64
#define sentence len = 29
type_set = set(type_list)
type_to_idx = {w:idx for idx, w in enumerate(list(type_set))}
idx_to_type = {idx:w for idx, w in enumerate(list(type_set))}

result_list = []
for sentence in corp_list:
    numbering = []
    for s in word_tokenize(sentence):
        try:
            index = word_to_index[s]
            numbering.append(index)
        except:
            pass
    result_list.append(numbering)
dataset = result_list
# a = 0
# for _ in dataset:
#     if len(_) > a:
#         a = len(_)
# print(a)
# 문장의 최대 길이 : 28
# 문장 길이는 단어 28개로 제한

max_len = 28
dataset = [i[:max_len] for i in dataset]
for _ in range(len(dataset)):
    if len(dataset[_]) < max_len:
        dataset[_] += [vocabulary_len]*(max_len-len(dataset[_]))
    else:
        pass

context_tuple_list = []
for _ in range(len(dataset)):
    context_tuple_list.append((dataset[_], type_to_idx[type_list[_]]))
device = ('cuda' if torch.cuda.is_available() else 'cpu')

def get_batches(context_tuple_list, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context = [], []
    for i in tqdm(range(len(context_tuple_list))):
        batch_target.append(context_tuple_list[i][0])
        batch_context.append(context_tuple_list[i][1])
        if (i + 1) % batch_size == 0 or i == len(context_tuple_list) - 1:
            tensor_target = torch.from_numpy(np.array(batch_target)).long().to(device)
            tensor_context = torch.from_numpy(np.array(batch_context)).long().to(device)
            batches.append((tensor_target, tensor_context))
            batch_target, batch_context = [], []
    return batches

def train_test_split(batches, train_ratio):
    elements = len(batches)
    middle = int(elements * train_ratio)
    return [batches[:middle],  batches[middle:-1]]

batches = get_batches(context_tuple_list, batch_size = 100)
train_batches, test_batches = train_test_split(batches, 0.8)

model = RNN_type(vocabulary_len, batch_size = 100, text_size = max_len, input_size = max_len, hidden_size = 50, num_layers = 3).to(device)

optimizer = torch.optim.Adam(model.parameters())
# scheduler = optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch:0.95 **epoch, last_epoch = -1)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0.001)
criterion = torch.nn.CrossEntropyLoss()
best_accuracy = 0
for _ in range(30):
    for i in tqdm(range(len(train_batches))):
        # print(i)
        text, sentiment = train_batches[i]
        model.zero_grad()
        output = model(text)
        loss = criterion(output, sentiment)
        loss.backward()
        optimizer.step()
        if i % 10 == 0:
            if i == 0:
                pass
            else:
                model.eval()
                ans = 0
                number = 0
                for _ in test_batches:
                    test_result = model.forward(_[0])
                    test_result = torch.argmax(test_result, dim=1)
                    test_answer = _[1]
                    number += len(test_answer)
                    ans += torch.sum(test_result == test_answer).item()
                accuracy = (ans / number)
                print(accuracy)
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    print(f'BEST ACCURACY : {best_accuracy}')
                    if i != 0:
                        torch.save(model.state_dict(), './model/model_rnn.bin')
                model.train()
        else:
            pass
    # scheduler.step()
    print(optimizer.param_groups[0]['lr'])

# torch.save(model.state_dict(), './model/model_rnn.bin')