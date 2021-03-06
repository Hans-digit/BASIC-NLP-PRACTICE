import pandas as pd
import re
from nltk import word_tokenize
from collections import Counter
import itertools
import random
from tqdm import tqdm
import torch
import numpy as np
from model import RNN_imdb
from torch import optim as optim

data = pd.read_csv('/home/james/data/IMDB Dataset.csv', dtype = object)
index_list = list(data.index)
review_list = [] # normalized_text 가 여기에서 review_list
for i in index_list:
    temp_data = data.loc[i,:]
    a = temp_data['review'].replace('<br />','') # br 제거
    a = re.sub(r'[^a-zA-Z ]', '', a.lower()) # 영어, 공백빼고 제거하기
    review_list.append(a)

data['review'] = review_list
result = [word_tokenize(sentence) for sentence in review_list]
word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 18}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)
word_to_index = {w: idx for idx, w in enumerate(vocabulary)}
index_to_word = {idx: w for idx, w in enumerate(vocabulary)}
data['sentiment'] = data['sentiment'].replace(['positive','negative'],[1,0])
dataset = data['review']
target = data['sentiment']

result_list = []
for sentence in list(dataset):
    numbering = []
    for s in word_tokenize(sentence):
        try:
            index = word_to_index[s]
            numbering.append(index)
        except:
            pass
    result_list.append(numbering)
dataset = result_list

# 문장의 최대 길이 : 2262
# 문장의 평균 길이 : 216.546620
# 문장 길이는 단어 400개로 제한
#%%
dataset = [i[:400] for i in dataset]
for _ in range(len(dataset)):
    if len(dataset[_]) < 400:
        dataset[_] += [vocabulary_len]*(400-len(dataset[_]))
    else:
        pass


context_tuple_list = []
for _ in range(len(dataset)):
    context_tuple_list.append((dataset[_], target[_]))

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
    return [batches[:middle],  batches[middle:]]

batches = get_batches(context_tuple_list, batch_size = 100)
train_batches, test_batches = train_test_split(batches, 0.8)

model = RNN_imdb(vocabulary_len, batch_size = 100, text_size = 400, input_size = 400, hidden_size = 200, num_layers = 3).to(device)
optimizer = torch.optim.Adam(model.parameters())
# scheduler = optim.lr_scheduler.LambdaLR(optimizer = optimizer, lr_lambda = lambda epoch:0.95 **epoch, last_epoch = -1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=3, eta_min=0.001)
criterion = torch.nn.CrossEntropyLoss()
best_accuracy = 0
for _ in range(20):
    for i in tqdm(range(len(batches)-10)):
        text, sentiment = batches[i]
        model.zero_grad()
        output = model(text)
        loss = criterion(output, sentiment)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            model.eval()
            ans = 0
            for _ in test_batches:
                test_result = model.forward(_[0])
                test_result = torch.argmax(test_result, dim=1)
                test_answer = _[1]
                ans += torch.sum(test_result == test_answer).item()
            accuracy = (ans / len(test_batches))
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                print(f'BEST ACCURACY : {best_accuracy}')
                if i != 0:
                    torch.save(model.state_dict(), './model/model_rnn.bin')
            model.train()
        else:
            pass
    scheduler.step()
    print(optimizer.param_groups[0]['lr'])

# torch.save(model.state_dict(), './model/model_rnn.bin')