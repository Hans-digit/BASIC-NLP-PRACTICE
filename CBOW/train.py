import torch
import pandas as pd
from model import CBOW
from tqdm import tqdm
import sentencepiece as spm
from random import randint
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from lxml import etree
from nltk.tokenize import word_tokenize, sent_tokenize
import re
import nltk
from tqdm import tqdm
import itertools
from collections import Counter
import random
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.nn.functional as F

device = ('cuda' if torch.cuda.is_available() else 'cpu')
targetXML = open('ted_en-20160408.xml', 'r', encoding='UTF8')
target_text = etree.parse(targetXML)

# xml 파일로부터 <content>와 </content> 사이의 내용만 가져온다.
parse_text = '\n'.join(target_text.xpath('//content/text()'))

# 정규 표현식의 sub 모듈을 통해 content 중간에 등장하는 (Audio), (Laughter) 등의 배경음 부분을 제거.
# 해당 코드는 괄호로 구성된 내용을 제거.
content_text = re.sub(r'\([^)]*\)', '', parse_text)

# 입력 코퍼스에 대해서 NLTK를 이용하여 문장 토큰화를 수행.
sent_text = sent_tokenize(content_text)

# 각 문장에 대해서 구두점을 제거하고, 대문자를 소문자로 변환.
normalized_text = []
for string in sent_text:
    tokens = re.sub(r"[^a-z0-9]+", " ", string.lower())
    normalized_text.append(tokens)

# 각 문장에 대해서 NLTK를 이용하여 단어 토큰화를 수행.

result = [word_tokenize(sentence) for sentence in normalized_text]
word_count = dict(Counter(list(itertools.chain.from_iterable(result))))
word_count = {key: word_count[key] for key in word_count if word_count[key] >= 5}
vocabulary = list(word_count.keys())
vocabulary_len = len(vocabulary)

word_to_index = {w: idx for idx, w in enumerate(vocabulary)}
index_to_word = {idx: w for idx, w in enumerate(vocabulary)}

for i in range(len(result)):
    result[i] = [word_to_index[j] for j in result[i] if j in vocabulary]

w = 2
context_tuple_list = []
for text in tqdm(result):
    for i, word in enumerate(text):
        first_context_word_index = max(0, i - w)
        last_context_word_index = min(i + w + 1, len(text))
        temp_list = []
        # target_list = [0]*vocabulary_len
        # target_list[word] = 1
        for j in range(first_context_word_index, last_context_word_index):
            if i != j:
                temp_list.append(text[j])

        if len(temp_list) < 4:
            temp_list = temp_list + [vocabulary_len] * (4 - len(temp_list))
        context_tuple_list.append((word, temp_list))


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


batches = get_batches(context_tuple_list, batch_size=200)
model = CBOW(vocabulary_len, 100).to(device)
optimizer = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()

# for each batch, get loss and learn
for _ in range(20):
    for i in tqdm(range(len(batches))):
        target_word, context_tensor = batches[i]
        model.zero_grad()
        output = model(context_tensor)
        loss = criterion(output, target_word)
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print('==========man and woman=============')
            print(torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['man']]).to(device)),
                                          model.Embedding(torch.LongTensor([word_to_index['woman']]).to(device))))
            print('==========cat and dog =============')
            print(torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['cat']]).to(device)),
                                          model.Embedding(torch.LongTensor([word_to_index['dog']]).to(device))))

            print('===========paper and rock===========')
            print(torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['paper']]).to(device)),
                                          model.Embedding(torch.LongTensor([word_to_index['rock']]).to(device))))

            print('=============death and fly===========')
            print(torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['death']]).to(device)),
                                          model.Embedding(torch.LongTensor([word_to_index['fly']]).to(device))))

            print('==========bee and cat ============')
            print(torch.cosine_similarity(model.Embedding(torch.LongTensor([word_to_index['bee']]).to(device)),
                                          model.Embedding(torch.LongTensor([word_to_index['cat']]).to(device))))


torch.save(model.state_dict(), './model/model_cbow.bin')