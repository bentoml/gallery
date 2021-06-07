import os
import time

import torch
from torchtext.datasets import AG_NEWS
from torchtext.data.utils import get_tokenizer
from collections import Counter
from torchtext.vocab import Vocab
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
EPOCHS = 10  # epoch
LR = 5  # learning rate
BATCH_SIZE = 64  # batch size for training
EMSIZE = 64


def get_tokenizer_vocab(dataset=AG_NEWS, tokenizer_fn='basic_english', root_data_dir='dataset'):
    print('Getting tokenizer and vocab...')
    tokenizer = get_tokenizer(tokenizer_fn)
    train_ = dataset(root=root_data_dir, split='train')
    counter = Counter()
    for (label, line) in train_:
        counter.update(tokenizer(line))
    vocab = Vocab(counter, min_freq=1)
    return tokenizer, vocab


def get_pipeline(tokenizer, vocab):
    print('Setup pipeline...')
    text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
    label_pipeline = lambda x: int(x) - 1
    return text_pipeline, label_pipeline


def get_train_valid_split(train_iter):
    train_dataset = list(train_iter)
    num_train = int(len(train_dataset) * 0.95)
    split_train_, split_valid_ = random_split(train_dataset, [num_train, len(train_dataset) - num_train])
    return split_train_, split_valid_


def get_model_params(vocab):
    print('Setup model params...')
    train_iter = AG_NEWS(root='../dataset', split='train')
    num_class = len(set([label for (label, text) in train_iter]))
    vocab_size = len(vocab)
    return vocab_size, EMSIZE, num_class


def collate_batch(batch):
    label_list, text_list, offsets = [], [], [0]
    for (_label, _text) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device)


def train(model, data_loader, optimizer, criterion, epoch):
    model.train()
    total_acc, total_count = 0, 0
    log_interval = 500

    for idx, (label, text, offsets) in enumerate(data_loader):
        optimizer.zero_grad()
        predicted = model(text, offsets=offsets)
        loss = criterion(predicted, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        total_acc += (predicted.argmax(1) == label).sum().item()
        total_count += label.size(0)
        if idx % log_interval == 0 and idx > 0:
            print(f'| epoch {epoch:3d} | {idx:5d}/{len(data_loader):5d} batches | accuracy {(total_acc / total_count):5.3f}')
            total_acc, total_count = 0, 0


def evaluate(model, data_loader, criterion):
    model.eval()
    total_acc, total_count = 0, 0

    with torch.no_grad():
        for idx, (label, text, offsets) in enumerate(data_loader):
            predited_label = model(text, offsets)
            loss = criterion(predited_label, label)
            total_acc += (predited_label.argmax(1) == label).sum().item()
            total_count += label.size(0)
    return total_acc / total_count