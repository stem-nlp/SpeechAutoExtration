import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup

import csv
import json
import numpy as np
import pandas as pd

# 超参数
EPOCHS = 10  # 训练的轮数
BATCH_SIZE = 8  # 批大小
MAX_LEN = 300  # 文本最大长度
LR = 1e-5  # 学习率
WARMUP_STEPS = 100  # 热身步骤
T_TOTAL = 1000  # 总步骤
weight_decay = 1e-2
hidden_dropout_prob = 0.3
num_labels = 2
# pytorch的dataset类 重写getitem,len方法
class SentimentDataset(Dataset):
    def __init__(self, filepath):
        self.dataset = pd.read_csv(filepath)

    def __getitem__(self, idx):
        text = self.dataset.loc[idx, "review"]
        label = self.dataset.loc[idx, "label"]
        sample = {"text": text, "label": label}
        return sample

    def __len__(self):
        return len(self.dataset)


# # 加载数据集
# def load_dataset(filepath, max_len):
#     # 根据max_len参数进行padding
#     tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#     features = []
#
#     df = pd.read_csv(filepath)
#     for val in df[['label', 'review']].values:
#         tokens = tokenizer.encode(val[1], max_length=max_len)
#         features.append((tokens, val[0]))
#     return features
def convert_text_to_ids(tokenizer, text, max_len=100):
    if isinstance(text, str):
        tokenized_text = tokenizer.encode_plus(text, max_length=max_len, add_special_tokens=True)
        input_ids = tokenized_text["input_ids"]
        token_type_ids = tokenized_text["token_type_ids"]
    elif isinstance(text, list):
        input_ids = []
        token_type_ids = []
        for t in text:
            tokenized_text = tokenizer.encode_plus(t, max_length=max_len, add_special_tokens=True)
            input_ids.append(tokenized_text["input_ids"])
            token_type_ids.append(tokenized_text["token_type_ids"])
    else:
        raise Exception("Input unknown")
    return input_ids, token_type_ids

def seq_padding(tokenizer, X):
    pad_id = tokenizer.convert_tokens_to_ids("[PAD]")
    if len(X) <= 1:
        return torch.tensor(X)
    max_len = max([len(x) for x in X])
    # padded = [x + [pad_id] * (max_len - len(x)) if len(x) < max_len else x for x in X]
    padded = [x + [pad_id] * (max_len - len(x)) for x in X]
    X = torch.Tensor(padded)
    return X


# 计算每个batch的准确率
def batch_accuracy(pre, label):
    pre = pre.argmax(dim=1)
    correct = torch.eq(pre, label).sum().float().item()
    accuracy = correct / float(len(label))

    return accuracy


def train(model, loader, tokenizer, optimizer, device):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for i, batch in enumerate(loader):
        label = batch["label"]
        text = batch["text"]
        input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
        input_ids = seq_padding(tokenizer, input_ids)
        token_type_ids = seq_padding(tokenizer, token_type_ids)
        # 标签形状为 (batch_size, 1)
        label = label.unsqueeze(1)
        # 需要 LongTensor
        input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
        # 梯度清零
        optimizer.zero_grad()
        # 迁移到GPU
        input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
        output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
        y_pred_prob = output[1]
        y_pred_label = y_pred_prob.argmax(dim=1)

        # 计算loss
        loss = output[0]

        # 计算acc
        acc = ((y_pred_label == label.view(-1)).sum()).item()
        # 反向传播
        loss.backward()
        optimizer.step()
        # epoch 中的 loss 和 acc 累加
        epoch_loss += loss.item()
        epoch_acc += acc
        if i % 200 == 0:
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(label)))

    return epoch_loss / len(loader), epoch_acc / len(loader.dataset.dataset)

def evaluate(model, loader, tokenizer, device):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    with torch.no_grad():
        for _, batch in enumerate(loader):
            label = batch["label"]
            text = batch["text"]
            input_ids, token_type_ids = convert_text_to_ids(tokenizer, text)
            input_ids = seq_padding(tokenizer, input_ids)
            token_type_ids = seq_padding(tokenizer, token_type_ids)
            label = label.unsqueeze(1)
            input_ids, token_type_ids, label = input_ids.long(), token_type_ids.long(), label.long()
            input_ids, token_type_ids, label = input_ids.to(device), token_type_ids.to(device), label.to(device)
            output = model(input_ids=input_ids, token_type_ids=token_type_ids, labels=label)
            y_pred_label = output[1].argmax(dim=1)
            loss = output[0]
            acc = ((y_pred_label == label.view(-1)).sum()).item()
            epoch_loss += loss.item()
            epoch_acc += acc

    return epoch_loss / len(loader), epoch_acc / len(loader.dataset.dataset)

if __name__ == "__main__":
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 生成数据集以及迭代器#
    print("加载数据")
    train_cus = SentimentDataset('../data/weibo_senti_100k.csv')
    test_cus = SentimentDataset('../data/weibo_senti_100k_test.csv')
    # test_cus = SentimentDataset('../data/weibo_senti_100k_test.csv')
    train_loader = DataLoader(dataset=train_cus, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(dataset=test_cus, batch_size=BATCH_SIZE, shuffle=False)

    # Bert模型以及相关配置
    config = BertConfig.from_pretrained('bert-base-chinese',
                                        num_labels=num_labels,
                                        hidden_dropout_prob=hidden_dropout_prob)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertForSequenceClassification.from_pretrained('bert-base-chinese', config=config)
    model.to(device)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR)

    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=T_TOTAL)


    # 训练
    print('开始训练...')
    for i in range(EPOCHS):
        print("Epoch:{}".format(i))
        train_loss, train_acc = train(model, train_loader, tokenizer, optimizer,  device)
        print("train loss: ", train_loss, "\t", "train acc:", train_acc)
        valid_loss, valid_acc = evaluate(model, test_loader, tokenizer,  device)
        print("valid loss: ", valid_loss, "\t", "valid acc:", valid_acc)

    torch.save(model.state_dict(), '../save/bert_cla.ckpt')
    
    print('保存训练完成的model...')

    # 测试
    # print('开始加载训练完成的model...')
    # model.load_state_dict(torch.load('../save/bert_cla.ckpt'))
    #