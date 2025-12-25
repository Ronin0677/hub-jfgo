#coding:utf8

import torch
import torch.nn as nn
import numpy as np
import math
import random
import os
import re
from transformers import BertModel, BertTokenizer, BertConfig

BERT_PATH = r'D:\Test\bert-base-chinese'
BERT_TOKENIZER = BertTokenizer.from_pretrained(BERT_PATH)

class LanguageModel(nn.Module):
    def __init__(self):
        super(LanguageModel, self).__init__()
        bert_config = BertConfig.from_pretrained(BERT_PATH)
        bert_config.num_hidden_layers = 2
        bert_config.return_dict = False
        bert_config.ignore_mismatched_sizes = True

        self.layer = BertModel.from_pretrained(BERT_PATH, config=bert_config)
        self.layer.config.num_hidden_layers = 2
        self.layer.config.ignore_mismatched_sizes = True

        self.pad_token_id = BERT_TOKENIZER.pad_token_id
        vocab_size = self.layer.config.vocab_size
        hidden_size = self.layer.config.hidden_size

        self.classify = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.1)
        self.loss = nn.functional.cross_entropy

    def generate_attn_mask(self, x):
        if x.dim() == 1:
            batch_size, seq_len = 1, x.size()[0]
        else:
            batch_size, seq_len = x.size()[:2]
        custom_mask = torch.tril(torch.ones((seq_len, seq_len)), diagonal=0).float()
        custom_mask = custom_mask.unsqueeze(0).expand(batch_size, -1, -1)
        return custom_mask

    def forward(self, x, y=None):
        mask_customized = self.generate_attn_mask(x)
        x, _ = self.layer(x, attention_mask=mask_customized)
        y_pred = self.classify(x)
        if y is not None:
            return self.loss(y_pred.view(-1, y_pred.shape[-1]), y.view(-1))
        else:
            return torch.softmax(y_pred, dim=-1)

def load_corpus(path):
    corpus = ""
    with open(path, encoding="gbk") as f:
        for line in f:
            corpus += line.strip()
    return corpus

def build_sample(window_size, corpus):
    start = random.randint(0, len(corpus) - 1 - window_size)
    end = start + window_size
    window = corpus[start:end]
    target = corpus[start + 1:end + 1]
    x = [BERT_TOKENIZER.vocab.get(ch, BERT_TOKENIZER.vocab['[UNK]']) for ch in window]
    y = [BERT_TOKENIZER.vocab.get(ch, BERT_TOKENIZER.vocab['[UNK]']) for ch in target]
    return x, y

def build_dataset(sample_length, window_size, corpus):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(window_size, corpus)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)

def build_model():
    model = LanguageModel()
    return model

def generate_sentence(openings, model, window_size):
    model.eval()
    with torch.no_grad():
        pred_char = ""
        while pred_char != "[SEP]" and len(openings) <= 30:
            openings += pred_char
            x = [BERT_TOKENIZER.vocab.get(char, BERT_TOKENIZER.vocab['[UNK]']) for char in openings[-window_size:]]
            x = [BERT_TOKENIZER.cls_token_id] + x
            x = torch.LongTensor([x])
            if torch.cuda.is_available():
                x = x.cuda()
            y = model(x)
            y = y.squeeze()[-2]
            index = sampling_strategy(y)
            pred_char = BERT_TOKENIZER.convert_ids_to_tokens(int(index))
    return openings

def sampling_strategy(prob_distribution):
    if random.random() > 0.1:
        strategy = "greedy"
    else:
        strategy = "sampling"

    if strategy == "greedy":
        return int(torch.argmax(prob_distribution))
    elif strategy == "sampling":
        prob_distribution = prob_distribution.cpu().numpy()
        return np.random.choice(list(range(len(prob_distribution))), p=prob_distribution)

def calc_perplexity(sentence, model, vocab, window_size):
    prob = 0
    model.eval()
    with torch.no_grad():
        for i in range(1, len(sentence)):
            start = max(0, i - window_size)
            window = sentence[start:i]
            x = [vocab.get(char, vocab["<UNK>"]) for char in window]
            x = torch.LongTensor([x])
            target = sentence[i]
            target_index = vocab.get(target, vocab["<UNK>"])
            if torch.cuda.is_available():
                x = x.cuda()
            pred_prob_distribute = model(x)[0][-1]
            target_prob = pred_prob_distribute[target_index]
            prob += math.log(target_prob, 10)
    return 2 ** (prob * ( -1 / len(sentence)))

def train(corpus_path, save_weight=True):
    epoch_num = 10
    batch_size = 32
    train_sample = 5000
    window_size = 10

    corpus = load_corpus(corpus_path)
    model = build_model()
    if torch.cuda.is_available():
        model = model.cuda()
    optim = torch.optim.Adam(model.parameters(), lr=0.01)
    print("文本词表模型加载完毕，开始训练")

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, window_size, corpus)
            if torch.cuda.is_available():
                x, y = x.cuda(), y.cuda()
            optim.zero_grad()
            loss = model(x, y.squeeze())
            loss.backward()
            optim.step()
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        print("生成测试:", generate_sentence("各位各位，我们要相信，瓦片也有翻身日", model, window_size))
 

    if not save_weight:
        return
    else:
        base_name = os.path.basename(corpus_path).replace(".txt", ".pth")
        model_path = os.path.join("model", base_name)
        os.makedirs("model", exist_ok=True)
        torch.save(model.state_dict(), model_path)
        print(f"模型权重已保存至: {model_path}")
        return

if __name__ == "__main__":
    train("corpus.txt", True)
