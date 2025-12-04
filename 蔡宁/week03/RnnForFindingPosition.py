# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""
基于pytorch的基础RNN网络
实现判断文本中特定字符（你、我、他）的位置
"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab_size, hidden_dim=64):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, vector_dim, padding_idx=0)  # embedding层
        # 使用基础RNN
        self.rnn = nn.RNN(
            input_size=vector_dim,
            hidden_size=hidden_dim,
            batch_first=True,  # 第一个维度为batch_size
            bidirectional=False  # 单向RNN
        )
        # 输出层，每个位置预测是否为特定字符
        self.classify = nn.Linear(hidden_dim, 1)
        self.activation = torch.sigmoid  # sigmoid归一化函数，输出0-1之间的概率
        self.loss = nn.BCELoss()  # 二元交叉熵损失

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        # RNN输出: output为所有时间步的输出，hidden为最后一个时间步的隐藏状态
        rnn_output, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, hidden_dim)
        x = self.classify(rnn_output)  # (batch_size, sen_len, hidden_dim) -> (batch_size, sen_len, 1)
        y_pred = self.activation(x).squeeze(-1)  # (batch_size, sen_len)，每个位置输出0-1的概率

        if y is not None:
            # 计算每个位置的损失并平均
            return self.loss(y_pred, y)
        else:
            return y_pred


# 构建字符集和映射表
def build_vocab():
    chars = "你我他abcdefghijklmnopqrstuvwxyz"  # 字符集，包含特定字符"你我他"
    vocab = {"pad": 0}  # 填充字符
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)  # 未知字符
    return vocab


# 随机生成一个样本
# 生成包含sentence_length个字符的文本，并标记特定字符的位置
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，排除填充字符
    x = [random.choice([k for k, v in vocab.items() if v != 0]) for _ in range(sentence_length)]

    # 构建标签：每个位置如果是特定字符则为1，否则为0
    target_chars = {"你", "我", "他"}
    y = [1.0 if char in target_chars else 0.0 for char in x]

    # 将字转换成序号
    x = [vocab.get(word, vocab['unk']) for word in x]
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for _ in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length, hidden_dim=64):
    model = TorchModel(char_dim, sentence_length, len(vocab), hidden_dim)
    return model


# 测试模型准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    sample_num = 200
    x, y = build_dataset(sample_num, vocab, sentence_length)
    print(f"测试样本中特定字符总出现次数: {int(torch.sum(y))}")

    correct = 0
    total = 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，(batch_size, sen_len)

        # 对每个位置进行判断
        for batch in range(y_pred.shape[0]):
            for pos in range(y_pred.shape[1]):
                pred = 1 if y_pred[batch, pos] >= 0.5 else 0
                true = int(y[batch, pos])
                if pred == true:
                    correct += 1
                total += 1

    accuracy = correct / total
    print(f"位置预测准确率: {accuracy:.4f}, 正确预测: {correct}/{total}")
    return accuracy


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 32  # 每次训练样本个数
    train_sample = 1000  # 每轮训练总样本数
    char_dim = 30  # 每个字的维度
    sentence_length = 10  # 句子长度
    hidden_dim = 64  # RNN隐藏层维度
    learning_rate = 0.001  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_dim)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算损失
            loss.backward()  # 反向传播
            optim.step()  # 更新权重
            watch_loss.append(loss.item())

        print(f"\n第{epoch + 1}轮，平均loss: {np.mean(watch_loss):.6f}")
        acc = evaluate(model, vocab, sentence_length)
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "basic_rnn_position_model.pth")
    # 保存词表
    with open("position_vocab.json", "w", encoding="utf8") as f:
        f.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 30
    sentence_length = 10
    hidden_dim = 64

    # 加载字符表
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length, hidden_dim)
    # 加载训练好的权重
    model.load_state_dict(torch.load(model_path))

    # 处理输入
    x = []
    processed_strings = []
    for input_string in input_strings:
        # 截断或填充到固定长度
        if len(input_string) > sentence_length:
            processed = input_string[:sentence_length]
        else:
            processed = input_string + "pad" * (sentence_length - len(input_string))
        processed_strings.append(processed)

        # 转换为序号
        x.append([vocab.get(char, vocab['unk']) for char in processed])

    model.eval()
    with torch.no_grad():
        result = model(torch.LongTensor(x))  # 模型预测

    # 输出结果
    target_chars = {"你", "我", "他"}
    for i, input_string in enumerate(processed_strings):
        print(f"\n输入: {input_strings[i]} (处理后: {input_string})")
        print("位置预测结果:")
        pred_probs = result[i].numpy()

        # 显示每个字符及其预测结果
        for pos, (char, prob) in enumerate(zip(input_string, pred_probs)):
            if char == "pad":  # 跳过填充字符
                continue
            is_target = "是特定字符" if char in target_chars else "不是特定字符"
            pred = "预测: 是" if prob >= 0.5 else "预测: 否"
            print(f"位置 {pos}: 字符 '{char}' ({is_target})，概率: {prob:.4f}，{pred}")


if __name__ == "__main__":
    main()
    # 测试文本
    test_strings = ["你好世界abcdef", "a我c他xyz", "abcdefghij", "你我他在一起", "woshinide", "他是谁呢abc"]
    # predict("basic_rnn_position_model.pth", "position_vocab.json", test_strings)
