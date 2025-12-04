# coding:utf8
# 解决 OpenMP 库冲突问题
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

""" 
基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，最大的数字在哪维就属于哪一类（0-4类）
"""

class TorchModel(nn.Module):
    def __init__(self, input_size, num_classes=5):
        super(TorchModel, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)  # 输出5个类别的分数
        # 移除sigmoid激活函数，因为交叉熵损失需要原始分数
        self.loss = nn.CrossEntropyLoss()  # 使用交叉熵损失函数【1✔】【2✔】
    
    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)  # 输出5个类别的原始分数（logits）
        if y is not None:
            # 交叉熵损失需要原始分数和类别索引【3✔】
            return self.loss(x, y.squeeze().long())  # 将标签转换为长整型
        else:
            # 预测时使用softmax得到概率分布
            return torch.softmax(x, dim=1)

# 生成一个样本，五维随机向量最大的数字在哪维就属于哪一类
def build_sample():
    x = np.random.random(5)
    max_index = np.argmax(x)  # 找到最大值所在的维度索引（0-4）
    return x, max_index

# 随机生成一批样本
def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample()
        X.append(x)
        Y.append([y])  # 保持与原始代码兼容的格式
    return torch.FloatTensor(X), torch.FloatTensor(Y)

# 测试代码，用来测试每轮模型的准确率
def evaluate(model):
    model.eval()
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    
    # 统计每个类别的样本数量
    unique, counts = np.unique(y.numpy(), return_counts=True)
    print("本次预测集中各类别样本数量：")
    for cls, count in zip(unique, counts):
        print(f"类别{int(cls)}: {count}个")
    
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回概率分布
        predicted_classes = torch.argmax(y_pred, dim=1)  # 取概率最大的类别【3✔】
        
        for y_p, y_t in zip(predicted_classes, y):
            if int(y_p) == int(y_t):
                correct += 1
            else:
                wrong += 1
                
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)

def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    num_classes = 5  # 输出类别数
    learning_rate = 0.001  # 学习率
    
    # 建立模型
    model = TorchModel(input_size, num_classes)
    
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    
    # 创建训练集
    train_x, train_y = build_dataset(train_sample)
    
    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            
            watch_loss.append(loss.item())
        
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果
        log.append([acc, float(np.mean(watch_loss))])
    
    # 保存模型
    torch.save(model.state_dict(), "model.bin")
    
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    num_classes = 5
    model = TorchModel(input_size, num_classes)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())
    
    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
        
        for vec, res in zip(input_vec, result):
            predicted_class = torch.argmax(res).item()  # 获取预测类别
            probability = res[predicted_class].item()  # 获取最大概率值
            print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, predicted_class, probability))

if __name__ == "__main__":
    main()
    
    # 测试预测功能
    test_vec = [
        [0.9, 0.1, 0.2, 0.3, 0.4],  # 最大值在第0维，应该是类别0
        [0.1, 0.95, 0.2, 0.3, 0.4],  # 最大值在第1维，应该是类别1
        [0.1, 0.2, 0.98, 0.3, 0.4],  # 最大值在第2维，应该是类别2
        [0.1, 0.2, 0.3, 0.97, 0.4],  # 最大值在第3维，应该是类别3
        [0.1, 0.2, 0.3, 0.4, 0.96]   # 最大值在第4维，应该是类别4
    ]
    predict("model.bin", test_vec)