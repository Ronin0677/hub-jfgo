# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class SentenceEncoderModule(nn.Module):
    """
    该模块负责将输入的句子（词索引序列）编码成固定维度的向量表示。
    """
    def __init__(self, config):
        super(SentenceEncoderModule, self).__init__()
        embedding_dim = config["hidden_size"]  # 词嵌入的维度
        vocabulary_size = config["vocab_size"] + 1  # 词汇表大小，加1是因为0通常作为padding_idx
        # self.max_seq_length = config["max_length"] # 最大序列长度，当前未直接使用，但可能用于其他模型
        
        # 词嵌入层
        self.token_embedding = nn.Embedding(vocabulary_size, embedding_dim, padding_idx=0)
        # 一个简单的线性层，用于转换嵌入的维度，或者用于进一步的特征提取
        self.linear_transform = nn.Linear(embedding_dim, embedding_dim)
        # Dropout层，用于正则化，防止过拟合
        self.dropout_layer = nn.Dropout(0.5)

    def forward(self, token_ids):
        """
        前向传播，将词索引序列转换为向量。
        Args:
            token_ids (torch.Tensor): 输入的词索引张量，形状为 (batch_size, sequence_length)。
        Returns:
            torch.Tensor: 编码后的句子向量，形状为 (batch_size, hidden_size)。
        """
        # 1. 词嵌入
        embedded_tokens = self.token_embedding(token_ids)  # (batch_size, sequence_length, hidden_size)

        # 2. 线性变换
        transformed_embeddings = self.linear_transform(embedded_tokens)  # (batch_size, sequence_length, hidden_size)

        # 3. 应用Dropout
        dropped_embeddings = self.dropout_layer(transformed_embeddings)

        # 4. 池化操作：这里使用max_pool1d来获取每个句子的全局表示
        # 需要将维度调整为 (batch_size, hidden_size, sequence_length)
        # 然后在 sequence_length 维度上进行最大池化
        # 最后 squeeze() 掉多余的维度（池化后会变成 (batch_size, hidden_size, 1)）
        pooled_vector = nn.functional.max_pool1d(
            dropped_embeddings.transpose(1, 2),  # 调整维度: (batch_size, hidden_size, sequence_length)
            kernel_size=dropped_embeddings.shape[1]  # 池化窗口大小等于序列长度
        ).squeeze()  # 移除最后一个维度 (batch_size, hidden_size)

        return pooled_vector


class SiameseNetworkModel(nn.Module):
    """
    西雅姆网络模型，用于学习句子相似度。
    它包含一个SentenceEncoderModule来编码句子，并根据训练或预测模式计算损失或相似度。
    """
    def __init__(self, config):
        super(SiameseNetworkModel, self).__init__()
        # 初始化句子编码器模块
        self.sentence_encoder_component = SentenceEncoderModule(config)
        # 设置默认的margin值，用于三元组损失
        self.triplet_margin_value = config.get("triplet_margin", 0.5)

    def calculate_cosine_distance(self, tensor_a, tensor_b):
        """
        计算两个向量之间的余弦距离。
        余弦距离 = 1 - 余弦相似度。
        Args:
            tensor_a (torch.Tensor): 第一个向量张量。
            tensor_b (torch.Tensor): 第二个向量张量。
        Returns:
            torch.Tensor: 余弦距离张量。
        """
        # L2归一化，确保向量长度为1，方便计算余弦相似度
        normalized_a = torch.nn.functional.normalize(tensor_a, dim=-1)
        normalized_b = torch.nn.functional.normalize(tensor_b, dim=-1)
        
        # 计算余弦相似度，沿最后一个维度（特征维度）求点积
        cosine_similarity_score = torch.sum(torch.mul(normalized_a, normalized_b), dim=-1)
        
        # 返回余弦距离
        return 1 - cosine_similarity_score

    def compute_cosine_triplet_loss(self, anchor_vec, positive_vec, negative_vec, margin=None):
        """
        计算基于余弦距离的三元组损失。
        损失函数旨在使anchor和positive之间的距离小于anchor和negative之间的距离，
        并且两者之差大于margin。
        
        Args:
            anchor_vec (torch.Tensor): Anchor句子的编码向量。
            positive_vec (torch.Tensor): Positive句子的编码向量。
            negative_vec (torch.Tensor): Negative句子的编码向量。
            margin (float, optional): 三元组损失的margin值。如果为None，则使用实例的默认margin。
        Returns:
            torch.Tensor: 计算得到的平均三元组损失。
        """
        if margin is None:
            margin = self.triplet_margin_value

        # 计算anchor与positive之间的距离
        pos_distance = self.calculate_cosine_distance(anchor_vec, positive_vec)
        # 计算anchor与negative之间的距离
        neg_distance = self.calculate_cosine_distance(anchor_vec, negative_vec)

        # 计算三元组损失：max(0, pos_distance - neg_distance + margin)
        # relu(x) = max(0, x)
        losses_per_triplet = torch.relu(pos_distance - neg_distance + margin)
        
        # 返回所有三元组的平均损失
        return losses_per_triplet.mean()

    def forward(self, sentence_batch1, sentence_batch2=None, sentence_batch3=None, operational_mode="train", loss_margin=None):
        """
        模型的前向传播函数。根据 operational_mode 执行不同的操作。
        
        Args:
            sentence_batch1 (torch.Tensor): 第一个句子批次 (anchor in train, first sentence in predict)。
            sentence_batch2 (torch.Tensor, optional): 第二个句子批次 (positive in train, second sentence in predict)。
            sentence_batch3 (torch.Tensor, optional): 第三个句子批次 (negative in train)。
            operational_mode (str): 模式，"train" (训练) 或 "predict" (预测)。
            loss_margin (float, optional): 训练模式下，用于三元组损失的margin值。
            
        Returns:
            torch.Tensor:
                - 训练模式下：计算得到的损失值。
                - 预测模式下：计算得到的句子相似度分数。
        Raises:
            ValueError: 当输入参数不满足当前operational_mode的要求时。
        """
        if operational_mode == "train":
            # 训练模式：需要三个句子，计算三元组损失
            if sentence_batch3 is None:
                raise ValueError("训练模式需要提供三个句子批次：anchor, positive, negative")
            
            # 编码所有三个句子批次
            anchor_encoded_vecs = self.sentence_encoder_component(sentence_batch1)  # anchor
            positive_encoded_vecs = self.sentence_encoder_component(sentence_batch2)  # positive
            negative_encoded_vecs = self.sentence_encoder_component(sentence_batch3)  # negative
            
            # 计算并返回三元组损失
            computed_loss = self.compute_cosine_triplet_loss(
                anchor_encoded_vecs, 
                positive_encoded_vecs, 
                negative_encoded_vecs, 
                margin=loss_margin
            )
            return computed_loss
            
        elif operational_mode == "predict":
            # 预测模式：需要两个句子，返回它们之间的相似度
            if sentence_batch2 is None:
                raise ValueError("预测模式需要提供两个句子批次")
            
            # 编码两个句子批次
            encoded_vec1 = self.sentence_encoder_component(sentence_batch1)
            encoded_vec2 = self.sentence_encoder_component(sentence_batch2)
            
            # 计算并返回余弦相似度 (1 - 距离)
            # 这里需要重新计算余弦相似度，而不是直接返回距离
            normalized_vec1 = torch.nn.functional.normalize(encoded_vec1, dim=-1)
            normalized_vec2 = torch.nn.functional.normalize(encoded_vec2, dim=-1)
            similarity_score = torch.sum(torch.mul(normalized_vec1, normalized_vec2), dim=-1)
            return similarity_score
        else:
            raise ValueError("operational_mode 必须是 'train' 或 'predict'")

    def get_sentence_embedding(self, sentence_token_ids):
        """
        编码单个句子，返回其向量表示。
        Args:
            sentence_token_ids (torch.Tensor): 输入句子的词索引张量。
        Returns:
            torch.Tensor: 编码后的句子向量。
        """
        return self.sentence_encoder_component(sentence_token_ids)


def select_optimizer(config_params, model_instance):
    """
    根据配置参数选择并实例化优化器。
    Args:
        config_params (dict): 包含优化器类型、学习率等参数的配置字典。
        model_instance (nn.Module): 需要优化的模型实例。
    Returns:
        torch.optim.Optimizer: 选定的优化器实例。
    """
    optimizer_type = config_params["optimizer"]
    learning_rate_value = config_params["learning_rate"]
    
    if optimizer_type == "adam":
        return Adam(model_instance.parameters(), lr=learning_rate_value)
    elif optimizer_type == "sgd":
        return SGD(model_instance.parameters(), lr=learning_rate_value)
    else:
        raise ValueError(f"不支持的优化器类型: {optimizer_type}")

# 预测函数示例
def predict_similarity_score(model_obj, first_sentence_tokens, second_sentence_tokens, device="cpu"):
    """
    预测两个句子之间的相似度分数。
    Args:
        model_obj (SiameseNetworkModel): 已训练的西雅姆网络模型。
        first_sentence_tokens (torch.Tensor): 第一个句子的词索引张量。
        second_sentence_tokens (torch.Tensor): 第二个句子的词索引张量。
        device (str): 运行模型的设备 ('cpu' 或 'cuda')。
    Returns:
        torch.Tensor: 预测的相似度分数。
    """
    # 将模型设置为评估模式，禁用Dropout等训练特有功能
    model_obj.eval()
    
    # 在no_grad()上下文中进行计算，不跟踪梯度，节省内存和计算
    with torch.no_grad():
        # 确保输入是批量的形式。如果输入是单一样本，需要添加一个批次维度。
        if len(first_sentence_tokens.shape) == 1:
            first_sentence_tokens = first_sentence_tokens.unsqueeze(0)
        if len(second_sentence_tokens.shape) == 1:
            second_sentence_tokens = second_sentence_tokens.unsqueeze(0)
            
        # 将输入数据移动到指定的设备
        sentence1_on_device, sentence2_on_device = first_sentence_tokens.to(device), second_sentence_tokens.to(device)
        
        # 调用模型进行预测
        similarity_result = model_obj(sentence1_on_device, sentence2_on_device, operational_mode="predict")
        
        return similarity_result


# 测试代码
if __name__ == "__main__":
    # 假设存在一个名为 Config 的字典，用于存储模型配置
    # 示例 Config 字典 (需要根据实际情况定义)
    class ConfigDict(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.__dict__ = self

    Config = ConfigDict({
        "hidden_size": 100,
        "vocab_size": 10,
        "max_length": 4,
        "optimizer": "adam",
        "learning_rate": 0.001,
        "triplet_margin": 0.2  # 示例margin值
    })
    
    # 实例化西雅姆网络模型
    siamese_model = SiameseNetworkModel(Config)
    
    # 准备示例输入数据 (词索引)
    # s1_sample: batch_size=2, sequence_length=4
    s1_sample = torch.LongTensor([[1, 2, 3, 0], [2, 2, 0, 0]]) 
    # s2_sample: batch_size=2, sequence_length=4 (positive sample for s1_sample[0])
    s2_sample = torch.LongTensor([[1, 2, 3, 4], [3, 2, 3, 4]]) 
    # s3_sample: batch_size=2, sequence_length=4 (negative sample for s1_sample[0])
    # 为了演示，我们简单地创建了一个与s2_sample不同的张量
    s3_sample = torch.LongTensor([[5, 6, 7, 0], [1, 1, 0, 0]]) 
    
    # 示例：在训练模式下计算损失
    print("--- 训练模式示例 ---")
    # 假设有标签 l (此处未用于损失计算，仅为示例)
    # l_sample = torch.LongTensor([[1], [0]]) # 示例标签
    
    # 传递句子批次 s1_sample (anchor), s2_sample (positive), s3_sample (negative)
    # 并指定operational_mode="train"
    training_loss = siamese_model(s1_sample, s2_sample, s3_sample, operational_mode="train", loss_margin=0.3)
    print(f"计算得到的训练损失: {training_loss.item()}")
    
    # 示例：在预测模式下计算相似度
    print("\n--- 预测模式示例 ---")
    # 传递句子批次 s1_sample (第一个句子), s2_sample (第二个句子)
    # 并指定operational_mode="predict"
    prediction_similarity = siamese_model(s1_sample, s2_sample, operational_mode="predict")
    print(f"计算得到的相似度分数 (batch size = {s1_sample.shape[0]}): {prediction_similarity}")

    # 示例：使用预测函数
    print("\n--- 使用预测函数示例 ---")
    # 假设有设备信息
    device = "cpu" 
    # 准备两个单一句子
    single_sentence1 = torch.LongTensor([1, 2, 3])
    single_sentence2 = torch.LongTensor([1, 2, 3, 4])
    
    predicted_score = predict_similarity_score(siamese_model, single_sentence1, single_sentence2, device=device)
    print(f"使用predict_similarity_score函数预测的相似度: {predicted_score}")

    # 示例：获取单个句子的编码向量
    print("\n--- 获取句子编码向量示例 ---")
    sentence_to_encode = torch.LongTensor([[1, 2, 3, 4]])
    encoded_vector = siamese_model.get_sentence_embedding(sentence_to_encode)
    print(f"句子 {sentence_to_encode} 的编码向量形状: {encoded_vector.shape}")
    print(f"编码向量: {encoded_vector}")
