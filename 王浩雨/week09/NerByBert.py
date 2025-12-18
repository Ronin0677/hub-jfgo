# -*- coding: utf-8 -*-
"""
BERT-based NER 使用 HuggingFace 进行训练、评估和简单预测。
数据：CoNLL-2003（或你自己的标注数据，标签应为 BIO 格式）
模型：bert-base-uncased（此处用 bert-base-uncased 作为示例，中文任务可改用 bert-base-chinese）
"""

import torch
from torch.utils.data import DataLoader
from transformers import BertForTokenClassification, BertTokenizerFast, AdamW, get_scheduler
from transformers import DataCollatorForTokenClassification
from datasets import load_dataset
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score, accuracy_score
from tqdm import tqdm

# 1) 配置与准备
MODEL_NAME = "bert-base-uncased"  # 如果是中文数据集，可以改成 "bert-base-chinese" 或者其他中文 BERT
LANG = "en"  # 数据语言，便于下游处理
NUM_EPOCHS = 3
BATCH_SIZE = 8
MAX_LENGTH = 128
LEARNING_RATE = 5e-5

# 标签映射（CoNLL-2003 常见标签）
LABELS = [
    "O",
    "B-PER", "I-PER",
    "B-ORG", "I-ORG",
    "B-LOC", "I-LOC",
    "B-MISC", "I-MISC"
]
label_to_id = {l: i for i, l in enumerate(LABELS)}
id_to_label = {i: l for l, i in label_to_id.items()}

# 2) 数据加载与处理
dataset = load_dataset("conll2003")

tokenizer = BertTokenizerFast.from_pretrained(MODEL_NAME)

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_idx in word_ids:
        if word_idx is None:
            new_labels.append(-100)
        elif word_idx != current_word:
            label = labels[word_idx]
            new_labels.append(label_to_id[label])
            current_word = word_idx
        else:
            new_labels.append(-100)
    return new_labels

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples["tokens"], is_split_into_words=True, truncation=True, max_length=MAX_LENGTH)
    labels = []
    for i, label_seq in enumerate(examples["ner"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        label_ids = align_labels_with_tokens(label_seq, word_ids)
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

train_dataset = dataset["train"].map(tokenize_and_align_labels, batched=True, remove_columns=dataset["train"].column_names)
valid_dataset = dataset["validation"].map(tokenize_and_align_labels, batched=True, remove_columns=dataset["validation"].column_names)
test_dataset  = dataset["test"].map(tokenize_and_align_labels, batched=True, remove_columns=dataset["test"].column_names)

# 3) 模型
model = BertForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=id_to_label, label2id=label_to_id)

# 4) 数据整理器
data_collator = DataCollatorForTokenClassification(tokenizer)

# 5) 训练准备
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=BATCH_SIZE, collate_fn=data_collator)
valid_dataloader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

num_training_steps = NUM_EPOCHS * len(train_dataloader)
lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 6) 训练循环
for epoch in range(NUM_EPOCHS):
    model.train()
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")
    total_loss = 0.0
    for batch in progress_bar:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": total_loss / (progress_bar.n + 1)})
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} average training loss: {avg_train_loss:.4f}")

    # 验证
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in valid_dataloader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)

            label_ids = batch["labels"].to("cpu").numpy()
            true_labels = []
            pred_labels = []
            for i in range(len(label_ids)):
                for j in range(len(label_ids[i])):
                    if label_ids[i][j] != -100:
                        true_labels.append(label_ids[i][j])
                        pred_labels.append(int(preds[i][j].cpu()))
            all_true.extend(true_labels)
            all_pred.extend(pred_labels)

    f1 = f1_score([id_to_label[i] for i in all_true], [id_to_label[i] for i in all_pred])
    acc = accuracy_score([id_to_label[i] for i in all_true], [id_to_label[i] for i in all_pred])
    print(f"Validation -> Acc: {acc:.4f}  F1: {f1:.4f}")

# 7) 测试集评估（可选）
# test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)
# model.eval()
# # ... 评估逻辑与验证类似 ...

# 8) 简单预测示例
def predict(text: str):
    model.eval()
    inputs = tokenizer(text.split(), is_split_into_words=True, return_tensors="pt", truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    preds = torch.argmax(outputs.logits, dim=-1)[0].cpu().tolist()
    
    # 获取原始 tokens (去除特殊 token)
    # 注意：这里的 tokens 对应的是 tokenizer 切分后的 sub-word tokens
    # 如果需要与原始句子词语一对一对应，需要更复杂的对齐逻辑
    # For simplicity, we output the predicted label for each sub-word token.
    # A more sophisticated approach would aggregate sub-word predictions back to word level.
    
    word_ids = inputs.word_ids(batch_index=0)
    original_tokens = text.split()
    aligned_preds = []
    current_word_idx = 0
    
    for i, word_id in enumerate(word_ids):
        if word_id is None: # Special tokens like CLS, SEP
            continue
        if word_id == current_word_idx: # Subsequent sub-word tokens of the same word
            if current_word_idx < len(original_tokens):
                aligned_preds.append((original_tokens[current_word_idx], id_to_label[preds[i]]))
            else: # Should not happen if MAX_LENGTH is sufficient
                aligned_preds.append(("[UNK]", id_to_label[preds[i]]))
        else: # Start of a new word
            current_word_idx = word_id
            if current_word_idx < len(original_tokens):
                aligned_preds.append((original_tokens[current_word_idx], id_to_label[preds[i]]))
            else: # Should not happen if MAX_LENGTH is sufficient
                 aligned_preds.append(("[UNK]", id_to_label[preds[i]]))
    
    # Deduplicate for cases where multiple sub-words for a single word get the same token/label
    # This simplified deduplication might not be perfect if different subwords get different labels
    # A more robust method is needed if precise word-level prediction is critical.
    final_preds = []
    seen_tokens = set()
    for token, label in aligned_preds:
        if token not in seen_tokens:
            final_preds.append((token, label))
            seen_tokens.add(token)
    
    return final_preds

# 使用一个简单示例
example_text = "George Washington founded the United States."
preds = predict(example_text)
print("Prediction (word, label):")
for tok, lab in preds:
    print(f"{tok:>12} -> {lab}")

example_text_2 = "Apple is looking at buying U.K. startup for $1 billion."
preds_2 = predict(example_text_2)
print("\nPrediction (word, label):")
for tok, lab in preds_2:
    print(f"{tok:>12} -> {lab}")
