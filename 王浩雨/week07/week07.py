import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC  # 使用线性SVM的更稳定实现
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.exceptions import UndefinedMetricWarning
import warnings

# 忽略某些无预测样本相关的警告，保持输出整洁
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# 1. 加载原始数据集
df_raw = pd.read_csv("文本分类练习.csv")

# 2. 初始化 TF-IDF 向量化器
tfidf_vec = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    lowercase=False
)

# 3. 使用原始“review”列生成特征矩阵
X_features = tfidf_vec.fit_transform(df_raw["review"]).toarray()
y_labels = df_raw["label"].values

# 4. 划分训练集和测试集（70%/30%），保持标签分布一致
X_tr, X_te, y_tr, y_te = train_test_split(
    X_features, y_labels, test_size=0.3, random_state=42, stratify=y_labels
)

print(f"\n特征提取结果：")
print(f"训练集规模：{X_tr.shape[0]} 条样本，{X_tr.shape[1]} 个特征")
print(f"测试集规模：{X_te.shape[0]} 条样本，{X_te.shape[1]} 个特征")

def train_and_evaluate(model, model_label):
    """训练模型并返回评估指标"""
    # 训练
    model.fit(X_tr, y_tr)
    # 测试集预测
    preds = model.predict(X_te)
    # 计算四个核心指标
    metrics = {
        "模型名称": model_label,
        "准确率(Accuracy)": round(accuracy_score(y_te, preds), 4),
        "精确率(Precision)": round(precision_score(y_te, preds, zero_division=0), 4),
        "召回率(Recall)": round(recall_score(y_te, preds, zero_division=0), 4),
        "F1值": round(f1_score(y_te, preds, zero_division=0), 4)
    }
    return metrics, model

# 5. 初始化三种经典分类模型
models_with_labels = [
    (LogisticRegression(max_iter=1000, random_state=42), "逻辑回归"),
    (MultinomialNB(), "朴素贝叶斯"),
    (LinearSVC(), "线性SVM")  # 使用适用于文本分类的线性SVM
]

# 6. 训练所有模型并收集结果
results_list = []
best_model_ref = None
best_f1_score = 0

for clf, label in models_with_labels:
    print(f"\n正在训练 {label}...")
    stats, trained_clf = train_and_evaluate(clf, label)
    results_list.append(stats)
    # 更新最优模型（以 F1 为准）
    if stats["F1值"] > best_f1_score:
        best_f1_score = stats["F1值"]
        best_model_ref = trained_clf

# 7. 将结果整理为 DataFrame
results_df = pd.DataFrame(results_list)
print("\n" + "="*80)
print("无文本预处理：各模型性能对比表")
print("="*80)
print(results_df.to_string(index=False))
best_model_name = results_df.loc[results_df["F1值"].idxmax(), "模型名称"]
print(f"\n最优模型：{best_model_name}（F1值：{best_f1_score}）")

# 8. 使用最优模型对原始文本进行直接预测（无需额外预处理）
def predict_with_raw_text(input_text, model, vectorizer):
    """直接对原始文本进行预测，复用训练好的向量器"""
    X_input = vectorizer.transform([input_text]).toarray()
    label_pred = model.predict(X_input)[0]
    # 线性分类器的概率输出可能不同，若模型不支持 predict_proba，则改为概率为 1/0 的简单表示
    try:
        prob_dist = model.predict_proba(X_input)[0]
        confidence = float(prob_dist[label_pred])
    except Exception:
        # 对不支持概率估计的模型，给出一个占位的置信度
        confidence = float('nan')
    return {
        "原始输入评论": input_text,
        "预测结果": "好评" if label_pred == 1 else "差评",
        "置信度": round(confidence, 4) if not np.isnan(confidence) else None
    }

# 9. 测试若干条原始评论（请按需要修改）
raw_test_comments = [
    "送餐超级快！汤还是热的，味道比店里吃还香～",
    "等了1个半小时，饭全凉了！菜还少了一份，客服也不回复！",
    "包装很严实，没有撒漏，分量足够，下次还点！"
]

print("\n最优模型：原始文本直接预测示例")
raw_predictions = [predict_with_raw_text(text, best_model_ref, tfidf_vec) for text in raw_test_comments]
raw_pred_df = pd.DataFrame(raw_predictions)
print(raw_pred_df.to_string(index=False))
