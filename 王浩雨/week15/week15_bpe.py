import re
import collections
from typing import List, Dict, Tuple, Set
import json

class BPETokenizer:
    def __init__(self, vocab_size: int = 10000, special_tokens: List[str] = None):
        """
        初始化BPE分词器
        
        Args:
            vocab_size: 目标词汇表大小
            special_tokens: 特殊token列表，如['<pad>', '<unk>', '<s>', '</s>']
        """
        self.vocab_size = vocab_size
        self.vocab = {}
        self.merges = {}  # 存储合并规则
        self.special_tokens = special_tokens or []
        self.special_token_ids = {}
        
    def preprocess_text(self, text: str) -> str:
        """预处理文本"""
        # 转换为小写（可选）
        text = text.lower()
        # 添加单词边界标记
        text = ' '.join(text.split())
        # 在单词前添加特殊符号表示单词开始
        text = re.sub(r'(\w)', r' \1', text)
        # 合并多个空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def get_stats(self, vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
        """统计相邻符号对的频率"""
        pairs = collections.defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += freq
        return pairs
    
    def merge_vocab(self, pair: Tuple[str, str], vocab: Dict[str, int]) -> Dict[str, int]:
        """合并指定的符号对"""
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        new_vocab = {}
        pattern = re.compile(r'(?<!\S)' + re.escape(bigram) + r'(?!\S)')
        
        for word, freq in vocab.items():
            new_word = pattern.sub(replacement, word)
            new_vocab[new_word] = freq
        return new_vocab
    
    def train(self, corpus: List[str], min_freq: int = 2):
        """
        训练BPE分词器
        
        Args:
            corpus: 训练语料列表
            min_freq: 最小词频阈值
        """
        # 1. 预处理语料并统计初始词汇
        word_freq = collections.defaultdict(int)
        for text in corpus:
            processed = self.preprocess_text(text)
            words = processed.split()
            for word in words:
                # 初始时每个字符用空格分隔
                tokenized_word = ' '.join(list(word))
                word_freq[tokenized_word] += 1
        
        # 2. 过滤低频词
        word_freq = {word: freq for word, freq in word_freq.items() 
                    if freq >= min_freq}
        
        # 3. BPE迭代合并
        vocab = word_freq.copy()
        merges = {}
        num_merges = self.vocab_size - len(self.special_tokens) - 256  # 256个基础字符
        
        for i in range(num_merges):
            # 统计当前频率
            stats = self.get_stats(vocab)
            if not stats:
                break
                
            # 找到频率最高的符号对
            best_pair = max(stats, key=stats.get)
            best_freq = stats[best_pair]
            
            if best_freq < min_freq:
                break
                
            # 合并符号对
            vocab = self.merge_vocab(best_pair, vocab)
            
            # 记录合并规则
            merges[best_pair] = i + 256  # 从256开始分配ID
            
            # 打印进度
            if (i + 1) % 100 == 0:
                print(f"Merged {i + 1} pairs, vocab size: {len(vocab) + 256}")
        
        # 4. 构建最终词汇表
        self.merges = merges
        
        # 基础ASCII字符 (0-255)
        self.vocab = {chr(i): i for i in range(256)}
        
        # 特殊token
        for idx, token in enumerate(self.special_tokens):
            self.vocab[token] = 256 + len(merges) + idx
            self.special_token_ids[token] = 256 + len(merges) + idx
        
        # 合并得到的token
        for (a, b), idx in merges.items():
            self.vocab[a + b] = idx
    
    def tokenize(self, text: str) -> List[str]:
        """将文本分词为token"""
        # 预处理
        processed = self.preprocess_text(text)
        words = processed.split()
        
        tokens = []
        for word in words:
            # 初始化为字符列表
            word_tokens = list(word)
            
            # 应用BPE合并规则
            while len(word_tokens) > 1:
                # 找到可以合并的pair
                pairs = [(word_tokens[i], word_tokens[i + 1]) 
                        for i in range(len(word_tokens) - 1)]
                
                # 查找合并规则
                merge_candidates = []
                for i, pair in enumerate(pairs):
                    if pair in self.merges:
                        merge_candidates.append((self.merges[pair], i, pair))
                
                if not merge_candidates:
                    break
                
                # 选择优先级最高的合并（ID最小的）
                _, idx, pair_to_merge = min(merge_candidates)
                
                # 执行合并
                word_tokens[idx] = pair_to_merge[0] + pair_to_merge[1]
                word_tokens.pop(idx + 1)
            
            tokens.extend(word_tokens)
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs"""
        tokens = self.tokenize(text)
        return [self.vocab.get(token, self.special_token_ids.get('<unk>', 0)) 
                for token in tokens]
    
    def decode(self, token_ids: List[int]) -> str:
        """将token IDs解码为文本"""
        # 创建反向映射
        id_to_token = {v: k for k, v in self.vocab.items()}
        
        tokens = []
        for token_id in token_ids:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                # 处理特殊token
                if token in self.special_tokens:
                    continue
                tokens.append(token)
            else:
                tokens.append('?')  # 未知token
        
        # 重建文本
        text = ''.join(tokens)
        # 移除单词边界标记
        text = text.replace(' ', '')
        return text
    
    def save(self, filepath: str):
        """保存分词器"""
        data = {
            'vocab_size': self.vocab_size,
            'vocab': self.vocab,
            'merges': {f"{k[0]} {k[1]}": v for k, v in self.merges.items()},
            'special_tokens': self.special_tokens,
            'special_token_ids': self.special_token_ids
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, filepath: str):
        """加载分词器"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab_size = data['vocab_size']
        self.vocab = data['vocab']
        self.merges = {tuple(k.split()): v for k, v in data['merges'].items()}
        self.special_tokens = data['special_tokens']
        self.special_token_ids = data['special_token_ids']


# ============ 使用示例 ============

def example_usage():
    # 1. 准备训练语料
    corpus = [
        "自然语言处理是人工智能的重要领域。",
        "深度学习在NLP中取得了巨大成功。",
        "BERT和GPT是当前最流行的预训练模型。",
        "分词是中文处理的基础步骤。",
        "BPE算法可以有效地处理未登录词。",
        "机器学习让计算机从数据中学习规律。",
        "神经网络模仿人脑的工作方式。",
        "注意力机制在Transformer中非常关键。",
        "预训练模型需要大量的计算资源。",
        "微调可以使模型适应特定任务。"
    ]
    
    # 2. 创建并训练分词器
    print("训练BPE分词器...")
    tokenizer = BPETokenizer(
        vocab_size=500,  # 较小的词汇表用于演示
        special_tokens=['<pad>', '<unk>', '<s>', '</s>']
    )
    
    tokenizer.train(corpus, min_freq=1)
    print(f"词汇表大小: {len(tokenizer.vocab)}")
    
    # 3. 测试分词
    test_text = "自然语言处理中的BERT模型"
    print(f"\n测试文本: {test_text}")
    
    tokens = tokenizer.tokenize(test_text)
    print(f"分词结果: {tokens}")
    
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    
    decoded = tokenizer.decode(token_ids)
    print(f"解码结果: {decoded}")
    
    # 4. 保存和加载模型
    tokenizer.save("bpe_tokenizer.json")
    
    # 创建新实例并加载
    new_tokenizer = BPETokenizer()
    new_tokenizer.load("bpe_tokenizer.json")
    
    # 5. 查看词汇表示例
    print("\n词汇表示例（前20个）:")
    for i, (token, idx) in enumerate(list(tokenizer.vocab.items())[:20]):
        print(f"{idx:4d}: {repr(token)}")
    
    # 6. 处理更大规模语料的示例函数
    process_large_corpus_example()


def process_large_corpus_example():
    """处理大规模语料的示例"""
    import glob
    
    def train_on_files(file_pattern: str, vocab_size: int = 30000):
        """从文件训练BPE"""
        print(f"\n从文件训练BPE (pattern: {file_pattern})")
        
        # 读取所有文件
        corpus = []
        for filepath in glob.glob(file_pattern):
            with open(filepath, 'r', encoding='utf-8') as f:
                corpus.extend(f.readlines())
        
        print(f"读取了 {len(corpus)} 行文本")
        
        # 训练分词器
        tokenizer = BPETokenizer(
            vocab_size=vocab_size,
            special_tokens=['<pad>', '<unk>', '<s>', '</s>', '<mask>']
        )
        
        tokenizer.train(corpus[:10000], min_freq=2)  # 使用前10000行训练
        return tokenizer
    
    # 假设有文本文件
    # tokenizer = train_on_files("data/*.txt")


def advanced_bpe_example():
    """高级BPE使用示例"""
    from collections import Counter
    
    # 1. 带子词统计的BPE变体
    class BPETokenizerWithStats(BPETokenizer):
        def train(self, corpus: List[str], min_freq: int = 2):
            super().train(corpus, min_freq)
            # 额外统计子词频率
            self.subword_freq = Counter()
            for text in corpus:
                tokens = self.tokenize(text)
                self.subword_freq.update(tokens)
        
        def get_rare_subwords(self, threshold: int = 5) -> List[str]:
            """获取罕见子词"""
            return [sw for sw, freq in self.subword_freq.items() 
                   if freq <= threshold]
    
    # 2. 使用示例
    corpus = [
        "hello world",
        "hello everyone",
        "world peace",
        "peace and love"
    ]
    
    tokenizer = BPETokenizerWithStats(vocab_size=50)
    tokenizer.train(corpus)
    
    print("\n子词频率统计:")
    for token, freq in tokenizer.subword_freq.most_common(10):
        print(f"{token}: {freq}")


if __name__ == "__main__":
    print("=" * 50)
    print("BPE分词器完整示例")
    print("=" * 50)
    
    # 运行基础示例
    example_usage()
    
    # 运行高级示例
    # advanced_bpe_example()