@ -0,0 +1,139 @@
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True):
        super(MultiHeadSelfAttention, self).__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        # 线性映射：Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # 输出线性映射
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        # 初始化，与常用实现相近
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.q_proj.bias is not None:
            nn.init.constant_(self.q_proj.bias, 0.)
        if self.k_proj.bias is not None:
            nn.init.constant_(self.k_proj.bias, 0.)
        if self.v_proj.bias is not None:
            nn.init.constant_(self.v_proj.bias, 0.)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, x, attn_mask=None):
        """
        x: (batch_size, seq_len, embed_dim)
        attn_mask: (batch_size, seq_len, seq_len) 或 None  -> 需要是广播后能与 scores 相加的形状
        """
        B, T, C = x.size()

        q = self.q_proj(x)  # (B, T, C)
        k = self.k_proj(x)  # (B, T, C)
        v = self.v_proj(x)  # (B, T, C)

        # reshape for multi-head: (B, heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, heads, T, head_dim)

        # 计算注意力分数
        # (B, heads, T, head_dim) @ (B, heads, head_dim, T) -> (B, heads, T, T)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            # attn_mask 可以是形状 (B, 1, T, T) 或 (B, heads, T, T) 等，尽量广播友好
            scores = scores + attn_mask

        attn = F.softmax(scores, dim=-1)  # (B, heads, T, T)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        # 加权和
        context = attn @ v  # (B, heads, T, head_dim)
        # 还原形状
        context = context.transpose(1, 2).contiguous().view(B, T, C)  # (B, T, C)

        out = self.out_proj(context)  # (B, T, C)
        return out, attn

class FeedForward(nn.Module):
    def __init__(self, embed_dim, ffn_dim, dropout=0.0, activation=nn.GELU()):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.activation = activation
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.dropout = dropout

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_dim, dropout=0.0, activation=nn.GELU(), layer_norm_eps=1e-5):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout=dropout)
        self.ffn = FeedForward(embed_dim, ffn_dim, dropout=dropout, activation=activation)

        self.norm1 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(embed_dim, eps=layer_norm_eps)
        self.dropout = dropout

    def forward(self, x, attn_mask=None):
        # 第1层归一化后自注意力（Pre-LN 版本若改成 Post-LN 需调整顺序）
        residual = x
        x = self.norm1(x)
        attn_out, _ = self.self_attn(x, attn_mask=attn_mask)
        if self.dropout:
            attn_out = F.dropout(attn_out, p=self.dropout, training=self.training)
        x = residual + attn_out

        # 第2层前馈网络
        residual = x
        x = self.norm2(x)
        ffn_out = self.ffn(x)
        if self.dropout:
            ffn_out = F.dropout(ffn_out, p=self.dropout, training=self.training)
        x = residual + ffn_out

        return x

# transformer层
class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, embed_dim, num_heads, ffn_dim, dropout=0.0, activation=nn.GELU()):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                activation=activation
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-5)

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = self.norm(x)
        return x
