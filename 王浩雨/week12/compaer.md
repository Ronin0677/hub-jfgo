

### 主流开源大模型结构维度对比表

| 维度模型 | **位置编码** | **Transformer结构** | **多头注意力机制** | **FFN层设计** | **归一化层选择** | **激活函数** | **是否使用Bias** |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Llama 3 (Meta)** | RoPE | 仅解码器 | GQA | SwiGLU | RMSNorm | SwiGLU (SiLU) | **否**（Linear, Norm层均无） |
| **Qwen 2.5 (阿里)** | RoPE | 仅解码器 | GQA / MHA | SwiGLU | RMSNorm | SwiGLU (SiLU) | **否**（Linear, Norm层均无） |
| **Gemma 2 (Google)** | RoPE | 仅解码器 | MHA / GQA | 普通GeGLU | RMSNorm | GeGLU (GELU) | **是**（部分层保留） |
| **Mixtral 8x7B (Mistral AI)** | RoPE | 仅解码器 | GQA | **MoE**（8个专家，SwiGLU） | RMSNorm | SwiGLU (SiLU) | **否** |
| **DeepSeek-V2 (深度求索)** | RoE + RoPE | **MLA**架构 | **MLA** | **MoE** + **MLA的FFN** | DeepSeekNorm | GLU变体 | **是**（部分层保留） |
| **Command R+ (Cohere)** | RoPE | 仅解码器 | MHA | SwiGLU | RMSNorm | SwiGLU (SiLU) | 未明确，通常为否 |
| **OLMo (AI2)** | ALiBi | 仅解码器 | MHA | SwiGLU | LayerNorm | SwiGLU (SiLU) | **是**（旨在完全透明） |
| **Falcon (TII)** | RoPE | 仅解码器 | **MQA** | 普通FFN (GELU) | LayerNorm | GELU | **否**（强调效率） |
| **BLOOM (BigScience)** | ALiBi | 仅解码器 | MHA | 普通FFN (GELU) | LayerNorm | GELU | **是** |
