# BlockSparseAttention for Ascend NPU

本仓库提供 block sparse-attn 在华为昇腾 NPU 上的实现。

## 安装

**环境要求：**

- CANN
- PyTorch 2.1 及以上
- torch_npu
- `packaging` Python 包（`pip install packaging`）
- `psutil` Python 包（`pip install psutil`）
- `ninja` Python 包（`pip install ninja`）
- Linux

**安装步骤：**

设置环境变量：
```

```

从源码编译：
```bash
git clone https://github.com/your-repo/Block-Sparse-Attention.git
cd Block-Sparse-Attention
git submodule update --init --recursive
python setup.py install
```

## 测试

运行测试：
```bash
pytest -q -s block_sparse_tests/test_bsa_attn_npu.py
```

## 接口说明

```python
def block_sparse_attn_func(
    q_unpad, k_unpad, v_unpad,
    cu_seqlens_q, cu_seqlens_k,
    head_mask_type,
    streaming_info,
    base_blockmask,
    max_seqlen_q_, max_seqlen_k_,
    p_dropout,
    deterministic=False,
    softmax_scale=None,
    is_causal=False,
    exact_streaming=False,
    return_attn_probs=False,
):
```

Block sparse attention forward pass for Ascend NPU。

对 Q/K/V 进行分块稀疏注意力计算，支持 TND（Total-NumHeads-Dim）格式输入，
结合 `base_blockmask` 指定每个 head 在 Q/KV 块网格上的稀疏模式，跳过掩码为零的块。

## 支持特性

| 特性 | 支持状态 |
|------|---------|
| FP16 (float16) | ✅ |
| BF16 (bfloat16) | ✅ |
| 分块稀疏掩码 (Block Sparse Mask) | ✅ |
| 变长序列 (TND) | ✅ |
| MQA/GQA | ✅ |
| 因果注意力 (Causal) | ❌ |
| 滑动窗口注意力 (Sliding Window) | ❌ |
| Dropout | ❌ |
| 流式注意力 (Streaming Attention) | ❌ |
| 精确流式 (Exact Streaming) | ❌ |
| 返回注意力概率 (Return Attn Probs) | ❌ |
| 反向传播 (Backward) | ❌ |
| 分页 KV 缓存 (Paged KV Cache) | ❌ |
| 旋转位置编码 (RoPE) | ❌ |
| FP8 量化 | ❌ |

### 参数

| 参数 | 形状 / 类型 | 说明 |
|------|-------------|------|
| `q_unpad` | `(total_q, num_heads, headdim)` | TND 格式的 Query，`total_q = sum(seqlen_i)` |
| `k_unpad` | `(total_k, num_heads_k, headdim)` | TND 格式的 Key，支持 GQA/MQA |
| `v_unpad` | `(total_k, num_heads_k, headdim_v)` | TND 格式的 Value，`headdim_v` 通常等于 `headdim` |
| `cu_seqlens_q` | `(batch_size + 1,)`，`torch.int64` | Q 的累积序列长度（前缀和），用于从 TND 格式还原每个 batch 的序列范围 |
| `cu_seqlens_k` | `(batch_size + 1,)`，`torch.int64` | KV 的累积序列长度 |
| `head_mask_type` | `(num_heads,)`，`torch.int32` | 每个 head 的掩码类型标记，用于区分不同 head 的稀疏策略。当前实现中 `1` 表示激活 |
| `streaming_info` | `(num_heads, 2)`，`torch.int32`，可选 | 流式注意力参数。**当前 NPU kernel 未实现**，仅 Python 接口预留 |
| `base_blockmask` | `(batch_size, num_heads, max_q_block, max_kv_block)`，`torch.int8/uint8`，可选 | 块稀疏掩码。`0` 表示跳过该 Q×KV 块，非零表示参与计算 |
| `max_seqlen_q_` | `int`，可选 | 所有 batch 中 Q 的最大序列长度，用于内存分配 |
| `max_seqlen_k_` | `int`，可选 | 所有 batch 中 KV 的最大序列长度 |
| `p_dropout` | `float` | dropout 概率。当前 NPU kernel 强制要求 `p_dropout == 0.0` |
| `deterministic` | `bool` | 是否确定性执行（影响 dropout 掩码生成），默认为 `False` |
| `softmax_scale` | `float`，可选 | QK^T 缩放因子，默认为 `1 / sqrt(headdim)` |
| `is_causal` | `bool` | 是否应用因果掩码（对齐到注意力矩阵右下角），默认为 `False` |
| `exact_streaming` | `bool` | 是否使用精确流式注意力模式。**当前 NPU kernel 未实现**，仅 Python 接口预留，默认为 `False` |
| `return_attn_probs` | `bool` | 是否返回注意力概率矩阵，默认为 `False` |

### 返回

- **`out`**: `(total_q, num_heads, headdim)` — 注意力输出。
- 如果 `return_attn_probs=True`，返回 `(out, softmax_lse, S_dmask)`：
  - `softmax_lse`: `(batch_size, num_heads, max_seqlen_q_rounded)`，每行的 logsumexp。
  - `S_dmask`: `(batch_size, num_heads, max_seqlen_q_rounded, max_seqlen_k_rounded)`，softmax 后的注意力概率（仅当 `p_dropout > 0` 时有意义）。