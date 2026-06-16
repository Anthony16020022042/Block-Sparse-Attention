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

**注意：**
- 不支持反向传播的 NPU kernel（C++ 侧 backward 为桩代码）。
- 不支持 dropout，`p_dropout` 必须为 `0.0`。
- 不支持滑动窗口，`window_size` 固定为 `(-1, -1)`。
- 不支持流式注意力（streaming attention）和精确流式模式（exact_streaming）。`streaming_info` 和 `exact_streaming` 参数虽在 Python 接口中预留，但 NPU kernel 中未实现对应逻辑，`maskType` 固定为 `NO_MASK`。

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

## 1 Tiling 切分

### 1.1 任务分解

计算被分解为 **Task × KV-Chunk × Pipeline-Stage** 三个维度的迭代。

**Task 定义：** 每个 Task 对应一个 `(qSBlockIdx, qNBlockIdx)` 对，即一个 Q 空间块 × 一个 N-Group（head 组）。Task 总数由所有 batch 的 `curQNBlockNum * curQSBlockNum` 累加得到。

```
Task 总数 = Σ_batch ( qNBlockNumPerGroup × kvHeads × GetQBlocks(qSeqlen, qBlockX) )
```

所有 Task 在硬件 AI Core 上通过 `coreIdx` 均匀分配：
```cpp
for (uint32_t taskIdx = coreIdx; taskIdx < totalTaskNum; taskIdx += coreNum)
```

### 1.2 四层 Tiling 层次

| 层次 | 维度 | 分块大小 | 说明 |
|------|------|---------|------|
| Batch | B | 动态 | 每个 batch 独立计算 seqlen，Task 按 batch 累积，通过 `curTotalTaskNum` 边界切换 |
| Q 空间块 (X) | Q seq | `qBlockX = 128` → `BASIC_BLOCK_SIZE = 128` | X 块内再切 `qBlockInX` 个 S 块 |
| Head 组 (N) | heads | `curQNBlockTile × group` | `groupSize = qHeads / kvHeads`，支持 GQA/MQA |
| KV 空间块 (Y) | KV seq | `pagedBlockSize = 128` | 通过 `selectIdx` 跳过掩码为零的块，不连续遍历 |

**Task 内部拆解（`mha_varlen_fwd_block.cpp` 514-526）：**
```cpp
uint32_t qSBlockIdx   = taskIdxCurBatch / curQNBlockNum;      // 空间块索引
uint32_t qNBlockIdx   = taskIdxCurBatch % curQNBlockNum;      // N-Group 索引
uint32_t kvHeadIdx    = qNBlockIdx / qNBlockNumPerGroup;      // KV head
uint32_t qHeadIdx     = kvHeadIdx × groupSize + qNBlockIdxCurGroup × curQNBlockTile;
```

### 1.3 稀疏块选取

不使用连续 KV 循环，而是从 `gSelectIdx` 中读出当前 Task 选中的 KV Y 块索引，仅在这些块上执行 QK 和 PV 计算：

```
selected_kv_y_blocks = gSelectIdx[curSelectIdx × maxKvBlockNum : curSelectIdx × maxKvBlockNum + curSelectNum]
```

每个选中的 Y 块内再按 `pagedBlockSize` 分为若干 `kvSLoop` 迭代，并通过 `blockStackNum = MAX_KV_STACK_LEN / pagedBlockSize` 将多个连续迭代合并为一个 `stack` 以减少循环开销。

### 1.4 掩码预处理（Mask → SelectIdx）

在 Vector 核心执行 `Mask2IdxAndCount`：将稠密 blockmask `(B, N, maxQBlock, maxKvBlock)` 转换为稀疏索引格式 `selectIdx [QBlockNum, N, maxKvBlockNum]` 和 `selectNumIdx [QBlockNum, N]`，消除值为 0 的掩码块，使主循环只需遍历有效块。

### 1.5 Workspace 布局

Global memory workspace 按以下偏移划分：

```
[0                  )  S 矩阵 (QK^T 结果)   → mm1OutSize
[mm1OutSize         )  P 矩阵 (Softmax 输出) → smOnlineOutSize
[+ smOnlineOutSize  )  O_tmp (PV 中间结果)  → mm2OutSize
[+ mm2OutSize       )  O_update (重缩放)    → updateSize
[+ updateSize        )  selectNumIdx         → selectNumIdxSize
[+ selectNumIdxSize  )  selectIdx
```

每个 core 在 S/P/O_tmp 区域有 `(PRE_LAUNCH + 1) × WORKSPACE_BLOCK_SIZE_DB` 的 ping-pong 槽位，用于流水线重叠。

---

## 2 Kernel 方案

### 2.1 片上内存分配策略

Ascend AtlasA2 架构包含三种片上存储：**L0A/L0B/L0C**（Cube 核心）、**L1**（Cube 核心）和 **UB（Unified Buffer）**（Vector 核心）。

#### L0 分配（Cube 核心）

每个 L0 缓冲区按 `STAGES = 2` 切分为 ping-pong 槽位：

```
L0A_PINGPONG_BUF_SIZE = L0A_SIZE / 2     // 存放 A 分片 (Q/P)
L0B_PINGPONG_BUF_SIZE = L0B_SIZE / 2     // 存放 B 分片 (K/V)
L0C_PINGPONG_BUF_SIZE = L0C_SIZE / 2     // 存放 C 分片 (S/O_tmp)
```

地址通过 `l0ABPingPongFlag` 交替切换。

#### L1 分配（Cube 核心）

L1 缓冲区由 `blockMmadQK` 和 `blockMmadPV` 共享：

**QK Matmul 阶段：**

| 缓冲区 | 大小 | 说明 |
|--------|------|------|
| `l1ATensor` (Q) | `L1TileShapeQK::M × L1TileShapeQK::K × sizeof(ElementQ)` | 每个 Task 只加载一次 |
| `l1BTensor[2]` (K) | `2 × L1TileShapeQK::N × L1TileShapeQK::K × sizeof(ElementK)` | 双缓冲 ping-pong |

其中 `L1TileShapeQK = GemmShape<128, 128, 128>`。

**PV Matmul 阶段：**

| 缓冲区 | 大小 | 说明 |
|--------|------|------|
| `l1ATensor[2]` (P) | `2 × L1TileShapePV::M × L1TileShapePV::K × sizeof(ElementP)` | 双缓冲 ping-pong |
| `l1BTensor` (V) | `L1TileShapePV::N × L1TileShapePV::K × sizeof(ElementV)` | 单缓冲，V 可一次性加载 |

其中 `L1TileShapePV = GemmShape<128, 128, 256>`。

L1 偏移对齐：PV 的 `l1ATensor` 起始地址在 QK 的 L1 使用量之后（`L1_QK_SIZE` = `L1A_SIZE + 2 × L1B_SIZE`）。

#### UB 分配（Vector 核心）

UB 划分为多个区域，用于 softmax 和 rescale 的 Vector 计算：

| 偏移（字节） | 区域 | 类型 | 大小 |
|-------------|------|------|------|
| 0 | `lsUbTensor` — S 矩阵 | `float` | 4 × 16384 |
| 4 × 16384 | `lpUbTensor` — P 矩阵 | `half/bf16` | 4 × 16384 |
| 4 × 16384 | `maskUbTensor` — Mask（与 P 共享空间） | `int8_t` | 4 × 16384 |
| 10 × 16384 + 8 × 1024 | `lmUbTensor` — 局部 max | `float` | 1 × 1024 |
| + 1 × 1024 | `hmUbTensor` — 全局 max | `float` | 1 × 1024 |
| + 1 × 1024 | `gmUbTensor` — 上一轮全局 max | `float` | 1 × 1024 |
| + 1 × 1024 | `llUbTensor` — 局部 sum | `float` | 1 × 1024 |
| + 1 × 1024 | `glUbTensor` — 全局 sum | `float` | 1 × 1024 |
| + 1 × 1024 | `dmUbTensor` — delta max | `float` | 1 × 1024 |
| 10 × 16384 | `tvUbTensor` — 临时/转置向量 | `float` | 10 × 16384 |

S 矩阵使用 ping-pong 地址切换：`sUbOffset = pingpongFlag × MAX_UB_S_ELEM_NUM`（`MAX_UB_S_ELEM_NUM = 8192` floats）。

---

### 2.2 片上流水编排策略

Kernel 采用 **Cube + Vector 双核异构流水线**，通过 `CrossCoreFlag` 硬件信号实现核间同步。主循环每轮处理 `blockStackNum` 个 KV 块，分为 4 个阶段：

```
  迭代 t     |  QK(t)  |  Softmax(t)  |  PV(t)  |  Rescale(t)  |  ...
  迭代 t+1   |         |  QK(t+1)     |  Softmax(t+1) |  PV(t+1) |  Rescale(t+1) | ...
  ─────────────────────────────────────────────────────────────────────────────
  CUBE 核心  |  QK(t) 🟦  |             |  PV(t) 🟦     |            |  QK(t+2) 🟦
  VEC 核心   |          |  Softmax(t) 🟩 |              |  Rescale(t) 🟩 |  Softmax(t+2) 🟩
```

**流水线阶段详情：**

1. **Stage 1 — QK Matmul（Cube 核心）**
   - DMA GM→L1：Q 加载一次（`copyGmToL1A`），K 按 tile 逐段加载（`copyGmToL1B`）
   - DMA L1→L0：`copyL1ToL0A` / `copyL1ToL0B` 按 `mL0Idx × kL0Idx` 子 tile 分步
   - 计算：`tileMmad` — 矩阵乘累加
   - 写回：`copyL0CToGm` → S 矩阵写入 workspace
   - 完成后通过 `CrossCoreSetFlag<0x2, PIPE_FIX>(qkReady)` 通知 Vector 核心

2. **Stage 2 — Online Softmax（Vector 核心）**
   - 等待 `CrossCoreWaitFlag(qkReady)`
   - DMA GM→UB：`CopySGmToUb` 加载 S 分片
   - 向量计算：`ScaleS` → `CalcLocalRowMax` → `UpdateGlobalRowMax` → `CalcExp` → `DownCastP` → `CalcLocalRowSum` → `UpdateGlobalRowSum`
   - DMA UB→GM：`CopyPUbToGm` 将 P 写回 workspace
   - 通过 `SetFlag<0x2, PIPE_FIX>(softmaxReady)` 通知 Cube 核心

3. **Stage 3 — PV Matmul（Cube 核心）**
   - DMA GM→L1：V 加载一次（`copyGmToL1B`）
   - 等待 `CrossCoreWaitFlag(softmaxReady)`
   - DMA GM→L1：P 按 tile 加载（`copyGmToL1A`，ping-pong）
   - 计算：P 与 V 的矩阵乘
   - 写回：O_tmp 写入 workspace
   - 通过 `CrossCoreSetFlag<0x2, PIPE_FIX>(pvReady)` 通知 Vector 核心

4. **Stage 4 — Rescale O（Vector 核心）**
   - 等待 `CrossCoreWaitFlag(pvReady)`
   - 加载 O_tmp，应用 safe softmax 的在线重缩放（`exp(prev_lse - new_lse) × O_prev + P×V`）
   - 最终 O 写回 global memory 输出

**预启动深度：** `preKVNum = PRE_LAUNCH × blockStackNum`（`PRE_LAUNCH = 2`），使得 QK/S/PV 三个阶段之间实现 2 轮迭代的流水线重叠。主循环遍历范围为 `kvSLoopNumTotal + preKVNum`：前 `preKVNum` 轮仅执行 QK（PV 阶段空转），最后 `preKVNum` 轮仅执行 PV 和 Rescale（QK 阶段空转），形成完整的流水线填充和排空。

```
循环迭代:     kvSIdx=0         kvSIdx=1         kvSIdx=2         kvSIdx=3
QK Matmul:  [QK_0─────────]  [QK_1─────────]  [QK_2─────────]
Softmax:                      [SM_0─────────]  [SM_1─────────]  [SM_2─────────]
PV Matmul:                                     [PV_0─────────]  [PV_1─────────]  [PV_2─────────]
Rescale O:                                                      [RS_0─────────]  [RS_1─────────]
```

**跨核同步机制：** 使用 `Arch::CrossCoreFlag` 硬件信号量（`QK_READY_ID = 1`, `SOFTMAX_READY_ID = 2`, `PV_READY_ID = 3`），Cube 核心通过 `CrossCoreSetFlag<0x2>` 设置信号，Vector 核心通过 `CrossCoreWaitFlag` 等待，`0x2` 表示目标管道为 Vector（V）核心。