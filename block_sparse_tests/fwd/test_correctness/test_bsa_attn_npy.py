import math
import pytest
import torch
import torch_npu
from einops import rearrange, repeat
from typing import Optional, Tuple
from block_sparse_attn import block_sparse_attn_func


def construct_streaming_mask(
    seqlen_q, seqlen_k, sink_size, local_size, device, causal=False
):
    row_idx = rearrange(torch.arange(seqlen_q, device=device), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device)
    offset = seqlen_k - seqlen_q
    if causal:
        future_bound = torch.minimum(
            row_idx + offset, torch.tensor(seqlen_k, device=device)
        )
        mask = torch.logical_or(
            col_idx > future_bound,
            torch.logical_and(
                col_idx < row_idx + offset - (local_size - 1),
                col_idx >= sink_size,
            ),
        )
    else:
        mask = torch.logical_or(
            col_idx > row_idx + offset,
            torch.logical_and(
                col_idx < row_idx + offset - (local_size - 1),
                col_idx >= sink_size,
            ),
        )
    return mask


def construct_local_mask(seqlen_q, seqlen_k, window_size, device):
    row_idx = rearrange(torch.arange(seqlen_q, device=device), "s -> s 1")
    col_idx = torch.arange(seqlen_k, device=device)
    offset = seqlen_k - seqlen_q
    if window_size[0] < 0:
        return col_idx > row_idx + offset + window_size[1]
    else:
        return torch.logical_or(
            col_idx > torch.minimum(row_idx + offset + window_size[1], torch.tensor(seqlen_k, device=device)),
            col_idx < row_idx + offset - window_size[0],
        )


def expand_block_mask_to_element(blockmask, m_block_dim, n_block_dim, seqlen_q, seqlen_k):
    expanded = repeat(blockmask, "b h nrow ncol -> b h (nrow d_m) (ncol d_n)", d_m=m_block_dim, d_n=n_block_dim)
    expanded = expanded[:, :, :seqlen_q, :seqlen_k]
    return expanded


def _build_sparse_exclude_mask(
    batch_size, nheads, seqlen_q, seqlen_k,
    base_blockmask, head_mask_type, streaming_info,
    m_block_dim, n_block_dim, exact_streaming, device,
):
    mask = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device)
    if head_mask_type is None and base_blockmask is None:
        return mask
    hmt = head_mask_type.clone() if head_mask_type is not None else None
    if hmt is not None:
        ones_mask = hmt == 1
        count = torch.cumsum(ones_mask, dim=-1).to(hmt.dtype)
        count = count * ones_mask
        hmt = hmt.masked_scatter(ones_mask, count[ones_mask])
    for h in range(nheads):
        mask_type = hmt[h].item() if hmt is not None else 1
        if mask_type == 0:
            continue
        elif mask_type > 0:
            if base_blockmask is None:
                continue
            active = base_blockmask[:, mask_type - 1: mask_type]
            active = expand_block_mask_to_element(active, m_block_dim, n_block_dim, seqlen_q, seqlen_k)
            mask[:, h: h + 1] = mask[:, h: h + 1] | ~active
        else:
            if streaming_info is None:
                continue
            sink_size = streaming_info[h * 2].item()
            local_size = streaming_info[h * 2 + 1].item()
            str_mask = construct_streaming_mask(seqlen_q, seqlen_k, sink_size, local_size, device, causal=exact_streaming)
            mask[:, h: h + 1] = mask[:, h: h + 1] | str_mask
    return mask


def block_sparse_attn_golden(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    base_blockmask: Optional[torch.Tensor] = None,
    head_mask_type: Optional[torch.Tensor] = None,
    streaming_info: Optional[torch.Tensor] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    m_block_dim: int = 128, n_block_dim: int = 128,
    exact_streaming: bool = False,
    dropout_mask: Optional[torch.Tensor] = None,
    upcast: bool = True,
    return_attn_probs: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    if is_causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    batch_size, seqlen_q, nheads, d = q.shape
    _, seqlen_k, nheads_k, _ = k.shape
    if softmax_scale is None:
        softmax_scale = d ** (-0.5)
    k = repeat(k, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    v = repeat(v, "b s h d -> b s (h g) d", g=nheads // nheads_k)
    scores = torch.einsum("bthd,bshd->bhts", q * softmax_scale, k)
    exclude = torch.zeros(batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device)
    if query_padding_mask is not None:
        exclude = exclude | rearrange(~query_padding_mask, "b s -> b 1 s 1")
    if key_padding_mask is not None:
        exclude = exclude | rearrange(~key_padding_mask, "b s -> b 1 1 s")
    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(seqlen_q, seqlen_k, window_size, q.device)
        exclude = exclude | rearrange(local_mask, "t s -> 1 1 t s")
    if base_blockmask is not None or head_mask_type is not None:
        sparse_exclude = _build_sparse_exclude_mask(
            batch_size, nheads, seqlen_q, seqlen_k,
            base_blockmask, head_mask_type, streaming_info,
            m_block_dim, n_block_dim, exact_streaming, q.device,
        )
        exclude = exclude | sparse_exclude
    scores = scores.masked_fill(exclude, float("-inf"))
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    attention = attention.masked_fill(exclude, 0.0)
    dropout_scaling = 1.0 / (1 - p_dropout) if p_dropout > 0.0 else 1.0
    attention_drop = attention * dropout_mask.to(attention.dtype) if dropout_mask is not None else attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)
    if query_padding_mask is not None:
        output = output.masked_fill(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)
    output = output.to(dtype=dtype_og)
    if return_attn_probs:
        return output, attention.to(dtype=dtype_og)
    return output


def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, batch_size, num_blocksparse_heads, sparsity_list, causal=False, device="npu:0"):
    assert len(sparsity_list) == num_blocksparse_heads
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, device=device, dtype=torch.bool)
    
    for batch in range(batch_size):
        for head_rank in range(num_blocksparse_heads):
            sparsity = sparsity_list[head_rank]
            if not sparsity == 0.0 and not sparsity == 1.0:
                for i in range(nrow):
                    idx = nrow - i - 1
                    if causal:
                        available_col_num = max(0, ncol - i)
                        num_one = max(1, int(sparsity * available_col_num))
                        base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
                    else:
                        available_col_num = ncol
                        num_one = max(1, int(sparsity * available_col_num))
                        base_mask[batch][head_rank][idx, torch.randperm(available_col_num)[:num_one]] = True
            elif sparsity == 1.0:
                base_mask[batch][head_rank] = torch.ones_like(base_mask[batch][head_rank])
                
    return base_mask.to(torch.int8)

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal)
    (torch.bfloat16, 1, 1, 1, 128, 128, 128, True)
]

def print_tensor_full(name, tensor):
    # 先打印基础信息
    print(f"\n===== {name} 完整数值 =====")
    print(f"shape: {tensor.shape}  dtype: {tensor.dtype}  device: {tensor.device}")
    
    # 打印真实值（自动转到CPU打印）
    print("数值内容：")
    print(tensor.detach().cpu())

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal", test_cases)
def test_bsa_varlen_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal):
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size * q_seqlen, num_heads, head_size)).to(data_type).npu()
    key = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    actual_seq_len = torch.full((batch_size,), q_seqlen, dtype=torch.int64, device="npu:0")
    actual_kv_len = torch.full((batch_size,), kv_seqlen, dtype=torch.int64, device="npu:0")

    print_tensor_full("query", query)
    print_tensor_full("key", key)
    print_tensor_full("value", value)
    print_tensor_full("actual_seq_len", actual_seq_len)
    print_tensor_full("actual_kv_len", actual_kv_len)


    max_seqlen_q = q_seqlen
    max_seqlen_k = kv_seqlen
    dropout_p = 0.0
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    return_attn_probs = False
    block_table = None
    head_mask_type = torch.tensor([0] * num_heads, device="npu:0", dtype=torch.int32)
    streaming_info = None

    sparsity = 1
    sparsity_list = [sparsity] * num_heads
    block_size = 128
    base_blockmask = generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, block_size, block_size, block_size, batch_size, num_heads, sparsity_list)
    print("[wjc] start")
    result = block_sparse_attn_func(
        query, 
        key, 
        value,
        actual_seq_len, 
        actual_kv_len,
        head_mask_type,
        streaming_info,
        base_blockmask,
        max_seqlen_q, 
        max_seqlen_k,
        dropout_p,
        deterministic=True,
        softmax_scale=scale,
        is_causal=is_causal,
        exact_streaming=False,
        return_attn_probs=return_attn_probs,
    )
    print("[wjc] end")
    # ==========================================
    # 🔥 万能打印：自动识别类型、长度、内容、shape
    # ==========================================
    print("\n" + "="*50)
    print("📌 函数返回结果类型:", type(result))
    print("📌 长度/元素个数:", len(result) if isinstance(result, (list, tuple)) else "不是列表")

    # 逐个打印每个返回值
    for idx, item in enumerate(result):
        print(f"\n返回值 [{idx}] 类型: {type(item)}")
        if hasattr(item, 'shape'):
            print(f"           shape: {item.shape}")
        if hasattr(item, 'dtype'):
            print(f"           dtype: {item.dtype}")
        print(f"           内容: {item}")

    print("="*50 + "\n")

    # ========== golden 对比 ==========
    # 构建标准 cu_seqlens（累积格式）用于 un-padded <-> padded 转换
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * q_seqlen, step=q_seqlen,
        dtype=torch.int32, device="npu:0"
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * kv_seqlen, step=kv_seqlen,
        dtype=torch.int32, device="npu:0"
    )

    # NPU 输出是 un-padded, 转回 padded
    out_npu = result  # (total_tokens, num_heads, head_size)
    out_npu_pad = torch.zeros(
        batch_size, q_seqlen, num_heads, head_size,
        dtype=out_npu.dtype, device=out_npu.device
    )
    for b in range(batch_size):
        s = cu_seqlens_q[b].item()
        e = cu_seqlens_q[b + 1].item()
        out_npu_pad[b, :e - s] = out_npu[s:e]

    # 从 un-padded QKV 还原 padded QKV
    q_pad = rearrange(query, "(b s) h d -> b s h d", b=batch_size, s=q_seqlen)
    k_pad = rearrange(key,   "(b s) h d -> b s h d", b=batch_size, s=kv_seqlen)
    v_pad = rearrange(value, "(b s) h d -> b s h d", b=batch_size, s=kv_seqlen)

    # Golden 参考
    out_golden = block_sparse_attn_golden(
        q_pad, k_pad, v_pad,
        base_blockmask=base_blockmask,
        head_mask_type=head_mask_type,
        streaming_info=streaming_info,
        is_causal=is_causal,
        softmax_scale=scale,
        upcast=True,
    )

    diff = (out_npu_pad.float() - out_golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_npu_pad.float().flatten().unsqueeze(0),
        out_golden.float().flatten().unsqueeze(0),
    ).item()

    print(f"\n===== NPU vs Golden 对比 =====")
    print(f"Max diff : {max_diff:.6e}")
    print(f"Mean diff: {mean_diff:.6e}")
    print(f"Cosine similarity: {cos_sim:.8f}")
    if max_diff > 1e-2:
        print("⚠️  差异较大, 请检查！")
    else:
        print("✅ 结果在合理误差范围内")
    print("="*50)