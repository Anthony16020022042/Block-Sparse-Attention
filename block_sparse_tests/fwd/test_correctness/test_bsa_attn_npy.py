import pytest
import torch
import torch_npu
from einops import rearrange, repeat
from block_sparse_attn import block_sparse_attn_func
from block_sparse_attn.block_sparse_attn_golden import block_sparse_attn_golden

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