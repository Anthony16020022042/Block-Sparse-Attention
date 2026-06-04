import pytest
import torch
import torch_npu
from einops import repeat
from block_sparse_attn import block_sparse_attn_func

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
                
    return base_mask

test_cases = [
    # (data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal)
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True)
]

@pytest.mark.parametrize("data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal", test_cases)
def test_bsa_varlen_ops(data_type, batch_size, num_heads, kv_heads, q_seqlen, kv_seqlen, head_size, is_causal):
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size * q_seqlen, num_heads, head_size)).to(data_type).npu()
    key = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    actual_seq_len = torch.full((batch_size,), q_seqlen, dtype=torch.int32, device="npu:0")
    actual_kv_len = torch.full((batch_size,), q_seqlen, dtype=torch.int32, device="npu:0")

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

    sparsity = 0.7
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