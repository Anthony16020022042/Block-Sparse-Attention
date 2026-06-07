"""
对比 NPU kernel 输出与 golden 参考实现。

用法:
    python compare_npu_golden.py

依赖:
    torch, torch_npu, einops
"""

import torch
import torch_npu
from einops import rearrange
from block_sparse_attn import block_sparse_attn_func
from block_sparse_attn.block_sparse_attn_golden import block_sparse_attn_golden


def build_cu_seqlens(seqlens, device):
    """(batch,) -> (batch+1,) cumulative"""
    return torch.cat([
        torch.zeros(1, dtype=torch.int32, device=device),
        seqlens.cumsum(dim=0, dtype=torch.int32),
    ])


def npu_to_padded(unpadded, cu_seqlens, batch_size, max_seqlen, nheads, d):
    """
    将 NPU kernel 输出的 un-padded (total_tokens, nheads, d)
    还原为 padded (batch_size, max_seqlen, nheads, d)
    """
    out = torch.zeros(batch_size, max_seqlen, nheads, d,
                      dtype=unpadded.dtype, device=unpadded.device)
    for b in range(batch_size):
        start = cu_seqlens[b].item()
        end = cu_seqlens[b + 1].item()
        out[b, :end - start] = unpadded[start:end]
    return out


def padded_to_unpadded(padded, cu_seqlens_q, batch_size, max_seqlen_q):
    """padded -> un-padded"""
    total_q = cu_seqlens_q[-1].item()
    B, S, H, D = padded.shape
    flat = rearrange(padded, "b s h d -> (b s) h d")
    mask = torch.arange(S, device=padded.device).unsqueeze(0) < (
        cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    ).unsqueeze(1)
    indices = torch.nonzero(mask.flatten()).squeeze()
    return flat[indices]


def compare(
    dtype=torch.bfloat16,
    batch_size=1,
    nheads=4,
    kv_heads=4,
    seqlen_q=128,
    seqlen_k=128,
    head_dim=128,
    is_causal=True,
    sparsity=0.5,
    p_dropout=0.0,
):
    device = "npu:0"
    block_size = 128

    # -------------------- 生成随机 QKV（padded）--------------------
    torch.manual_seed(42)
    q_pad = torch.randn(batch_size, seqlen_q, nheads, head_dim,
                        dtype=dtype, device=device)
    k_pad = torch.randn(batch_size, seqlen_k, kv_heads, head_dim,
                        dtype=dtype, device=device)
    v_pad = torch.randn(batch_size, seqlen_k, kv_heads, head_dim,
                        dtype=dtype, device=device)

    # -------------------- 构建 mask --------------------
    # cu_seqlens（等长，无 padding）
    cu_seqlens_q = build_cu_seqlens(
        torch.full((batch_size,), seqlen_q, device=device), device)
    cu_seqlens_k = build_cu_seqlens(
        torch.full((batch_size,), seqlen_k, device=device), device)
    max_seqlen_q = seqlen_q
    max_seqlen_k = seqlen_k

    # head_mask_type: 全部 block-sparse
    head_mask_type = torch.ones(nheads, dtype=torch.int32, device=device)
    streaming_info = None

    # base_blockmask: block-level 稀疏 mask
    nrow = (seqlen_q + block_size - 1) // block_size
    ncol = (seqlen_k + block_size - 1) // block_size
    base_blockmask = torch.zeros(batch_size, nheads, nrow, ncol,
                                 dtype=torch.bool, device=device)
    for b in range(batch_size):
        for h in range(nheads):
            for r in range(nrow):
                na = max(1, int(sparsity * ncol))
                perm = torch.randperm(ncol, device=device)[:na]
                if is_causal:
                    # causal: 只能选对角线及之前的 block
                    available = min(ncol, r + 1)
                    na = max(1, int(sparsity * available))
                    perm = torch.randperm(available, device=device)[:na]
                base_blockmask[b, h, r, perm] = True

    softmax_scale = head_dim ** (-0.5)

    # -------------------- 1) NPU kernel --------------------
    # 将 padded QKV 转为 un-padded
    total_q = cu_seqlens_q[-1].item()
    total_k = cu_seqlens_k[-1].item()
    q_unpad = rearrange(q_pad, "b s h d -> (b s) h d")[:total_q].contiguous()
    k_unpad = rearrange(k_pad, "b s h d -> (b s) h d")[:total_k].contiguous()
    v_unpad = rearrange(v_pad, "b s h d -> (b s) h d")[:total_k].contiguous()

    out_npu = block_sparse_attn_func(
        q_unpad, k_unpad, v_unpad,
        cu_seqlens_q, cu_seqlens_k,
        head_mask_type, streaming_info, base_blockmask,
        max_seqlen_q, max_seqlen_k,
        p_dropout,
        deterministic=True,
        softmax_scale=softmax_scale,
        is_causal=is_causal,
        exact_streaming=False,
        return_attn_probs=False,
    )
    # out_npu 是 un-padded (total_q, nheads, head_dim)
    out_npu_padded = npu_to_padded(
        out_npu, cu_seqlens_q, batch_size, max_seqlen_q, nheads, head_dim)

    # -------------------- 2) Golden 参考 --------------------
    out_golden = block_sparse_attn_golden(
        q_pad, k_pad, v_pad,
        base_blockmask=base_blockmask,
        head_mask_type=head_mask_type,
        is_causal=is_causal,
        softmax_scale=softmax_scale,
        upcast=True,
    )

    # -------------------- 3) 对比 --------------------
    diff = (out_npu_padded.float() - out_golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        out_npu_padded.float().flatten().unsqueeze(0),
        out_golden.float().flatten().unsqueeze(0),
    ).item()

    print(f"dtype={dtype}, B={batch_size}, H={nheads}, "
          f"q={seqlen_q}, k={seqlen_k}, d={head_dim}, "
          f"causal={is_causal}, sparsity={sparsity}")
    print(f"  Max diff : {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")
    print(f"  Cosine sim: {cos_sim:.8f}")

    # 宽松阈值（kernel 通常与 fp32 golden 在 <= 2*ulp 内匹配）
    rtol = 2 * (out_npu_padded.float() - out_golden.float()).abs().max().item()
    print(f"  RTOL bound: {rtol:.6e}")
    print()

    return max_diff


if __name__ == "__main__":
    # 基础 case
    compare(dtype=torch.bfloat16, batch_size=1, nheads=4, seqlen_q=128,
            seqlen_k=128, head_dim=128, is_causal=True, sparsity=0.5)

    # 非 causal
    compare(dtype=torch.bfloat16, batch_size=1, nheads=4, seqlen_q=128,
            seqlen_k=128, head_dim=128, is_causal=False, sparsity=0.5)

    # GQA
    compare(dtype=torch.bfloat16, batch_size=1, nheads=8, kv_heads=2,
            seqlen_q=256, seqlen_k=256, head_dim=64, is_causal=True, sparsity=0.3)

    # 不同 seqlen
    compare(dtype=torch.bfloat16, batch_size=1, nheads=4, seqlen_q=64,
            seqlen_k=192, head_dim=128, is_causal=True, sparsity=0.7)
