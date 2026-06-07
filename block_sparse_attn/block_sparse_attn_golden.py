import math
import torch
from einops import rearrange, repeat
from typing import Optional, Tuple


def round_multiple(x, m):
    return (x + m - 1) // m * m


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


def expand_block_mask_to_element(
    blockmask, m_block_dim, n_block_dim, seqlen_q, seqlen_k
):
    expanded = repeat(
        blockmask,
        "b h nrow ncol -> b h (nrow d_m) (ncol d_n)",
        d_m=m_block_dim,
        d_n=n_block_dim,
    )
    expanded = expanded[:, :, :seqlen_q, :seqlen_k]
    return expanded


def block_sparse_attn_golden(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    base_blockmask: Optional[torch.Tensor] = None,
    head_mask_type: Optional[torch.Tensor] = None,
    streaming_info: Optional[torch.Tensor] = None,
    query_padding_mask: Optional[torch.Tensor] = None,
    key_padding_mask: Optional[torch.Tensor] = None,
    p_dropout: float = 0.0,
    softmax_scale: Optional[float] = None,
    is_causal: bool = False,
    window_size: Tuple[int, int] = (-1, -1),
    m_block_dim: int = 128,
    n_block_dim: int = 128,
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

    exclude = torch.zeros(
        batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=q.device
    )

    if query_padding_mask is not None:
        exclude = exclude | rearrange(~query_padding_mask, "b s -> b 1 s 1")

    if key_padding_mask is not None:
        exclude = exclude | rearrange(~key_padding_mask, "b s -> b 1 1 s")

    if window_size[0] >= 0 or window_size[1] >= 0:
        local_mask = construct_local_mask(
            seqlen_q, seqlen_k, window_size, q.device
        )
        exclude = exclude | rearrange(local_mask, "t s -> 1 1 t s")

    if base_blockmask is not None or head_mask_type is not None:
        sparse_exclude = _build_sparse_exclude_mask(
            batch_size,
            nheads,
            seqlen_q,
            seqlen_k,
            base_blockmask,
            head_mask_type,
            streaming_info,
            m_block_dim,
            n_block_dim,
            exact_streaming,
            q.device,
        )
        exclude = exclude | sparse_exclude

    scores = scores.masked_fill(exclude, float("-inf"))

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    attention = attention.masked_fill(exclude, 0.0)

    dropout_scaling = 1.0 / (1 - p_dropout) if p_dropout > 0.0 else 1.0
    if dropout_mask is not None:
        attention_drop = attention * dropout_mask.to(attention.dtype)
    else:
        attention_drop = attention

    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)

    if query_padding_mask is not None:
        output = output.masked_fill(
            rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0
        )

    output = output.to(dtype=dtype_og)

    if return_attn_probs:
        return output, attention.to(dtype=dtype_og)
    return output


def _build_sparse_exclude_mask(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    base_blockmask,
    head_mask_type,
    streaming_info,
    m_block_dim,
    n_block_dim,
    exact_streaming,
    device,
):
    mask = torch.zeros(
        batch_size, nheads, seqlen_q, seqlen_k, dtype=torch.bool, device=device
    )

    if head_mask_type is None and base_blockmask is None:
        return mask

    hmt = head_mask_type.clone() if head_mask_type is not None else None

    if hmt is not None:
        ones_mask = hmt == 1
        count = torch.cumsum(ones_mask, dim=-1).to(hmt.dtype)
        count = count * ones_mask
        hmt = hmt.masked_scatter(ones_mask, count[ones_mask])

    for h in range(nheads):
        if hmt is not None:
            mask_type = hmt[h].item()
        else:
            mask_type = 1

        if mask_type == 0:
            continue

        elif mask_type > 0:
            if base_blockmask is None:
                continue
            bs_idx = mask_type - 1
            active = base_blockmask[:, bs_idx : bs_idx + 1]
            active = expand_block_mask_to_element(
                active, m_block_dim, n_block_dim, seqlen_q, seqlen_k
            )
            mask[:, h : h + 1] = mask[:, h : h + 1] | ~active

        else:
            if streaming_info is None:
                continue
            sink_idx = h * 2
            local_idx = h * 2 + 1
            sink_size = streaming_info[sink_idx].item()
            local_size = streaming_info[local_idx].item()

            str_mask = construct_streaming_mask(
                seqlen_q, seqlen_k, sink_size, local_size, device,
                causal=exact_streaming,
            )
            mask[:, h : h + 1] = mask[:, h : h + 1] | str_mask

    return mask


class BlockSparseAttentionGolden(torch.nn.Module):
    def __init__(
        self,
        head_mask_type: Optional[torch.Tensor] = None,
        streaming_info: Optional[torch.Tensor] = None,
        base_blockmask: Optional[torch.Tensor] = None,
        p_dropout: float = 0.0,
        softmax_scale: Optional[float] = None,
        is_causal: bool = False,
        window_size: Tuple[int, int] = (-1, -1),
        m_block_dim: int = 128,
        n_block_dim: int = 128,
        exact_streaming: bool = False,
        upcast: bool = True,
    ):
        super().__init__()
        self.head_mask_type = head_mask_type
        self.streaming_info = streaming_info
        self.base_blockmask = base_blockmask
        self.p_dropout = p_dropout
        self.softmax_scale = softmax_scale
        self.is_causal = is_causal
        self.window_size = window_size
        self.m_block_dim = m_block_dim
        self.n_block_dim = n_block_dim
        self.exact_streaming = exact_streaming
        self.upcast = upcast

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        query_padding_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
        dropout_mask: Optional[torch.Tensor] = None,
        return_attn_probs: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        return block_sparse_attn_golden(
            q=q,
            k=k,
            v=v,
            base_blockmask=self.base_blockmask,
            head_mask_type=self.head_mask_type,
            streaming_info=self.streaming_info,
            query_padding_mask=query_padding_mask,
            key_padding_mask=key_padding_mask,
            p_dropout=self.p_dropout,
            softmax_scale=self.softmax_scale,
            is_causal=self.is_causal,
            window_size=self.window_size,
            m_block_dim=self.m_block_dim,
            n_block_dim=self.n_block_dim,
            exact_streaming=self.exact_streaming,
            dropout_mask=dropout_mask,
            upcast=self.upcast,
            return_attn_probs=return_attn_probs,
        )
