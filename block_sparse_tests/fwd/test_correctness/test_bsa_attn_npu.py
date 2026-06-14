import math
import pytest
import torch
import torch_npu
import numpy as np
import ctypes
import random
from ml_dtypes import bfloat16
from einops import rearrange, repeat
from typing import Optional, Tuple
from block_sparse_attn import block_sparse_attn_func

class TestBlockSparseAttentionTorch():
    @classmethod
    def online_softmax_attention_torch_high(cls, q_block, kv_blocks, scale):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device('cpu')
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]
        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full((1, q_len, 1), -float('inf'), dtype=torch.float32, device=device)  # running max
        l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)  # running sum
        O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)  # running output
        is_first = 1
        # 逐块处理

        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            S_i = torch.matmul(q_block.to(torch.float32), k_block.to(torch.float32))  # (1, q_len, k_len)

            S_i = S_i * scale

            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)
            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new) # (1, q_len, 1)
            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new)  # (1, q_len, k_len)

            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 7. 更新 running output
            # O_new = alpha * O_old + P_i @ V_i
            O_i = alpha * O_i + torch.matmul(P_i.to(torch.float32), v_block.to(torch.float32))  # (1, q_len, head_size)

            # 8. 更新 running max
            m_i = m_new

        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)
        lse = (m_i + torch.log(l_i)).to(torch.float32)
        return O_final, lse

    @classmethod
    def online_softmax_attention_torch(cls, q_block, kv_blocks, scale, torch_dtype, input_dtype, inner_precise):
        # 确保在 CPU 上运行
        q_block = q_block.cpu()
        device = torch.device('cpu')
        q_len = q_block.shape[1]
        head_size = q_block.shape[2]
        # 初始化状态量（确保在 CPU 上）
        m_i = torch.full((1, q_len, 1), -float('inf'), dtype=torch.float32, device=device)  # running max
        l_i = torch.zeros((1, q_len, 1), dtype=torch.float32, device=device)  # running sum
        O_i = torch.zeros((1, q_len, head_size), dtype=torch.float32, device=device)  # running output
        is_first = 1
        # 逐块处理
        for k_block, v_block in kv_blocks:
            # 确保 kv blocks 在 CPU 上
            k_block = k_block.cpu()
            v_block = v_block.cpu()
            # 1. 计算注意力分数 S_i = Q @ K_i^T * scale
            # CPU不支持half精度matmul，需要转换为float32计算
            S_i = torch.matmul(q_block.float(), k_block.float()).to(torch_dtype)  # (1, q_len, k_len)
            S_i = S_i * scale

            # 2. 计算当前块的最大值
            m_block, _ = torch.max(S_i, dim=-1, keepdim=True)  # (1, q_len, 1)

            # 3. 计算新的全局最大值
            m_new = torch.maximum(m_i, m_block)  # (1, q_len, 1)
            # 4. 计算修正因子
            # alpha: 旧输出的修正系数 = exp(m_old - m_new)
            # beta: 当前块的修正系数（用于 softmax）= exp(m_block - m_new)
            alpha = torch.exp(m_i - m_new).to(torch_dtype)  # (1, q_len, 1)

            # 5. 计算当前块的稳定 softmax 分子
            P_i = torch.exp(S_i - m_new).to(torch_dtype)  # (1, q_len, k_len)

            # 6. 更新 running sum
            l_i = alpha * l_i + torch.sum(P_i, dim=-1, keepdim=True)  # (1, q_len, 1)
            # 7. 更新 running output
            # CPU不支持half精度matmul，需要转换为float32计算
            O_i = alpha * O_i + torch.matmul(P_i.float(), v_block.float()).to(torch_dtype)  # (1, q_len, head_size)
            # 8. 更新 running max
            m_i = m_new

        # 最终归一化
        O_final = O_i / l_i  # (1, q_len, head_size)
        # LSE = m + log(l)，shape: (1, q_len, 1)
        lse = (m_i + torch.log(l_i)).to(torch.float32)

        return O_final, lse

    def ref_select_idx_attention_torch(self,
                                       query,
                                       key,
                                       value,
                                       scale: float,
                                       select_idx_list: list,
                                       select_num_idx_list: list,
                                       s_block_x: int,
                                       s_block_y: int,
                                       total_q_blocks: int,
                                       max_kv_block_num: int,
                                       q_seqlen_list: list,
                                       kv_seqlen_list: list,
                                       batch: int,
                                       torch_dtype,
                                       inner_precise
                                       ):
        """
        PyTorch 版本的 ref_select_idx_attention
        确保所有计算在 CPU 上进行
        """
        # 确保输入在 CPU 上
        if query.device.type != 'cpu':
            query = query.cpu()
        if key.device.type != 'cpu':
            key = key.cpu()
        if value.device.type != 'cpu':
            value = value.cpu()

        device = torch.device('cpu')

        # 转置操作
        query = query.permute(1, 0, 2)  # (total_q_tokens, num_heads, head_size) -> (num_heads, total_q_tokens, head_size)
        key = key.permute(1, 2, 0)      # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, head_size, total_kv_tokens)
        value = value.permute(1, 0, 2)   # (total_kv_tokens, kv_heads, head_size) -> (kv_heads, total_kv_tokens, head_size)
        num_heads = query.shape[0]
        kv_heads = key.shape[0]

        # 初始化输出 - 注意这里应该是total_q_tokens而不是max_q_seqlen
        total_q_tokens = query.shape[1]
        head_size = query.shape[2]
        out_high = torch.zeros((num_heads, total_q_tokens, head_size), dtype=torch.float32, device=device)
        out = torch.zeros((num_heads, total_q_tokens, head_size), dtype=query.dtype, device=device)
        lse_out = torch.zeros((num_heads, total_q_tokens, 1), dtype=torch.float32, device=device)

        # 【关键修复】：添加batch级别的累计偏移量
        q_token_offset = 0   # Q方向token累计偏移
        kv_token_offset = 0  # KV方向token累计偏移
        q_block_offset = 0   # Q块累计偏移（用于selectIdx索引）

        for batch_idx in range(batch):
            q_seqlen = q_seqlen_list[batch_idx]
            kv_seqlen = kv_seqlen_list[batch_idx]

            # 计算当前batch的分块数量
            s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
            s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y

            for t_local in range(s_block_num_q):
                t_global = q_block_offset + t_local
                q_block_idx = t_local  # 当前batch内的Q块索引

                # 【关键修复】：batch内的相对位置
                q_start_local = q_block_idx * s_block_x
                q_end_local = min((q_block_idx + 1) * s_block_x, q_seqlen)

                # 【关键修复】：加上batch偏移得到全局位置
                q_start_global = q_token_offset + q_start_local
                q_end_global = q_token_offset + q_end_local

                for head in range(num_heads):
                    # 获取该头对应的selectIdx
                    select_idx_offset = t_global * num_heads * max_kv_block_num + head * max_kv_block_num
                    select_num_offset = t_global * num_heads + head

                    select_num = select_num_idx_list[select_num_offset]
                    selected_kv_blocks = select_idx_list[select_idx_offset:select_idx_offset + max_kv_block_num]

                    # 【GQA修复】：在循环之前提取 q_block 和计算 GQA 参数（只需要一次）
                    q_block = query[head:head+1, q_start_global:q_end_global, :]  # (1, q_block_size, head_size)

                    # 处理 group attention 的情况
                    group_size = num_heads // kv_heads
                    kv_head_idx = head // group_size

                    # 收集所有选中的KV块数据（作为 (K, V) 元组列表）
                    kv_blocks = []
                    k_blocks = []
                    v_blocks = []
                    if select_num == 0:
                        continue

                    for kv_block_idx in selected_kv_blocks[:select_num]:
                        if kv_block_idx == -1:  # 跳过填充的-1
                            continue

                        # 【关键修复】：batch内的相对位置
                        k_start_local = kv_block_idx * s_block_y
                        k_end_local = min((kv_block_idx + 1) * s_block_y, kv_seqlen)

                        # 【关键修复】：加上batch偏移得到全局位置
                        k_start_global = kv_token_offset + k_start_local
                        k_end_global = kv_token_offset + k_end_local

                        # 【修复】：使用全局位置访问key和value
                        k_block = key[kv_head_idx:kv_head_idx+1, :, k_start_global:k_end_global]    # (1, head_size, k_block_size)
                        v_block = value[kv_head_idx:kv_head_idx+1, k_start_global:k_end_global, :]  # (1, k_block_size, head_size)

                        k_blocks.append(k_block)
                        v_blocks.append(v_block)

                    k_block_com = torch.cat(k_blocks, dim=2)
                    v_block_com = torch.cat(v_blocks, dim=1)

                    k_total_len = k_block_com.shape[2]
                    v_total_len = v_block_com.shape[1]

                    if k_total_len <= 512:
                        kv_blocks.append((k_block_com, v_block_com))
                    else:
                        num_chunks = k_total_len // 512
                        remainder = k_total_len % 512
                        for i in range(num_chunks):
                            start = i * 512
                            end = start + 512
                            k_chunk = k_block_com[ : , : , start : end]
                            v_chunk = v_block_com[ : , start : end , : ]
                            kv_blocks.append((k_chunk, v_chunk))
                        if remainder > 0:
                            k_last_chunk = k_block_com[ : , : , -remainder : ]
                            v_last_chunk = v_block_com[ : , -remainder : , : ]
                            kv_blocks.append((k_last_chunk, v_last_chunk))

                    # 使用 Online Softmax 计算注意力（FlashAttention 风格）
                    if inner_precise == 0:
                        out_block, lse_block = self.online_softmax_attention_torch_high(q_block, kv_blocks, scale)  # (1, q_block_size, head_size)
                    else:
                        out_block, lse_block = self.online_softmax_attention_torch(q_block, kv_blocks, scale, torch_dtype, query.dtype, inner_precise)
                    # 【修复】：输出到全局位置
                    # out_high[head:head+1, q_start_global:q_end_global, :] = out_block_high
                    out[head:head+1, q_start_global:q_end_global, :] = out_block
                    lse_out[head:head+1, q_start_global:q_end_global, :] = lse_block.to(torch.float32)
            # 【关键修复】：更新累计偏移量，为下一个batch做准备
            q_token_offset += q_seqlen
            kv_token_offset += kv_seqlen
            q_block_offset += s_block_num_q
        # 转置回原始格式
        # out_high = out_high.permute(1, 0, 2)  # (num_heads, total_q_tokens, head_size) -> (total_q_tokens, num_heads, head_size)
        out = out.permute(1, 0, 2)
        lse_out = lse_out.permute(1, 0, 2)
        return out, lse_out

    @classmethod
    def change_bnsd_to_tnd(self, tensor, seqlenList):
        """
        把BNSD格式的tensor转换成TND格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        tokenNum = sum(seqlenList)
        batch = len(seqlenList)
        res = torch.zeros((tokenNum, headNum, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[count:count + seqlenList[i], :, :] = tensor[i,:,:seqlenList[i], :].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res

    @classmethod
    def change_tnd_to_bnsd(self, tensor, seqlenList, maxQSeqlen):
        """
        把TND格式的tensor转换成BNSD格式
        """
        headDim = tensor.shape[-1]
        headNum = tensor.shape[1]
        batch = len(seqlenList)
        res = torch.zeros((batch, headNum, maxQSeqlen, headDim), dtype=tensor.dtype)
        count = 0
        for i in range(batch):
            res[i, :, :seqlenList[i], :] = tensor[count:count+seqlenList[i], :, :].permute(1, 0, 2)
            count = count + seqlenList[i]
        return res

    def calc_data(self, query_dtype, query, key, value, select_idx, select_num_idx, block_shape, q_seqlen_list, kv_seqlen_list, scale_value, q_input_layout, kv_input_layout, inner_precise):
        """
        PyTorch 版本的 calc_data
        确保所有计算在 CPU 上进行
        """
        # 确保输入是 torch.Tensor，如果是 numpy 数组则转换
        if not isinstance(query, torch.Tensor):
            query = safe_to_tensor(query).cpu()
        if not isinstance(key, torch.Tensor):
            key = safe_to_tensor(key).cpu()
        if not isinstance(value, torch.Tensor):
            value = safe_to_tensor(value).cpu()
        if not isinstance(select_idx, torch.Tensor):
            select_idx = safe_to_tensor(select_idx).cpu()
        if not isinstance(select_num_idx, torch.Tensor):
            select_num_idx = safe_to_tensor(select_num_idx).cpu()

        # 确保在 CPU 上
        query = query.cpu()
        key = key.cpu()
        value = value.cpu()
        select_idx = select_idx.cpu()
        select_num_idx = select_num_idx.cpu()

        # return query, query  # todo 测试性能使用，后续删除
        maxQSeqlen = 0
        if q_input_layout == "BNSD":
            maxQSeqlen = query.shape[-2]
            query = self.change_bnsd_to_tnd(query, q_seqlen_list)
            key = self.change_bnsd_to_tnd(key, kv_seqlen_list)
            value = self.change_bnsd_to_tnd(value, kv_seqlen_list)
        embedding_size = query.shape[2]
        num_heads = query.shape[1]
        batch_size = len(q_seqlen_list)

        if inner_precise == 1 :
            scale_value = np.float16(scale_value)

        # if q_input_layout == 'TND' and kv_input_layout == 'TND':
        # 使用 torch 计算总和
        if isinstance(q_seqlen_list, (list, tuple)):
            num_tokens = sum(q_seqlen_list)
        else:
            num_tokens = q_seqlen_list.clone().detach().sum().item()
        head_size_vo = embedding_size

        shape_out = (num_tokens, num_heads, head_size_vo)
        # 根据 query_dtype 确定 torch dtype
        if isinstance(query_dtype, torch.dtype):
            torch_dtype = query_dtype
        elif query_dtype == np.float32 or str(query_dtype) == 'float32':
            torch_dtype = torch.float32
        elif query_dtype == np.float16 or str(query_dtype) == 'float16':
            torch_dtype = torch.float16
        elif query_dtype == np.bfloat16 or str(query_dtype) == 'bfloat16':
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        input_type = torch_dtype
        if inner_precise == 1 :
            torch_dtype = torch.float16

        ref_output = torch.zeros(shape_out, dtype=torch_dtype, device=torch.device('cpu'))
        ref_output_high = torch.zeros(shape_out, dtype=torch.float32, device=torch.device('cpu'))

        total_q_blocks = select_idx.shape[0]
        max_kv_block_num = select_idx.shape[2]
        s_block_x = block_shape[0]
        s_block_y = block_shape[1]
        select_idx_list = select_idx.flatten().tolist()
        select_num_idx_list = select_num_idx.flatten().tolist()
        ref_output, ref_lse = self.ref_select_idx_attention_torch(
            query, key, value, scale_value,
            select_idx_list, select_num_idx_list,
            s_block_x, s_block_y,
            total_q_blocks, max_kv_block_num,
            q_seqlen_list, kv_seqlen_list, batch_size, torch_dtype, inner_precise
        )
        if query.dtype != torch.float32:
            ref_output_h = ref_output.to(torch.float32)
            return ref_output_h, ref_lse
        else:
            return ref_output, ref_lse


def change_block_sparsemask_to_selectidx_selctnumidx(
    block_sparse_mask, q_seqlen_list, kv_seqlen_list, block_shape, batch
):
    """
    把blocksparseMask转为selectIdx和selectNumIdx
    其中blockSparseMask为[b, headNum, maxQBlockNum, maxKvBlockNum]
    selectIdx为[QBlockNum, headNum, maxKvBlockNum]维度，其中第一维表示每个batch块的合轴
    selectNumIdx为[QBlockNum, headNum]维度表示每个Q方向块有效的KV块数
    """

    s_block_x = block_shape[0]
    s_block_y = block_shape[1]

    # blockSparseMask shape: [b, headNum, maxQBlockNum, maxKvBlockNum]
    num_heads = block_sparse_mask.shape[1]
    max_q_block_num = block_sparse_mask.shape[2]
    max_kv_block_num = block_sparse_mask.shape[3]

    # 计算总的Q块数
    total_q_blocks = 0
    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        total_q_blocks += s_block_num_q

    # 初始化selectIdx和selectNumIdx列表
    select_idx_list = []
    select_num_idx_list = []

    # 遍历每个batch
    q_block_offset = 0
    for b in range(batch):
        q_seqlen = q_seqlen_list[b]
        kv_seqlen = kv_seqlen_list[b]

        # 计算当前batch的分块数量
        s_block_num_q = (q_seqlen + s_block_x - 1) // s_block_x
        s_block_num_kv = (kv_seqlen + s_block_y - 1) // s_block_y

        # 遍历当前batch的每个Q块
        for q_block_idx in range(s_block_num_q):
            # 遍历每个head
            for head in range(num_heads):
                # 从blockSparseMask中提取当前位置的mask: [maxKvBlockNum]
                mask_row = block_sparse_mask[b, head, q_block_idx, :]

                # 找出所有非零的KV块索引（稀疏mask中非零表示该KV块被选中）
                selected_kv_blocks = []
                for kv_block_idx in range(s_block_num_kv):
                    if kv_block_idx < max_kv_block_num and mask_row[kv_block_idx] != 0:
                        selected_kv_blocks.append(kv_block_idx)

                # 填充到max_kv_block_num长度，不足的用-1填充
                padded_blocks = selected_kv_blocks + [-1] * (
                    max_kv_block_num - len(selected_kv_blocks)
                )

                # 添加到selectIdx
                select_idx_list.extend(padded_blocks)

            # 为当前Q块的每个head记录选中的KV块数量
            for head in range(num_heads):
                mask_row = block_sparse_mask[b, head, q_block_idx, :]
                num_selected = 0
                for kv_block_idx in range(s_block_num_kv):
                    if kv_block_idx < max_kv_block_num and mask_row[kv_block_idx] != 0:
                        num_selected += 1
                select_num_idx_list.append(num_selected)

        q_block_offset += s_block_num_q
    # 将列表转换为tensor并reshape为正确的shape
    # selectIdx: [QBlockNum, headNum, maxKvBlockNum]
    # selectNumIdx: [QBlockNum, headNum]
    select_idx_tensor = torch.tensor(select_idx_list, dtype=torch.int32).view(
        total_q_blocks, num_heads, max_kv_block_num
    )
    select_num_idx_tensor = torch.tensor(select_num_idx_list, dtype=torch.int32).view(
        total_q_blocks, num_heads
    )

    # print("=" * 20, f"selectIdx shape: {select_idx_tensor.shape}, selectNumIdx shape: {select_num_idx_tensor.shape}")
    return select_idx_tensor, select_num_idx_tensor


def generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, round_base, m_block_dim, n_block_dim, batch_size, num_blocksparse_heads, sparsity_list, causal=False):
    assert len(sparsity_list) == num_blocksparse_heads
    def round_to_multiple(x, base):
        return ((x + base - 1) // base) * base
    
    nrow, ncol = round_to_multiple(max_seqlen_q, round_base) // m_block_dim, round_to_multiple(max_seqlen_k, round_base) // n_block_dim
    base_mask = torch.zeros(batch_size, num_blocksparse_heads, nrow, ncol, dtype=torch.bool).npu()
    
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
    (torch.bfloat16, 1, 1, 1, 512, 1024, 128, True),
    # (torch.bfloat16, 2, 4, 4, 1024, 1024, 128, False),
    # (torch.float16, 7, 5, 1, 512, 512, 128, True),
    # (torch.float16, 7, 5, 1, 777, 888, 128, False),
    # (torch.float16, 7, 5, 1, 1777, 1888, 128, True),
    # (torch.bfloat16, 1, 1, 1, 7777, 8192, 64, True),
    # (torch.bfloat16, 7, 5, 1, 711, 8192, 64, True)
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
    torch.npu.set_device(1)
    q_min_range = -5.0
    q_max_range = 5.0
    kv_min_range = -5.0
    kv_max_range = 5.0
    query = (q_min_range + (q_max_range - q_min_range) * torch.rand(batch_size * q_seqlen, num_heads, head_size)).to(data_type).npu()
    key = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    value = (kv_min_range + (kv_max_range - kv_min_range) * torch.rand(batch_size * kv_seqlen, kv_heads, head_size)).to(data_type).npu()
    actual_seq_len = torch.full((batch_size,), q_seqlen, dtype=torch.int64).npu()
    actual_kv_len = torch.full((batch_size,), kv_seqlen, dtype=torch.int64).npu()

    # print_tensor_full("query", query)
    # print_tensor_full("key", key)
    # print_tensor_full("value", value)
    # print_tensor_full("actual_seq_len", actual_seq_len)
    # print_tensor_full("actual_kv_len", actual_kv_len)

    max_seqlen_q = q_seqlen
    max_seqlen_k = kv_seqlen
    dropout_p = 0.0
    scale = 1.0 / (head_size ** 0.5)
    window_size_left = -1
    window_size_right = -1
    return_attn_probs = False
    block_table = None
    head_mask_type = torch.tensor([1] * num_heads, dtype=torch.int32).npu()
    streaming_info = None

    sparsity = 0.5
    sparsity_list = [sparsity] * num_heads
    block_size = 128
    base_blockmask = generate_base_sparsity_mask(max_seqlen_q, max_seqlen_k, block_size, block_size, block_size, batch_size, num_heads, sparsity_list)
    print("mask shape:", base_blockmask.shape)
    print("mask dtype:", base_blockmask.dtype)
    print("device:", base_blockmask.device)
    # 打印全部数值（小块掩码可用，大尺寸会刷屏）
    print(base_blockmask)
    # print("[wjc] start")
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
    # print("[wjc] end")
    # # ==========================================
    # # 🔥 万能打印：自动识别类型、长度、内容、shape
    # # ==========================================
    # print("\n" + "="*50)
    # print("📌 函数返回结果类型:", type(result))
    # print("📌 长度/元素个数:", len(result) if isinstance(result, (list, tuple)) else "不是列表")

    # # 逐个打印每个返回值
    # for idx, item in enumerate(result):
    #     print(f"\n返回值 [{idx}] 类型: {type(item)}")
    #     if hasattr(item, 'shape'):
    #         print(f"           shape: {item.shape}")
    #     if hasattr(item, 'dtype'):
    #         print(f"           dtype: {item.dtype}")
    #     print(f"           内容: {item}")

    # print("="*50 + "\n")

    q_input_value = query.cpu()
    k_input_value = key.cpu()
    v_input_value = value.cpu()
    q_seqlen_list = actual_seq_len.cpu()
    kv_seqlen_list = actual_kv_len.cpu()
    block_shape = [128, 128]
    # 从blockSparseMask转换为selectIdx和selectNumIdx
    select_idx_input, select_num_idx_input = change_block_sparsemask_to_selectidx_selctnumidx(
        base_blockmask, q_seqlen_list, kv_seqlen_list, block_shape, batch_size
    )

    testObj = TestBlockSparseAttentionTorch()
    atten_out_golden, lse_golden = testObj.calc_data(data_type, q_input_value, k_input_value, v_input_value, select_idx_input, select_num_idx_input, block_shape, q_seqlen_list, kv_seqlen_list, scale, "TND", "TND", 0)

    atten_out_npu = result.cpu()
    diff = (atten_out_npu.float() - atten_out_golden.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    cos_sim = torch.nn.functional.cosine_similarity(
        atten_out_npu.float().flatten().unsqueeze(0),
        atten_out_golden.float().flatten().unsqueeze(0),
    ).item()

    # print(f"\n===== NPU vs Golden 对比 =====")
    # print(f"Max diff : {max_diff:.6e}")
    # print(f"Mean diff: {mean_diff:.6e}")
    # print(f"Cosine similarity: {cos_sim:.8f}")
    # if max_diff > 1e-2:
    #     print("⚠️  差异较大, 请检查！")
    # else:
    #     print("✅ 结果在合理误差范围内")
    # print("="*50)

    torch.testing.assert_close(
        atten_out_npu.float(),
        atten_out_golden.float(),
        rtol=1e-2,  # 相对误差
        atol=1e-2,  # 绝对误差
    )
