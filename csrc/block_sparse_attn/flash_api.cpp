#include <torch/extension.h>

#include "acl/acl.h"
#include "runtime/rt_ffts.h"
#include "tiling/platform/platform_ascendc.h"
#include "torch_npu/csrc/core/npu/NPUStream.h"
#include "kernel_operator.h"
#include "mha_varlen_fwd_block.cpp"
#include "tiling_data.h"


static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQNBlockTile()
{
    uint32_t qNBlockTile = 1;
    return qNBlockTile;
}

uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    constexpr uint32_t BASIC_BLOCK_SIZE = 128;
    uint32_t qBlocksInX = (x + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE;
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = (remainingSeqlen + BASIC_BLOCK_SIZE - 1) / BASIC_BLOCK_SIZE;
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

void CalculateBatchTaskSplit(int64_t qSeqlen, uint32_t groupSize, uint32_t kvHeads, uint32_t numHeads, int64_t blockShapeX,
                             uint32_t &curTaskNum, uint32_t &curQBlockNum)
{
    uint32_t curQBlockTile = GetQNBlockTile();
    uint32_t qNBlockNumPerGroup = CeilDiv(groupSize, curQBlockTile);
    uint32_t curQNBlockNum = qNBlockNumPerGroup * kvHeads;
    curTaskNum = GetQBlocks(qSeqlen, blockShapeX) * curQNBlockNum;
    curQBlockNum = CeilDiv(qSeqlen, blockShapeX) * numHeads;
}

std::vector<at::Tensor>
mha_varlen_fwd_block(at::Tensor &q,                              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                     const at::Tensor &k,                        // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     const at::Tensor &v,                        // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     const at::Tensor &cu_seqlens_q,             // b+1
                     const at::Tensor &cu_seqlens_k,             // b+1
                     const at::Tensor &head_mask_type,           // (num_heads)
                     std::optional<at::Tensor> &streaming_info_, // (num_heads, 2)
                     std::optional<at::Tensor> &row_blockmask_,  // (batch_size, num_blocksparse_heads, seqlen_m / m_block_dim, seqlen_n / n_block_dim)
                     int max_seqlen_q,
                     const int max_seqlen_k,
                     const float p_dropout,
                     const float softmax_scale,
                     bool is_causal,
                     int window_size_left,
                     int window_size_right,
                     const int m_block_dim,
                     const int n_block_dim,
                     const bool exact_streaming,
                     const bool return_softmax,
                     std::optional<at::Generator> gen_)
{
    const c10::OptionalDeviceGuard device_guard(device_of(q));
    auto aclStream = c10_npu::getCurrentNPUStream().stream(false);
    at::Tensor tiling_cpu_tensor = at::empty({1024}, at::device(c10::kCPU).dtype(at::kByte));
    BlockSparseAttentionTilingData *tiling_cpu_ptr = reinterpret_cast<BlockSparseAttentionTilingData *>(tiling_cpu_tensor.data_ptr<uint8_t>());
    uint32_t blockDim = platform_ascendc::PlatformAscendCManager::GetInstance()->GetCoreNumAic();
    uint64_t libapiSize = platform_ascendc::PlatformAscendCManager::GetInstance()->GetLibApiWorkSpaceSize();

    bool is_bf16 = q.dtype() == torch::kBFloat16;
    bool is_fp16 = q.dtype() == torch::kFloat16;

    // 校验拦截不支持的模式
    TORCH_CHECK(is_bf16 || is_fp16, "NPU BlockSparseAttention only supports Float16 or BFloat16.");
    TORCH_CHECK(p_dropout == 0.0, "NPU BlockSparseAttention does not support dropout.");
    TORCH_CHECK(window_size_left == -1, "NPU BlockSparseAttention does not support window_size_left.");
    TORCH_CHECK(window_size_right == -1, "NPU BlockSparseAttention does not support window_size_right.");
    TORCH_CHECK(k.dtype() == q.dtype(), "query and key must have the same dtype");
    TORCH_CHECK(v.dtype() == q.dtype(), "query and value must have the same dtype");
    TORCH_CHECK(q.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(k.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(v.stride(-1) == 1, "Input tensor must have contiguous last dimension");
    TORCH_CHECK(row_blockmask_.has_value(), "Row block mask is not initialized");

    const auto sizes = q.sizes();
    int T = sizes[0];
    int num_heads = sizes[1];
    const int head_size_og = sizes[2];
    const int batch_size = cu_seqlens_q.numel() - 1;
    auto blockMaskSizes = row_blockmask_.value().sizes();
    int64_t maxQBlockNum = blockMaskSizes[2];
    int64_t maxKvBlockNum = blockMaskSizes[3];

    const int num_heads_k = k.size(1);

    uint32_t totalTaskNum = 0;
    uint32_t totalQBlocks = 0;
    uint32_t firstBatchTaskNum = 0;
    uint32_t firstQBlockNum = 0;
    auto cu_seqlens_q_cpu = cu_seqlens_q.to(at::kCPU); // kernel->host
    const int64_t *qSeqLenList = static_cast<const int64_t *>(cu_seqlens_q_cpu.data_ptr());

    // 遍历每个batch进行分核计算
    for (auto i = 0; i < batch_size; i++) {
        // 根据useUniformQSeqlen_标志位决定使用actualSeqLengths数组还是maxQSeqlen_
        int64_t qSeqlen;
        // 使用actualSeqLengths数组（TND格式或BNSD格式但提供了actualSeqLengths）
        qSeqlen = qSeqLenList[i+1] - qSeqLenList[i];

        uint32_t curTaskNum = 0;
        uint32_t curQBlockNum = 0;
        CalculateBatchTaskSplit(qSeqlen, num_heads/num_heads_k, num_heads_k, num_heads, m_block_dim, curTaskNum, curQBlockNum);

        if (i == 0) {
            firstBatchTaskNum = curTaskNum;
            firstQBlockNum = curQBlockNum;
        }
        totalTaskNum += curTaskNum;
        totalQBlocks += curQBlockNum;
    }
    blockDim = std::min(blockDim, totalTaskNum);

    int64_t selectIdxSize = CeilDiv(m_block_dim, 128) * CeilDiv(maxKvBlockNum, 32) * 32 * sizeof(uint32_t) * batch_size * num_heads * maxQBlockNum;
    int64_t selectNumIdxSize = CeilDiv(m_block_dim, 128) * sizeof(uint32_t) * 32 * batch_size * num_heads * maxQBlockNum;
    int64_t syncSize = sizeof(uint32_t) * 256;

    uint64_t WORKSPACE_BLOCK_SIZE_DB = 131072; // 工作空间块大小
    uint64_t PRELANCH_NUM = 3;

    uint64_t mm1OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
                          sizeof(float) * PRELANCH_NUM;
    uint64_t smOnlineOutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
                               2 * PRELANCH_NUM;
    uint64_t mm2OutSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
                          sizeof(float) * PRELANCH_NUM;
    uint64_t UpdateSize = static_cast<uint64_t>(blockDim) * WORKSPACE_BLOCK_SIZE_DB *
                          sizeof(float) * PRELANCH_NUM;
    int64_t workSpaceSize = libapiSize + mm1OutSize + smOnlineOutSize + mm2OutSize + UpdateSize + selectNumIdxSize + selectIdxSize + syncSize;
    uint32_t totalTaskNumMask = batch_size * num_heads * maxQBlockNum;
    uint32_t avgRowNumPerSubCore = CeilDiv(totalTaskNumMask, blockDim * 2);
    uint32_t preActivateSubCoreNum = CeilDiv(totalTaskNumMask, avgRowNumPerSubCore);


    tiling_cpu_ptr->set_batch(static_cast<uint32_t>(batch_size));           // B
    tiling_cpu_ptr->set_numHeads(static_cast<uint32_t>(num_heads));         // N
    tiling_cpu_ptr->set_kvHeads(static_cast<uint32_t>(num_heads_k));        // S
    tiling_cpu_ptr->set_embeddingSize(static_cast<uint32_t>(head_size_og)); // D
    tiling_cpu_ptr->set_blockSize(128);
    tiling_cpu_ptr->set_maxNumBlocksPerBatch(static_cast<uint32_t>(0)); // 0
    tiling_cpu_ptr->set_firstBatchTaskNum(firstBatchTaskNum);
    tiling_cpu_ptr->set_totalTaskNum(totalTaskNum);
    tiling_cpu_ptr->set_maskType(0);
    tiling_cpu_ptr->set_scaleValue(softmax_scale);
    tiling_cpu_ptr->set_totalQBlocks(totalQBlocks);
    tiling_cpu_ptr->set_firstQBlockNum(firstQBlockNum);
    tiling_cpu_ptr->set_blockShapeX(m_block_dim);
    tiling_cpu_ptr->set_blockShapeY(n_block_dim);
    tiling_cpu_ptr->set_maxKvBlockNum(maxKvBlockNum);
    tiling_cpu_ptr->set_maxQBlockNum(maxQBlockNum);
    tiling_cpu_ptr->set_avgRowNumPerSubCore(avgRowNumPerSubCore);
    tiling_cpu_ptr->set_preActivateSubCoreNum(preActivateSubCoreNum);
    tiling_cpu_ptr->set_queryLayout(0);
    tiling_cpu_ptr->set_kvCacheLayout(0);
    tiling_cpu_ptr->set_maxQSeqlen(max_seqlen_q);
    tiling_cpu_ptr->set_maxKvSeqlen(max_seqlen_k);
    tiling_cpu_ptr->set_useUniformQSeqlen(0);
    tiling_cpu_ptr->set_useUniformKvSeqlen(0);
    tiling_cpu_ptr->set_selectNumIdxSize(selectNumIdxSize);
    tiling_cpu_ptr->set_selectIdxSize(selectIdxSize);
    tiling_cpu_ptr->set_mm1OutSize(mm1OutSize);
    tiling_cpu_ptr->set_smOnlineOutSize(smOnlineOutSize);
    tiling_cpu_ptr->set_mm2OutSize(mm2OutSize);
    tiling_cpu_ptr->set_updateSize(UpdateSize);
    tiling_cpu_ptr->set_workSpaceSize(workSpaceSize);

    at::Tensor workspace_tensor = at::empty({workSpaceSize}, at::device(at::kPrivateUse1).dtype(at::kByte)); // workspace
    at::Tensor softmaxlse = at::empty({T, num_heads}, at::device(at::kPrivateUse1).dtype(at::kFloat));       // lse
    softmaxlse.fill_(std::numeric_limits<float>::infinity());

    at::Tensor out;
    out = torch::zeros_like(q);

    at::Tensor tiling_gpu_tensor = tiling_cpu_tensor.to(at::Device(at::kPrivateUse1)); // Tiling to Device

    uint64_t fftsAddr{0};
    uint32_t fftsLen{0};
    rtError_t error = rtGetC2cCtrlAddr(&fftsAddr, &fftsLen);
    auto qDevice = static_cast<uint8_t *>(const_cast<void *>(q.data_ptr()));
    auto kDevice = static_cast<uint8_t *>(const_cast<void *>(k.data_ptr()));
    auto vDevice = static_cast<uint8_t *>(const_cast<void *>(v.data_ptr()));
    auto blockSparseMaskDevice = static_cast<uint8_t *>(const_cast<void *>(row_blockmask_.value().data_ptr()));
    auto oDevice = static_cast<uint8_t *>(const_cast<void *>(out.data_ptr()));
    auto qSeqDevice = static_cast<uint8_t *>(const_cast<void *>(cu_seqlens_q.data_ptr()));
    auto kvSeqDevice = static_cast<uint8_t *>(const_cast<void *>(cu_seqlens_k.data_ptr()));
    auto workspaceDevice = static_cast<uint8_t *>(const_cast<void *>(workspace_tensor.data_ptr()));
    auto tilingDevice = static_cast<uint8_t *>(const_cast<void *>(tiling_gpu_tensor.data_ptr()));
    auto softmaxLseDevice = static_cast<uint8_t *>(const_cast<void *>(softmaxlse.data_ptr()));

    if (is_bf16) {
        BlockSparse::BlockSparseAttentionInfer<bfloat16_t, float, Epilogue::LseMode::NONE, 0, 0><<<blockDim, nullptr, aclStream>>>(
            fftsAddr, qDevice, kDevice, vDevice, blockSparseMaskDevice, nullptr, nullptr, oDevice,
            qSeqDevice, kvSeqDevice, nullptr, workspaceDevice, softmaxLseDevice, tilingDevice);
    } else {
        BlockSparse::BlockSparseAttentionInfer<half, float, Epilogue::LseMode::NONE, 0, 0><<<blockDim, nullptr, aclStream>>>(
            fftsAddr, qDevice, kDevice, vDevice, blockSparseMaskDevice, nullptr, nullptr, oDevice,
            qSeqDevice, kvSeqDevice, nullptr, workspaceDevice, softmaxLseDevice, tilingDevice);
    }

    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(at::Device(at::kPrivateUse1));
    at::Tensor rng_state = torch::empty({2}, options.dtype(torch::kInt64));
    at::Tensor p = torch::empty({0}, options.dtype(torch::kInt64));
    return {out, softmaxlse, p, rng_state};
}

std::vector<at::Tensor>
mha_varlen_bwd_block(const at::Tensor &dout,           // total_q x num_heads, x head_size
                     const at::Tensor &q,              // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                     const at::Tensor &k,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     const at::Tensor &v,              // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     const at::Tensor &out,            // total_q x num_heads x head_size
                     const at::Tensor &softmax_lse,    // h x total_q, softmax logsumexp
                     std::optional<at::Tensor> &dq_,   // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
                     std::optional<at::Tensor> &dk_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     std::optional<at::Tensor> &dv_,   // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i
                     const at::Tensor &cu_seqlens_q,   // b+1
                     const at::Tensor &cu_seqlens_k,   // b+1
                     const at::Tensor &head_mask_type, // (num_heads)
                     std::optional<at::Tensor> &streaming_info_,
                     std::optional<at::Tensor> &col_blockmask_, // (batch_size, num_blocksparse_heads, seqlen_n / n_block_dim, seqlen_m / m_block_dim)
                     const int max_seqlen_q,
                     const int max_seqlen_k, // max sequence length to choose the kernel
                     const float p_dropout,  // probability to drop
                     const float softmax_scale,
                     const bool zero_tensors,
                     const bool is_causal,
                     int window_size_left,
                     int window_size_right,
                     const int m_block_dim,
                     const int n_block_dim,
                     const bool deterministic,
                     std::optional<at::Generator> gen_,
                     std::optional<at::Tensor> &rng_state)
{
    std::vector<at::Tensor> result;
    return result;
}

PYBIND11_MODULE(block_sparse_attn_C, m)
{
    m.doc() = "BlockSparseAttention";
    m.def("fwd_block", &mha_varlen_fwd_block, "Forward pass, with blockmask");
    m.def("bwd_block", &mha_varlen_bwd_block, "Backward pass, with blockmask");
}
