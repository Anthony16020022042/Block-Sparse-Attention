#ifndef TILING_DATA_H
#define TILING_DATA_H

struct BlockSparseAttentionTilingData {
    uint32_t batch;                 // batch 数
    uint32_t numHeads;              // Q head 数
    uint32_t kvHeads;               // KV head 数
    uint32_t embeddingSize;         // head dim
    uint32_t blockSize;             // 分页 KV 缓存的块大小
    uint32_t maxNumBlocksPerBatch;  // 每个 batch 最大块数（分页 KV 用）
    uint32_t firstBatchTaskNum;     // 首个 batch 的 task 数，用于 task 索引边界计算
    uint32_t totalTaskNum;          // 所有 batch 的 task 总数
    uint32_t maskType;              // 掩码类型（NO_MASK/CAUSAL/SPARSE 等），当前固定为 0
    float    scaleValue;            // QK^T softmax 缩放因子
    uint32_t totalQBlocks;          // 所有 batch 的 Q 块总数
    uint32_t firstQBlockNum;        // 首个 batch 的 Q 块数
    uint64_t blockShapeX;           // Q 块大小（空间维度，如 128）
    uint64_t blockShapeY;           // KV 块大小（空间维度，如 128）
    uint32_t maxKvBlockNum;         // KV 方向最大块数（所有 batch 取 max）
    uint32_t maxQBlockNum;          // Q 方向最大块数（所有 batch 取 max）
    uint32_t avgRowNumPerSubCore;   // 每个 sub core 平均处理的行数（掩码转换用）
    uint32_t preActivateSubCoreNum; // 需要激活的 sub core 数（掩码转换用）
    uint32_t queryLayout;           // Q 布局：0=TND, 1=BNSD
    uint32_t kvCacheLayout;         // KV 布局：0=TND, 1=BNSD
    uint32_t maxQSeqlen;            // Q 最大序列长度
    uint32_t maxKvSeqlen;           // KV 最大序列长度
    uint32_t useUniformQSeqlen;     // 是否所有 batch 的 Q seqlen 相同
    uint32_t useUniformKvSeqlen;    // 是否所有 batch 的 KV seqlen 相同
    uint64_t selectNumIdxSize;      // selectNumIdx 缓冲区大小（bytes）
    uint64_t selectIdxSize;         // selectIdx 缓冲区大小（bytes）
    uint64_t mm1OutSize;            // S = QK^T 输出缓冲区大小
    uint64_t smOnlineOutSize;       // P = softmax(S) 输出缓冲区大小
    uint64_t mm2OutSize;            // O_tmp = PV 输出缓冲区大小
    uint64_t updateSize;            // O_update 重缩放缓冲区大小
    uint64_t workSpaceSize;         // workspace 总大小

    uint32_t get_batch() const { return batch; }
    uint32_t get_numHeads() const { return numHeads; }
    uint32_t get_kvHeads() const { return kvHeads; }
    uint32_t get_embeddingSize() const { return embeddingSize; }
    uint32_t get_blockSize() const { return blockSize; }
    uint32_t get_maxNumBlocksPerBatch() const { return maxNumBlocksPerBatch; }
    uint32_t get_firstBatchTaskNum() const { return firstBatchTaskNum; }
    uint32_t get_totalTaskNum() const { return totalTaskNum; }
    uint32_t get_maskType() const { return maskType; }
    float get_scaleValue() const { return scaleValue; }
    uint32_t get_totalQBlocks() const { return totalQBlocks; }
    uint32_t get_firstQBlockNum() const { return firstQBlockNum; }
    uint64_t get_blockShapeX() const { return blockShapeX; }
    uint64_t get_blockShapeY() const { return blockShapeY; }
    uint32_t get_maxKvBlockNum() const { return maxKvBlockNum; }
    uint32_t get_maxQBlockNum() const { return maxQBlockNum; }
    uint32_t get_avgRowNumPerSubCore() const { return avgRowNumPerSubCore; }
    uint32_t get_preActivateSubCoreNum() const { return preActivateSubCoreNum; }
    uint32_t get_queryLayout() const { return queryLayout; }
    uint32_t get_kvCacheLayout() const { return kvCacheLayout; }
    uint32_t get_maxQSeqlen() const { return maxQSeqlen; }
    uint32_t get_maxKvSeqlen() const { return maxKvSeqlen; }
    uint32_t get_useUniformQSeqlen() const { return useUniformQSeqlen; }
    uint32_t get_useUniformKvSeqlen() const { return useUniformKvSeqlen; }
    uint32_t get_selectNumIdxSize() const { return selectNumIdxSize; }
    uint32_t get_selectIdxSize() const { return selectIdxSize; }
    uint64_t get_mm1OutSize() const { return mm1OutSize; }
    uint64_t get_smOnlineOutSize() const { return smOnlineOutSize; }
    uint64_t get_mm2OutSize() const { return mm2OutSize; }
    uint64_t get_updateSize() const { return updateSize; }
    uint64_t get_workSpaceSize() const { return workSpaceSize; }

    void set_batch(uint32_t value) { batch = value; }
    void set_numHeads(uint32_t value) { numHeads = value; }
    void set_kvHeads(uint32_t value) { kvHeads = value; }
    void set_embeddingSize(uint32_t value) { embeddingSize = value; }
    void set_blockSize(uint32_t value) { blockSize = value; }
    void set_maxNumBlocksPerBatch(uint32_t value) { maxNumBlocksPerBatch = value; }
    void set_firstBatchTaskNum(uint32_t value) { firstBatchTaskNum = value; }
    void set_totalTaskNum(uint32_t value) { totalTaskNum = value; }
    void set_maskType(uint32_t value) { maskType = value; }
    void set_scaleValue(float value) { scaleValue = value; }
    void set_totalQBlocks(uint32_t value) { totalQBlocks = value; }
    void set_firstQBlockNum(uint32_t value) { firstQBlockNum = value; }
    void set_blockShapeX(uint64_t value) { blockShapeX = value; }
    void set_blockShapeY(uint64_t value) { blockShapeY = value; }
    void set_maxKvBlockNum(uint32_t value) { maxKvBlockNum = value; }
    void set_maxQBlockNum(uint32_t value) { maxQBlockNum = value; }
    void set_avgRowNumPerSubCore(uint32_t value) { avgRowNumPerSubCore = value; }
    void set_preActivateSubCoreNum(uint32_t value) { preActivateSubCoreNum = value; }
    void set_queryLayout(uint32_t value) { queryLayout = value; }
    void set_kvCacheLayout(uint32_t value) { kvCacheLayout = value; }
    void set_maxQSeqlen(uint32_t value) { maxQSeqlen = value; }
    void set_maxKvSeqlen(uint32_t value) { maxKvSeqlen = value; }
    void set_useUniformQSeqlen(uint32_t value) { useUniformQSeqlen = value; }
    void set_useUniformKvSeqlen(uint32_t value) { useUniformKvSeqlen = value; }
    void set_selectNumIdxSize(uint64_t value) { selectNumIdxSize = value; }
    void set_selectIdxSize(uint64_t value) { selectIdxSize = value; }
    void set_mm1OutSize(uint64_t value) { mm1OutSize = value; }
    void set_smOnlineOutSize(uint64_t value) { smOnlineOutSize = value; }
    void set_mm2OutSize(uint64_t value) { mm2OutSize = value; }
    void set_updateSize(uint64_t value) { updateSize = value; }
    void set_workSpaceSize(uint64_t value) { workSpaceSize = value; }
};

#endif