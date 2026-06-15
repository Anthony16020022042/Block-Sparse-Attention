#ifndef FAI_BLOCK_HPP
#define FAI_BLOCK_HPP

#include "catlass/catlass.hpp"
#include "catlass/arch/arch.hpp"
#include "catlass/gemm/dispatch_policy.hpp"

using namespace Catlass;

namespace Catlass::Epilogue {
    enum class LseMode { NONE = 0,
                         OUT_ONLY = 1 };
    // For AtlasA2, FA Infer online Softmax
    template <LseMode LSE_MODE_, typename SM_DTYPE_>
    struct EpilogueAtlasA2OnlineSoftmaxT {
        using ArchTag = Arch::AtlasA2;
        using IntermPrec = SM_DTYPE_;
        static constexpr LseMode LSE_MODE = LSE_MODE_;
    };

    // For AtlasA2, FA Infer RescaleO
    template <LseMode LSE_MODE_, typename SM_DTYPE_>
    struct EpilogueAtlasA2RescaleOT {
        using ArchTag = Arch::AtlasA2;
        using IntermPrec = SM_DTYPE_;
        static constexpr LseMode LSE_MODE = LSE_MODE_;
    };
}

namespace Catlass::Gemm {
    template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
    struct MmadAtlasA2SFAIQK : public MmadAtlasA2 {
        static constexpr uint32_t STAGES = 2;
        static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    };

    template <bool PAGED_CACHE_FLAG_ = false, bool ENABLE_UNIT_FLAG_ = false>
    struct MmadAtlasA2SFAIPV : public MmadAtlasA2 {
        static constexpr uint32_t STAGES = 2;
        static constexpr bool PAGED_CACHE_FLAG = PAGED_CACHE_FLAG_;
        static constexpr bool ENABLE_UNIT_FLAG = ENABLE_UNIT_FLAG_;
    };
}
#endif