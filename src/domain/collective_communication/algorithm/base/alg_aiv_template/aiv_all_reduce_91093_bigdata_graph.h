/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aiv_communication_base.h"

using namespace AscendC;

class AivAllReduceBigGraph91093 : public AivCommBase {
public:
    __aicore__ inline AivAllReduceBigGraph91093() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllReduceBigGraph91093::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t blockNumPerGroup = block_num / rankSize_; 
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    uint64_t avgLengthPerBlock = CeilDiv(len, block_num);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;
    // 使用19个flag
    uint32_t flagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_91093_BIGDATA_GRAPH;

    GM_ADDR flagAddrSelf = GM_OUT[rank_] + flagOffset;
    GM_ADDR flagAddrOther = GM_OUT[dstRank] + flagOffset;

    uint32_t flagSetOffset = rank_ * blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE;
    uint32_t flagCheckOffset = block_idx * FLAG_SIZE; // dstRank * blockNumPerGroup * FLAG_SIZE

    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[dstRank]);

    // 本卡已进入算子，通知其他卡可以搬运，使用第1个flag
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + 2 * blockNumPerGroup * FLAG_SIZE + flagSetOffset), localSetTensor, tag);

    // 确认对端已经将对应的数据拉走
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + 2 * blockNumPerGroup * FLAG_SIZE + flagCheckOffset), localCheckTensor, tag);

    PipeBarrier<PIPE_ALL>();
    
    // ReduceScatter
    if (dstRank != rank_) {
        uint32_t sliceIdx = rank_ * blockNumPerGroup + blockIdxInGroup;
        count = CalActualCount(sliceIdx, sliceCount, avgLengthPerSlice, tailLength);

        uint64_t gmOffset = sliceIdx * avgLengthPerSlice;

        CpGM2GM(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, true, reduceOp_);

        PipeBarrier<PIPE_MTE3>();

        // 本aiv reduce完成，使用第2个flag
        AddSignalValue((__gm__ int32_t*)(flagAddrSelf + blockIdxInGroup * FLAG_SIZE), localSetTensor, tag);
    }
    
    // 全卡同步
    PipeBarrier<PIPE_ALL>();
    if (dstRank == rank_) {
        // check 本端aiv 所有reduce结果是否完成
        WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + blockIdxInGroup * FLAG_SIZE), localCheckTensor, (rankSize_ - 1) * tag);
        PipeBarrier<PIPE_ALL>();
        SetSignalValue((__gm__ int32_t *)(flagAddrSelf + blockIdxInGroup * FLAG_SIZE), localSetTensor, 0);

        SyncFunc<HardEvent::MTE3_S>();

        // 告诉别人自己已经加完所有卡了，使用第3个flag
        SetSignalValue((__gm__ int32_t *)(flagAddrSelf + blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE), localSetTensor, tag);

        SyncFunc<HardEvent::MTE3_MTE2>();
    }

    // 每个aiv读相应对端的flag
    WaitSignalValue((__gm__ int32_t *)(flagAddrOther + blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE), localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();

    // AllGather
    uint32_t sliceIdx = dstRank * blockNumPerGroup + blockIdxInGroup;
    uint64_t gmOffset = sliceIdx * avgLengthPerSlice;
    count = CalActualCount(sliceIdx, sliceCount, avgLengthPerSlice, tailLength);
    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);

    PipeBarrier<PIPE_ALL>();
    // 通知对端，自己已经把对端的那片数据拉回来了
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + 2 * blockNumPerGroup * FLAG_SIZE + block_num * FLAG_SIZE + flagSetOffset), localSetTensor, tag);
    // 确认对端已经将对应的数据拉走
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + 2 * blockNumPerGroup * FLAG_SIZE + block_num * FLAG_SIZE + flagCheckOffset), localCheckTensor, tag);
    return;
}

template<typename T>
__aicore__ inline void aiv_all_reduce_91093_bigdata_graph(KERNEL_ARGS_DEF)
{
    AivAllReduceBigGraph91093 op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
