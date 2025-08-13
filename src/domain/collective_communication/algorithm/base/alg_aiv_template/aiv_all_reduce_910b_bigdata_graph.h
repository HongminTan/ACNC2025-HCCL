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

class AivAllReduceBigGraph910B : public AivCommBase {
public:
    __aicore__ inline AivAllReduceBigGraph910B() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag);
};

template<typename T>
__aicore__ inline void AivAllReduceBigGraph910B::Process(GM_ADDR input, GM_ADDR output, uint64_t len, int32_t tag)
{
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
    uint64_t avgLengthPerRank = CeilDiv(len, rankSize_);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerRank, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice;

    uint64_t count = 0;
    // 使用19个flag
    uint32_t flagOffset = BASE_FLAG_OFFSET * AIV_ALL_REDUCE_910B_BIGDATA_GRAPH;

    GM_ADDR flagAddrSelf = GM_OUT[rank_] + flagOffset;
    GM_ADDR flagAddrOther = GM_OUT[block_idx] + flagOffset;

    __gm__ T *outputGm = (__gm__ T *)output;
    __gm__ T *cclGmSelf = (__gm__ T *)(GM_IN[rank_]);
    __gm__ T *cclGmOther = (__gm__ T *)(GM_IN[block_idx]);

    // 本卡已进入算子，通知其他卡可以搬运，使用第1个flag
    SetSignalValue((__gm__ int32_t *)(flagAddrSelf + 3 * FLAG_SIZE + block_idx * FLAG_SIZE * 2), localSetTensor, tag);

    // 确认对端已经将对应的数据拉走
    WaitSignalValue((__gm__ int32_t *)(flagAddrOther + 3 * FLAG_SIZE + rank_ * FLAG_SIZE * 2), localCheckTensor, tag);

    PipeBarrier<PIPE_ALL>();

    // ReduceScatter
    if (block_idx != rank_) {
        count = CalActualCount(rank_, sliceCount, avgLengthPerSlice, tailLength);

        uint64_t gmOffset = rank_ * avgLengthPerSlice;

        CpGM2GM(cclGmSelf + gmOffset, cclGmOther + gmOffset, count, true, reduceOp_);

        PipeBarrier<PIPE_MTE3>();

        // 本aiv reduce完成，使用第2个flag
        AddSignalValue((__gm__ int32_t*)(flagAddrSelf + FLAG_SIZE), localSetTensor, tag);
    }

    // 全卡同步
    PipeBarrier<PIPE_ALL>();
    if (block_idx == rank_) {
        // check 本端aiv 所有reduce结果是否完成
        WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + FLAG_SIZE), localCheckTensor, (block_num - 1) * tag);
        PipeBarrier<PIPE_ALL>();
        SetSignalValue((__gm__ int32_t *)(flagAddrSelf + FLAG_SIZE), localSetTensor, 0);

        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID2);

        // 告诉别人自己已经加完所有卡了，使用第3个flag
        SetSignalValue((__gm__ int32_t *)(flagAddrSelf + 2 * FLAG_SIZE), localSetTensor, rankSize_ * tag);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
    }

    // 每个aiv读相应对端的flag
    WaitSignalGEValue((__gm__ int32_t *)(flagAddrOther + 2 * FLAG_SIZE), localCheckGETensor, tag);
    PipeBarrier<PIPE_ALL>();
    AddSignalValue((__gm__ int32_t *)(flagAddrOther + 2 * FLAG_SIZE), localSetTensor, -tag);

    PipeBarrier<PIPE_ALL>();

    // AllGather
    uint64_t gmOffset = block_idx * avgLengthPerSlice;
    count = CalActualCount(block_idx, sliceCount, avgLengthPerSlice, tailLength);
    CpGM2GM(outputGm + gmOffset, cclGmOther + gmOffset, count);

    PipeBarrier<PIPE_ALL>();
    // 通知对端，自己已经把对端的那片数据拉回来了
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + 3 * FLAG_SIZE + rank_ * FLAG_SIZE * 2 + FLAG_SIZE), localSetTensor, tag);
    // 确认对端已经将对应的数据拉走
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + 3 * FLAG_SIZE + block_idx * FLAG_SIZE * 2 + FLAG_SIZE), localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();
    SetSignalValue((__gm__ int32_t *)(flagAddrSelf + 3 * FLAG_SIZE + block_idx * FLAG_INTERVAL), localSetTensor, 0);
    SetSignalValue((__gm__ int32_t *)(flagAddrSelf + 3 * FLAG_SIZE + block_idx * FLAG_INTERVAL + FLAG_SIZE), localSetTensor, 0);
    return;
}

template<typename T>
__aicore__ inline void aiv_all_reduce_910b_bigdata_graph(KERNEL_ARGS_DEF)
{
    AivAllReduceBigGraph910B op;
    op.Init(KERNEL_CLASS_INIT, true);
    op.HeadCounter();
    op.Process<T>(input, output, len, tag);
    op.TailCounter();
}
