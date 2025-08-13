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

class AivAll2AllGraph91093Single : public AivCommBase {
public:
    __aicore__ inline AivAll2AllGraph91093Single() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset,
        uint64_t localRecvOffset, uint64_t remoteSendCount, uint32_t baseFlagOffset);
};

template<typename T>
__aicore__ inline void AivAll2AllGraph91093Single::Process(GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t remoteSendOffset, uint64_t localRecvOffset, uint64_t remoteSendCount, uint32_t baseFlagOffset)
{
    uint32_t blockNumPerGroup = block_num / rankSize_; 
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[dstRank]);

    // 使用96个flag
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + baseFlagOffset;
    GM_ADDR flagAddrOther = GM_OUT[dstRank] + baseFlagOffset;
    uint32_t initAckFlagOffset = 0;
    uint32_t finalAckFlagOffset = block_num * FLAG_SIZE;

    uint32_t flagSetOffset = rank_ * blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE;
    uint32_t flagCheckOffset = block_idx * FLAG_SIZE;

    uint64_t blockRecvCount = 0;
    uint64_t blockRecvOffset = 0;
    CalBlockCountAndOffset(remoteSendCount, blockNumPerGroup, blockIdxInGroup, padCount, blockRecvCount,
        blockRecvOffset);

    // 确认对端已经准备好
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + initAckFlagOffset + flagSetOffset), localSetTensor, tag);
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + initAckFlagOffset + flagCheckOffset), localCheckTensor, tag);
    PipeBarrier<PIPE_ALL>();

    CpGM2GM(outputGM + localRecvOffset + blockRecvOffset, cclGMOther + remoteSendOffset + blockRecvOffset,
        blockRecvCount);
    PipeBarrier<PIPE_ALL>();

    // 确认对端已经读完本端
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + finalAckFlagOffset + flagSetOffset), localSetTensor, tag);
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + finalAckFlagOffset + flagCheckOffset), localCheckTensor, tag);

    return;
}

template<typename T>
__aicore__ inline void aiv_all_to_all_vc_91093_single_graph(KERNEL_ARGS_DEF, ExtraArgs* extraArgs)
{
    AivAll2AllGraph91093Single op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint32_t blockNumPerGroup = block_num / rankSize; 
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_VC_91093_SINGLE_GRAPH;

    uint64_t remoteSendOffset = 0;
    for (uint32_t i = 0; i < rank; i++) {
        remoteSendOffset += extraArgs->sendCountMatrix[dstRank * rankSize + i];
    }
    uint64_t localRecvOffset = 0;
    for (uint32_t i = 0; i < dstRank; i++) {
        localRecvOffset += extraArgs->sendCountMatrix[i * rankSize + rank];
    }
    uint64_t remoteSendCount = extraArgs->sendCountMatrix[dstRank * rankSize + rank];

    op.HeadCounter();
    op.Process<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount, baseFlagOffset);
    op.TailCounter();
}

template<typename T>
__aicore__ inline void aiv_all_to_all_91093_single_graph(KERNEL_ARGS_DEF)
{
    AivAll2AllGraph91093Single op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint32_t blockNumPerGroup = block_num / rankSize; 
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_91093_SINGLE_GRAPH;

    uint64_t remoteSendOffset = rank * len;
    uint64_t localRecvOffset = dstRank * len;
    uint64_t remoteSendCount = len;

    op.HeadCounter();
    op.Process<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount, baseFlagOffset);
    op.TailCounter();
}