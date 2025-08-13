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

class AivAll2All91093SinglePingPong : public AivCommBase {
public:
    __aicore__ inline AivAll2All91093SinglePingPong() {}

    template<typename T>
    __aicore__ inline void Process(GM_ADDR input, GM_ADDR output, int32_t tag, uint64_t remoteSendOffset,
        uint64_t localRecvOffset, uint64_t remoteSendCount, uint32_t baseFlagOffset);
};

template<typename T>
__aicore__ inline void AivAll2All91093SinglePingPong::Process(GM_ADDR input, GM_ADDR output, int32_t tag,
    uint64_t remoteSendOffset, uint64_t localRecvOffset, uint64_t remoteSendCount, uint32_t baseFlagOffset)
{
    uint32_t blockNumPerGroup = block_num / rankSize_; 
    uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);

    uint32_t flagOffset = (((tag % 2 == 0) ? 0 : block_num * FLAG_SIZE)) + baseFlagOffset;
    uint32_t dataOffset = (tag % 2 == 0) ? AIV_INIT_OFFSET : AIV_PING_PONG_SIZE;

    // 内存准备
    __gm__ T *inputGM = (__gm__ T *)input;
    __gm__ T *outputGM = (__gm__ T *)output;
    __gm__ T *cclGMSelf = (__gm__ T *)(GM_IN[rank_] + dataOffset);
    __gm__ T *cclGMOther = (__gm__ T *)(GM_IN[dstRank] + dataOffset);

    // 使用96个flag
    GM_ADDR flagAddrSelf = GM_OUT[rank_] + flagOffset;
    GM_ADDR flagAddrOther = GM_OUT[dstRank] + flagOffset;

    uint32_t flagSetOffset = rank_ * blockNumPerGroup * FLAG_SIZE + blockIdxInGroup * FLAG_SIZE;
    uint32_t flagCheckOffset = block_idx * FLAG_SIZE;

    uint64_t blockRecvCount = 0;
    uint64_t blockRecvOffset = 0;
    CalBlockCountAndOffset(remoteSendCount, blockNumPerGroup, blockIdxInGroup, padCount, blockRecvCount,
        blockRecvOffset);

    // localcopy
    CpGM2GM(cclGMSelf + localRecvOffset + blockRecvOffset, inputGM + localRecvOffset + blockRecvOffset,
        blockRecvCount);

    PipeBarrier<PIPE_ALL>();

    // 卡间同步，确认对端已经准备好
    SetSignalValue((__gm__ int32_t *)(flagAddrOther + flagSetOffset), localSetTensor, tag);
    WaitSignalValue((__gm__ int32_t *)(flagAddrSelf + flagCheckOffset), localCheckTensor, tag);

    PipeBarrier<PIPE_ALL>();

    CpGM2GM(outputGM + localRecvOffset + blockRecvOffset, cclGMOther + remoteSendOffset + blockRecvOffset,
        blockRecvCount);

    return;
}

template<typename T>
__aicore__ inline void aiv_all_to_all_91093_single_pingpong(KERNEL_ARGS_DEF)
{
    AivAll2All91093SinglePingPong op;
    op.Init(KERNEL_CLASS_INIT, true);

    uint32_t blockNumPerGroup = block_num / rankSize; 
    uint32_t dstRank = block_idx / blockNumPerGroup;
    uint32_t baseFlagOffset = BASE_FLAG_OFFSET * AIV_ALL_TO_ALL_91093_SINGLE_PINGPONG;

    uint64_t remoteSendOffset = rank * len;
    uint64_t localRecvOffset = dstRank * len;
    uint64_t remoteSendCount = len;

    op.HeadCounter();
    op.Process<T>(input, output, tag, remoteSendOffset, localRecvOffset, remoteSendCount, baseFlagOffset);
    op.TailCounter();
}