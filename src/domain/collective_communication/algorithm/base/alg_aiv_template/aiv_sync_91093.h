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
#include "aiv_sync_91093_base.h"

using namespace AscendC;

class AivSync91093 : public AivSync91093Base {
public:
    __aicore__ inline AivSync91093() {}
    __aicore__ inline void Process(GM_ADDR buffOut0, int32_t tag);
};

__aicore__ inline void AivSync91093::Process(GM_ADDR buffOut0, int32_t tag)
{
    if (block_idx >= usedBlockNum_) {
        return;
    }
    
    uint32_t flagOffset = 2 * 1024 * 1024;
    flagOffset += ((tag % AIV_PING_PONG_FACTOR_TWO == 0) ? 0 : rankSize_ * FLAG_SIZE);

    GlobalTensor<uint64_t> bufferArgsGT;
    __gm__ uint64_t *buffersGmAddr = (__gm__ uint64_t *)(buffOut0 + AIV_FLAG_BUFFER_SIZE - COMM_INFO_OFFSET);
    bufferArgsGT.SetGlobalBuffer(buffersGmAddr, FLAG_SIZE * rankSize_ / sizeof(uint64_t));

    // 准备参数，buffer地址
    GM_ADDR buffersOut[MAX_TARGET_NUM] = {};

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank = targetRanks[i];
        DataCopy(bufferArgsTensor[i * 4], bufferArgsGT[2 * targetRank], UB_ADDRESS_PAD_COUNT); // buffersIn buffersOut
    }

    SyncFunc<HardEvent::MTE2_S>();

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t curIdx = i * 4;
        buffersOut[i] = (GM_ADDR)(bufferArgsTensor.GetValue(curIdx + 1));
    }

    PipeBarrier<PIPE_ALL>();

    BatchRecordWait(buffersOut, flagOffset, tag);
}

__aicore__ inline void aiv_sync_91093_inner(KERNEL_ARGS_DEF)
{
    AivSync91093 op;
    op.Init(buffOut0, rank, rankSize);
    op.Process(buffOut0, tag);
}
