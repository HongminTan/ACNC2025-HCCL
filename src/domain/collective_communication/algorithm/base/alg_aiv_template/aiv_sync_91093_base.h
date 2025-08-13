/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_SYNC_91093_BASE_H
#define AIV_SYNC_91093_BASE_H

#include "aiv_communication_base.h"

using namespace AscendC;

class AivSync91093Base {
public:
    __aicore__ inline AivSync91093Base() {}

    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize);

    template<HardEvent event> 
    __aicore__ inline void SyncFunc();

    __aicore__ inline void BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag);

protected:
    uint32_t baseFlagOffset_ = 0;
    GM_ADDR flagAddrSelf_;
    uint32_t rank_;
    uint32_t rankSize_;
    uint32_t usedBlockNum_;

    TPipe pipe;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localClearTensor;
    TBuf<> bufferArgsBuf;
    LocalTensor<uint64_t> bufferArgsTensor; // buffer地址GM-UB

    uint32_t numTargets; // 每个aiv需要顺序与几个对端通信，ranksize太大时，aiv不够用，需要多次
    uint32_t targetRanks[MAX_TARGET_NUM] = {}; // 最多768/48 = 16 次（一次代表服务48张卡）
};

__aicore__ inline void AivSync91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize)
{
    flagAddrSelf_ = buffOut0;

    rank_ = rank;
    rankSize_ = rankSize;

    pipe.InitBuffer(localFlagBuf, UB_FLAG_SIZE * FLAG_BUF_NUM);
    localSetTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, 0);
    localCheckTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE);
    localClearTensor = localFlagBuf.GetWithOffset<int32_t>(UB_FLAG_PAD_COUNT, UB_FLAG_SIZE * IDX_2);
    localClearTensor.SetValue(0, 0);
    pipe.InitBuffer(bufferArgsBuf, UB_FLAG_SIZE * MAX_TARGET_NUM);
    bufferArgsTensor = bufferArgsBuf.Get<uint64_t>();

    pipe.InitBuffer(inOutQue, DOUBLE, UB_DB_DATA_BATCH_SIZE);

    // 计算本core的numTargets和targetsList
    // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
    usedBlockNum_ = block_num - block_num % 2;
    uint32_t halfConcurrent = usedBlockNum_ / 2; // block_num需要为偶数
    numTargets = (rankSize_ - 1) / usedBlockNum_; // 除去本rank，可能需要补上一个
    uint32_t tailRankSize = (rankSize_ - 1) % usedBlockNum_;
    uint32_t leftTailRankSize = 0;
    uint32_t rightTailRankSize = 0;
    if (tailRankSize > 0) {
        if (tailRankSize <= halfConcurrent) {
            leftTailRankSize = tailRankSize;
        } else {
            leftTailRankSize = halfConcurrent;
            rightTailRankSize = tailRankSize - halfConcurrent;
        }
        if (block_idx < halfConcurrent && (halfConcurrent - block_idx) <= leftTailRankSize) {
            numTargets += 1;
        }
        if (block_idx >= halfConcurrent && (block_idx - halfConcurrent + 1) <= rightTailRankSize) {
            numTargets += 1;
        }
    }

    for (uint32_t i = 0; i < numTargets; i++) {
        uint32_t targetRank;
        if (block_idx < halfConcurrent) {
            targetRank = (rank_ + rankSize_ - (halfConcurrent - block_idx) - i * halfConcurrent) % rankSize_; // left
        } else {
            targetRank = (rank_ + (block_idx - halfConcurrent + 1) + i * halfConcurrent) % rankSize_; // right
        }
        targetRanks[i] = targetRank;
    }
}

template<HardEvent event> 
__aicore__ inline void AivSync91093Base::SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline void AivSync91093Base::BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag)
{
    // 写所有对端的flag
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i];
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + rank_ * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }

    // 读自己的所有flag
    for (uint32_t i = 0; i < numTargets; i++) {
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + targetRanks[i] * FLAG_SIZE),
            UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag) {
                break;
            }
        }
        // 然后清零标志位
        DataCopy(globalTag, localClearTensor, UB_FLAG_PAD_COUNT); //清零
    }
}

#endif  /* AIV_SYCN_91093_BASE_H */