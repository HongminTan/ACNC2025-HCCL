/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_ALLGATHER_CROSSNODE_91093_BASE_H
#define AIV_ALLGATHER_CROSSNODE_91093_BASE_H

#include "aiv_communication_base.h"

using namespace AscendC;

class AivAllGatherCrossNode91093Base {
public:
    __aicore__ inline AivAllGatherCrossNode91093Base() {}

    template<typename T>
    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize, uint32_t baseFlagOffset,
        uint64_t bufferCount, uint64_t len); // 单算子的init

    template<typename T>
    __aicore__ inline void Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize, uint32_t baseFlagOffset,
        uint64_t len); // 图模式的init

    template<typename T>
    __aicore__ inline void SetAtomicOp(uint32_t atomicOp);

    __aicore__ inline uint64_t CeilDiv(uint64_t a, uint64_t b);
    __aicore__ inline uint64_t CalActualCount(uint32_t sliceIdx, uint64_t sliceCount, uint64_t avgLengthPerSlice,
        uint64_t tailLength);
    __aicore__ inline void CalCountAndBlockOffset(uint64_t len, uint32_t blockNumPerGroup, uint32_t blockIdxInGroup, 
        uint32_t padCount, uint64_t &count, uint64_t &blockOffset);

    template<typename T>
    __aicore__ inline void DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
        const uint32_t calCount);

    template<typename T>
    __aicore__ inline void CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count);

    template<typename T>
    __aicore__ inline void CpGM2GMInlineReduce(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t atomicOp);

    template<HardEvent event> 
    __aicore__ inline void SyncFunc();

    __aicore__ inline void SingleRecordBatchWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag, bool isTheSingleCore);
    __aicore__ inline void BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag);
    __aicore__ inline void BatchRecordSingleWaitCoreLevel(uint32_t flagOffset, int32_t curTag, bool isTheSingleCore);

    __aicore__ inline void InitOpCounter(GM_ADDR headCountMem, GM_ADDR tailCountMem, GM_ADDR addOneMem, uint32_t counterMemSize,
        bool isEnableCounter)
    {
        headCountMem_ = headCountMem;
        tailCountMem_ = tailCountMem;
        addOneMem_ = addOneMem;
        counterMemSize_ = counterMemSize;
        isEnableCounter_ = isEnableCounter;
    }

    __aicore__ inline void HeadCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GMInlineReduce((__gm__ int32_t*)headCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t),
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

    __aicore__ inline void TailCounter()
    {
        if (block_idx == 0 && isEnableCounter_) {
            CpGM2GMInlineReduce((__gm__ int32_t*)tailCountMem_, (__gm__ int32_t*)addOneMem_, counterMemSize_ / sizeof(int32_t),
                HcclReduceOp::HCCL_REDUCE_SUM);
        }
    }

protected:
    uint32_t baseFlagOffset_ = 0;
    GM_ADDR flagAddrSelf_;
    uint32_t rank_;
    uint32_t rankSize_;

    TPipe pipe;

    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, 1> inOutQue;
    TBuf<> localFlagBuf;
    LocalTensor<int32_t> localSetTensor;
    LocalTensor<int32_t> localCheckTensor;
    LocalTensor<int32_t> localClearTensor;
    TBuf<> bufferArgsBuf;
    LocalTensor<uint64_t> bufferArgsTensor; // buffer地址GM-UB

    // 每个aiv核的数据搬运参数
    uint32_t numTargets; // 每个aiv需要顺序与几个对端通信，ranksize太大时，aiv不够用，需要多次
    uint32_t targetRanks[MAX_TARGET_NUM] = {}; // 最多768/48 = 16 次（一次代表服务48张卡）
    uint32_t blockNumPerGroup; // 多少个aiv服务一个rank
    uint64_t countMid; // 中间轮一个aiv负责搬运的数据量（一轮代表一次ccl buffer装满）
    uint64_t countTail; // 尾轮一个aiv负责搬运的数据量
    uint64_t blockOffsetMid; // 数据块offset，区分中间轮和尾轮
    uint64_t blockOffsetTail;
    uint32_t flagOffsetInGroup; // 标志位offset，不区分中间轮和尾轮
    uint64_t blockOffset; // 数据块offset，不区分中间轮和尾轮
    uint64_t countPerCore; // 每个核负责的数据块大小，不区分中间轮和尾轮

    // 维测相关
    GM_ADDR headCountMem_;
    GM_ADDR tailCountMem_;
    GM_ADDR addOneMem_;
    uint32_t counterMemSize_;
    bool isEnableCounter_;
};

__aicore__ inline uint64_t AivAllGatherCrossNode91093Base::CeilDiv(uint64_t a, uint64_t b)
{
    if (b == 0) {
        return a;
    }
    return (a + b - 1) / b;
}

__aicore__ inline uint64_t AivAllGatherCrossNode91093Base::CalActualCount(uint32_t sliceIdx, uint64_t sliceCount,
    uint64_t avgLengthPerSlice, uint64_t tailLength)
{
    if (sliceIdx == sliceCount - 1) {
        return tailLength;
    } else if (sliceIdx < sliceCount - 1) {
        return avgLengthPerSlice;
    } else {
        return 0;
    }
}

__aicore__ inline void AivAllGatherCrossNode91093Base::CalCountAndBlockOffset(uint64_t len, uint32_t blockNumPerGroup, 
    uint32_t blockIdxInGroup, uint32_t padCount, uint64_t &count, uint64_t &blockOffset)
{
    uint64_t avgLengthPerBlock = CeilDiv(len, blockNumPerGroup);
    uint64_t avgLengthPerSlice = CeilDiv(avgLengthPerBlock, padCount) * padCount; // 32B对齐
    uint64_t sliceCount = CeilDiv(len, avgLengthPerSlice);
    uint64_t tailLength = len - (sliceCount - 1) * avgLengthPerSlice; // 多核并行搬数据，最后一核搬运的数据量
    count = CalActualCount(blockIdxInGroup, sliceCount, avgLengthPerSlice, tailLength);
    blockOffset = blockIdxInGroup * avgLengthPerSlice;
}

// 单算子的Init
template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
    uint32_t baseFlagOffset, uint64_t bufferCount, uint64_t len)
{
    baseFlagOffset_ = baseFlagOffset;
    flagAddrSelf_ = buffOut0 + baseFlagOffset;

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

    // 以下根据不同情况，计算每个aiv核的数据搬运参数
    // 当rankSize大于总aiv核数的一半时，使用1个aiv服务一个对端，需要多次通信
    if (rankSize > HALF_MAX_BLOCK_DIM) {
        // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
        uint32_t halfConcurrent = block_num / 2; // block_num需要为偶数
        numTargets = (rankSize_ - 1) / block_num; // 除去本rank，可能需要补上一个
        uint32_t tailRankSize = (rankSize_ - 1) % block_num;
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

        blockNumPerGroup = 1;
        if (len <= bufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;
            countTail = len;
        } else if (len % bufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            countMid = bufferCount;
            countTail = bufferCount;
        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            countMid = bufferCount;
            countTail = len % bufferCount;
        }
        blockOffsetMid = 0;
        blockOffsetTail = 0;
        flagOffsetInGroup = 0;
        countPerCore = len;
        blockOffset = 0;

    // 当rankSize小于等于总aiv核数的一半时，根据ranksize和数据量大小选择使用多个aiv服务一个对端（多核并行），只需一次通信
    } else {
        numTargets = 1;
        blockNumPerGroup = block_num / rankSize_; // 多少个aiv服务一个rank
        targetRanks[0] = block_idx / blockNumPerGroup;

        uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
        uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;

        if (len <= bufferCount) { // ccl够用，只需要搬一轮的情况
            countMid = 0;
            blockOffsetMid = 0;

            CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);

        } else if (len % bufferCount == 0) { // ccl不够用，要搬多轮的情况1: 能整除
            CalCountAndBlockOffset(bufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
            
            countTail = countMid;
            blockOffsetTail = blockOffsetMid;

        } else { // ccl不够用，要搬多轮的情况2: 不能整除
            CalCountAndBlockOffset(bufferCount, blockNumPerGroup, blockIdxInGroup, padCount, countMid, blockOffsetMid);
            
            uint64_t remainLen = len % bufferCount;
            CalCountAndBlockOffset(remainLen, blockNumPerGroup, blockIdxInGroup, padCount, countTail, blockOffsetTail);
        }
        flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;

        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    }
}

// 图模式的Init
template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::Init(GM_ADDR buffOut0, uint32_t rank, uint32_t rankSize,
    uint32_t baseFlagOffset, uint64_t len)
{
    baseFlagOffset_ = baseFlagOffset;
    flagAddrSelf_ = buffOut0 + baseFlagOffset;

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

    // 以下根据不同情况，计算每个aiv核的数据搬运参数
    // 当rankSize大于总aiv核数的一半时，使用1个aiv服务一个对端，需要多次通信
    if (rankSize > HALF_MAX_BLOCK_DIM) {
        // 前concurrentSize/2个aiv负责与左边rank号的通信，后concurrentSize/2个负责与右边rank号的通信
        uint32_t halfConcurrent = block_num / 2; // block_num需要为偶数
        numTargets = (rankSize_ - 1) / block_num; // 除去本rank，可能需要补上一个
        uint32_t tailRankSize = (rankSize_ - 1) % block_num;
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

        blockNumPerGroup = 1;
        flagOffsetInGroup = 0;
        countPerCore = len;
        blockOffset = 0;

    // 当rankSize小于等于总aiv核数的一半时，根据ranksize和数据量大小选择使用多个aiv服务一个对端（多核并行），只需一次通信
    } else {
        numTargets = 1;
        blockNumPerGroup = block_num / rankSize_; // 多少个aiv服务一个rank
        targetRanks[0] = block_idx / blockNumPerGroup;

        uint32_t padCount = UB_ALIGN_SIZE / sizeof(T);
        uint32_t blockIdxInGroup = block_idx % blockNumPerGroup;

        flagOffsetInGroup = blockIdxInGroup * FLAG_SIZE;
        CalCountAndBlockOffset(len, blockNumPerGroup, blockIdxInGroup, padCount, countPerCore, blockOffset);
    }
}

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::SetAtomicOp(uint32_t atomicOp)
{
    switch (atomicOp) {
        case HcclReduceOp::HCCL_REDUCE_SUM:
            SetAtomicAdd<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MAX:
            SetAtomicMax<T>(); break;
        case HcclReduceOp::HCCL_REDUCE_MIN:
            SetAtomicMin<T>(); break;
        default:
            SetAtomicNone(); break;
    }
}

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::DataCopyGM2UB(const LocalTensor<T>& dstLocal, const GlobalTensor<T>& srcGlobal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstLocal, srcGlobal, calCount);
    } else {
        // 结构体DataCopyExtParams最后一个参数是rsv保留位
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{true, 0, 1, 0};
        DataCopyPad(dstLocal, srcGlobal, copyParams, padParams);
    }
}

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::DataCopyUB2GM(const GlobalTensor<T>& dstGlobal, const LocalTensor<T>& srcLocal,
    const uint32_t calCount)
{
    if ((calCount * sizeof(T)) % UB_ALIGN_SIZE == 0) {
        DataCopy(dstGlobal, srcLocal, calCount);
    } else {
        DataCopyExtParams copyParams{1, calCount * (uint32_t)sizeof(T), 0, 0, 0};
        DataCopyPad(dstGlobal, srcLocal, copyParams);
    }
}

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::CpGM2GM(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);

    uint64_t maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);

    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;

        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);

        count -= curCount;
        curOffset += curCount;
    }
    return;
}

template<typename T>
__aicore__ inline void AivAllGatherCrossNode91093Base::CpGM2GMInlineReduce(__gm__ T *outputGM, __gm__ T *inputGM, uint64_t count, uint32_t atomicOp)
{
    GlobalTensor<T> inputGT;
    inputGT.SetGlobalBuffer(inputGM, count);
    GlobalTensor<T> outputGT;
    outputGT.SetGlobalBuffer(outputGM, count);
    
    SetAtomicOp<T>(atomicOp);
 
    uint64_t maxCountPerLoop = UB_DB_DATA_BATCH_SIZE / sizeof(T);
 
    uint64_t curOffset = 0;
    while (count > 0) {
        uint64_t curCount = count > maxCountPerLoop ? maxCountPerLoop : count;
 
        LocalTensor<T> localIn = inOutQue.AllocTensor<T>();
        DataCopyGM2UB(localIn, inputGT[curOffset], curCount);
        inOutQue.EnQue(localIn);
        LocalTensor<T> localOut = inOutQue.DeQue<T>();
        DataCopyUB2GM(outputGT[curOffset], localOut, curCount);
        inOutQue.FreeTensor(localOut);
 
        count -= curCount;
        curOffset += curCount;
    }
    SetAtomicNone();
    return;
}

template<HardEvent event> 
__aicore__ inline void AivAllGatherCrossNode91093Base::SyncFunc() {
    int32_t eventID = static_cast<int32_t>(GetTPipePtr()->FetchEventID(event));
    SetFlag<event>(eventID);
    WaitFlag<event>(eventID);
}

__aicore__ inline void AivAllGatherCrossNode91093Base::SingleRecordBatchWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag, bool isTheSingleCore)
{
    GlobalTensor<int32_t> globalTag;
    // 写一个自己的flag，标志自己的那一片localcopy(input->ccl)完成
    if (isTheSingleCore) {
        localSetTensor.SetValue(0, curTag);
        SyncFunc<HardEvent::S_MTE3>();
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + flagOffsetInGroup), UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }

    // 读所有对端的flag，确保所有对端的localcopy(input->ccl)完成了
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i] + baseFlagOffset_;
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + flagOffsetInGroup), UB_FLAG_PAD_COUNT);
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag) {
                break;
            }
        }
    }
}

__aicore__ inline void AivAllGatherCrossNode91093Base::BatchRecordWait(GM_ADDR* buffersOut, uint32_t flagOffset, int32_t curTag)
{
    // 写所有对端的flag
    localSetTensor.SetValue(0, curTag);
    GlobalTensor<int32_t> globalTag;
    SyncFunc<HardEvent::S_MTE3>();
    for (uint32_t i = 0; i < numTargets; i++) {
        GM_ADDR flagAddrOther = buffersOut[i] + baseFlagOffset_;
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrOther + flagOffset + rank_* blockNumPerGroup * FLAG_SIZE + flagOffsetInGroup),
            UB_FLAG_PAD_COUNT);
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
    }

    // 读自己的所有flag
    for (uint32_t i = 0; i < numTargets; i++) {
        globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + targetRanks[i] * blockNumPerGroup * FLAG_SIZE + flagOffsetInGroup),
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

__aicore__ inline void AivAllGatherCrossNode91093Base::BatchRecordSingleWaitCoreLevel(uint32_t flagOffset, int32_t curTag, bool isTheSingleCore)
{
    GlobalTensor<int32_t> globalTag;
    globalTag.SetGlobalBuffer((__gm__ int32_t *)(flagAddrSelf_ + flagOffset + flagOffsetInGroup), UB_FLAG_PAD_COUNT);
    // 负责localcopy的核去查该flag，等所有其他核已经完成写（原子累加）
    if (isTheSingleCore) {
        while (true) {
            DataCopy(localCheckTensor, globalTag, UB_FLAG_PAD_COUNT);
            SyncFunc<HardEvent::MTE2_S>();
            if (localCheckTensor.GetValue(0) == curTag * (rankSize_ - 1)) {
                break;
            }
        }
        // 然后清零标志位
        DataCopy(globalTag, localClearTensor, UB_FLAG_PAD_COUNT); //清零

    // 其他核去写该flag，做原子累加达到核间同步的目的
    } else {   
        Duplicate<int32_t>(localSetTensor, curTag, UB_FLAG_PAD_COUNT);  
        SyncFunc<HardEvent::S_MTE3>();
        SetAtomicAdd<int32_t>();
        DataCopy(globalTag, localSetTensor, UB_FLAG_PAD_COUNT);
        SetAtomicNone();
    }
}

#endif  /* AIV_ALLGATHER_CROSSNODE_91093_BASE_H */