/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_all_gather_v_semi_ring_executor.h"
#include <numeric>

namespace hccl {

CollAllGatherVSemiRingExecutor::CollAllGatherVSemiRingExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherSemiRingExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = true;
    isAllGatherV_ = true;
}

bool CollAllGatherVSemiRingExecutor::IsSmallData(const u64 size)
{
    return false;
}

u64 CollAllGatherVSemiRingExecutor::CalcDstMemOffset(const OpParam &param, u32 perDataSize, u64 inputMemSize) const
{
    const auto *counts = static_cast<const u64 *>(param.VDataDes.counts);
    const u64 offset = std::accumulate(counts, counts + topoAttr_.userRank, 0);
    return offset * perDataSize;
}

HcomCollOpInfo CollAllGatherVSemiRingExecutor::GetHcomCollOpInfo(const OpParam &param, const ExecMem &execMem) const
{
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, execMem.count, param.VDataDes.dataType, param.root,
        param.reduceType, 0 // 暂不支持MC2的strideCount特性
    };
    if (!DMAReduceFlag_) {
        opInfo.inputAddr = execMem.inputMem.ptr();
        opInfo.outputAddr = execMem.outputMem.ptr();
    }
    return opInfo;
}

HcclResult CollAllGatherVSemiRingExecutor::PrepareSlicesL0(std::vector<std::vector<Slice>> &multRingsSlice,
    const OpParam &param, const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo,
    const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    HCCL_CONFIG_INFO(HCCL_ALG,
        "[CollAllGatherVSemiRingExecutor][PrepareSlicesL0] userRank[%u] starts.", topoAttr_.userRank); 
    const auto *counts = static_cast<u64 *>(param.VDataDes.counts);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> dataSegsSlice;
    for (u32 k = 0; k < level0RankSize; k++) {  // 根据数据量计算每个环上数据的偏移和大小
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) {
                Slice sliceTemp;
                const u32 rank = (i * level1RankSize * level0RankSize) + (j * level0RankSize) + k;
                sliceTemp.size = counts[rank] * perDataSize;
                const u64 offset = std::accumulate(counts, counts + rank, 0);
                sliceTemp.offset = offset * perDataSize;    // no displs
                dataSegsSlice.push_back(sliceTemp);
            }
        }
    }
    multRingsSlice.push_back(dataSegsSlice);
    return HCCL_SUCCESS;
}

// AGV不支持MC2的strideCount特性
HcclResult CollAllGatherVSemiRingExecutor::PrepareUserMemSlices(std::vector<std::vector<Slice>> &userMemSlices,
    const std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param, const SubCommInfo &level2CommInfo,
    const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    const auto *counts = static_cast<u64 *>(param.VDataDes.counts);
    const auto *displs = static_cast<u64 *>(param.VDataDes.displs);
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> dataSegsSlice;
    for (u32 k = 0; k < level0RankSize; k++) {  // 根据数据量计算每个环上数据的偏移和大小
        for (u32 i = 0; i < level2RankSize; i++) {
            for (u32 j = 0; j < level1RankSize; j++) {
                Slice sliceTemp;
                const u32 rank = (i * level1RankSize * level0RankSize) + (j * level0RankSize) + k;
                sliceTemp.size = counts[rank] * perDataSize;
                sliceTemp.offset = displs[rank] * perDataSize;    // with displs
                dataSegsSlice.push_back(sliceTemp);
            }
        }
    }
    userMemSlices.push_back(dataSegsSlice);
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherVSemiRingExecutor", AllGatherVDoubleRingMidCount, CollAllGatherVSemiRingExecutor);

} // namespace hccl