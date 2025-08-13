/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_PLANT_LOCAL_REDUCE_COMBINE_H
#define REDUCE_SCATTER_PLANT_LOCAL_REDUCE_COMBINE_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterPlantLocalReduceCombine : public AlgTemplateBase {
public:
    explicit ReduceScatterPlantLocalReduceCombine(const HcclDispatcher dispatcher);
    ~ReduceScatterPlantLocalReduceCombine() override;

    HcclResult Prepare(DeviceMem &cclInMem, DeviceMem &outputMem, const Stream &stream, std::vector<Stream> &subStreams,
        std::vector<std::shared_ptr<LocalNotify>> &meshSignal, std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
        MemBlockInfo &memBlockInfo, const HcclReduceOp reductionOp, const HcclDataType dataType, 
        bool isUseCclIn, bool isLevel0LastRank) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult MainRecordSub(Stream &mainStream, u32 firstSubStreamIndex, u32 totalTask);    // 主流通知从流
    HcclResult SubWaitMain(u32 firstSubStreamIndex, u32 totalTask);      // 从流等待主流
    HcclResult MainWaitSub(Stream &mainStream, u32 firstSubStreamIndex, u32 totalTask);      // 主流等待从流
    HcclResult SubRecordMain(u32 firstSubStreamIndex, u32 totalTask);    // 从流等待主流
    u32 CalcOutputIndex(const u32 round);
    bool isLastRank(const u32 rankId);
    bool isLastBlockData(const u32 outputIndex);
    HcclResult LocalCopy();
    HcclResult RunAlltoAllRDMA(u32 round, u64 sliceSize, const std::vector<LINK> &links);
    HcclResult RunAlltoAllSDMA(u32 round, u64 sliceSize, const std::vector<LINK> &links);
    HcclResult RunAlltoAll(const std::vector<LINK> &links);
    HcclResult RunLocalReduce();
    u32 rankSize_{0};
    u32 localRank_{0};
    bool isUseCclIn_{false};
    bool isLevel0LastRank_{false};
    std::vector<Stream> subStreams_;  // 从流
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalPtr_{nullptr};
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalAuxPtr_{nullptr};
    MemBlockInfo memBlockInfo_;
};
} // namespace hccl
#endif /* REDUCE_SCATTER_PLANT_LOCAL_REDUCE_COMBINE_H */
