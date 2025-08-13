/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef REDUCE_SCATTER_LOCAL_REDUCE_PUB_H
#define REDUCE_SCATTER_LOCAL_REDUCE_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class ReduceScatterLocalReduce : public AlgTemplateBase {
public:
explicit ReduceScatterLocalReduce(const HcclDispatcher dispatcher);
~ReduceScatterLocalReduce() override;

HcclResult Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem, const u64 count,
    const HcclDataType dataType, const Stream &stream, const HcclReduceOp reductionOp, 
    const u32 root, const std::vector<Slice> &slices, const u64 baseOffset, 
    const u64 reduceAttrBitMap, std::vector<Stream> &meshStreams, 
    std::vector<std::shared_ptr<LocalNotify>> &meshSignal, 
    std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, 
    u32 userRank, const HcomCollOpInfo *opInfo = nullptr) override;
HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult MainRecordSub(u32 strean_num = -1);
    HcclResult SubWaitMain(u32 strean_num = -1);
    HcclResult MainWaitSub(u32 strean_num = -1);
    HcclResult SubRecordMain(u32 strean_num = -1);
    HcclResult PrepareOffset(u32 rankSize);
    HcclResult CalAlign(u64 totalSize, u32 rankSize, u64 &alignSize);
    HcclResult RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK> &links);
    HcclResult RunLocalReduce(u32 rank, u32 rankSize);
    u64 reduceAttr_ = 0;
    const u32 base = 2;
    u32 localRank_ = 0;
    u32 localRankSize_ = 0;
    u32 userRank_ = 0;
    std::vector<Stream> meshStreams_;               /* * 多stream* */
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalPtr_{nullptr};    /* 每个ring创建一个signal */
    const std::vector<std::shared_ptr<LocalNotify>> *meshSignalAuxPtr_{nullptr}; /* 从stream wait，主stream record */
    const HcomCollOpInfo *opInfo_{nullptr};
};
} // namespace hccl
#endif /* REDUCE_SCATTER_LOCAL_REDUCE_PUB_H */
