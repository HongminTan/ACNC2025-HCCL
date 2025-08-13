/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SCATTER_NHR_PUB_H
#define SCATTER_NHR_PUB_H

#include "nonuniform_hierarchical_ring_base_pub.h"

#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"

namespace hccl {
class ScatterNHR : public NHRBase {
public:
    explicit ScatterNHR(const HcclDispatcher dispatcher);

    ~ScatterNHR() override;

    // should be called soon after template ScatterNHR instance created
    HcclResult Prepare(bool needMerge) override;

    HcclResult RunAsync(const u32 rank, const u32 rankSize,
                                   const std::vector<std::shared_ptr<Transport> > &links) override;
    HcclResult GetNslbAdjInfo(const u32 rank, const u32 rankSize,
                              const std::vector<LINK> &links, AdjInfo& nslbAdjInfo) override;

protected:
private:
    void PrepareSlicesData(const u32 unitSize, const u64 totalCount, const u32 rankSize) const;
    HcclResult SdmaRx(LINK &linkLeft, LINK &linkRight, InterServerAlgoStep &stepInfo,
        const std::vector<LINK> &links);
    HcclResult RdmaTxRx(LINK &linkLeft, LINK &linkRight, InterServerAlgoStep &stepInfo,
        const std::vector<LINK> &links);
    HcclResult RdmaTxRx();
    HcclResult RunScatterNHR(const std::vector<std::shared_ptr<Transport> > &links);
    HcclResult Tx(const LINK &link, std::vector<Slice> &txSlices);
    HcclResult Rx(const LINK &link, std::vector<Slice> &rxSlices);

    HcclResult GetStepInfo(u32 step, u32 nSteps, u32 rank, u32 rankSize, InterServerAlgoStep &stepInfo) override;

    u32 interRank_;       // comm内的rank排序
    u32 interRankSize_;  // 本comm内ranksize总数
};
}  // namespace hccl

#endif /* SCATTER_NHR_PUB_H */
