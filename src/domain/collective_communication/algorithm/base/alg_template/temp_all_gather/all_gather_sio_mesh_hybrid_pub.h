/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_SIO_MESH_HYBRID_PUB_H
#define ALL_GATHER_SIO_MESH_HYBRID_PUB_H

#include "alg_template_base.h"
#include "device_capacity.h"
#include "comm_base_pub.h"

namespace hccl {
class AllGatherSioMeshHybrid : public AlgTemplateBase {
public:
    explicit AllGatherSioMeshHybrid(const HcclDispatcher dispatcher);
    ~AllGatherSioMeshHybrid() override;
    
    HcclResult Prepare(std::vector<Stream> &meshStreams,
        std::vector<std::shared_ptr<LocalNotify>> &meshSignal, 
        std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
        u32 userRank, HcomCollOpInfo *opInfo, u32 interRank, u32 interRankSize);
    
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

private:
    HcclResult RunSioAllGather(const std::vector<LINK> &links, const std::vector<Slice> &outputSlices,
        const std::vector<Slice> &inputSlices);
    HcclResult RunMeshAllGather(const std::vector<LINK> &links, const std::vector<Slice> &outputSlices,
        const std::vector<Slice> &inputSlices);
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    
    std::vector<Stream> meshStreams_;
    std::vector<std::shared_ptr<LocalNotify>> *meshSignal_;
    std::vector<std::shared_ptr<LocalNotify>> *meshSignalAux_;
    u32 interRank_;
    u32 interRankSize_;
    u32 userRank_;
};

} // namespace hccl
#endif /* ALL_GATHER_SIO_MESH_HYBRID_PUB_H */
