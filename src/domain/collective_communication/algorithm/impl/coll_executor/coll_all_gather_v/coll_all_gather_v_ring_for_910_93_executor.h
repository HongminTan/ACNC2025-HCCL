/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_V_RING_FOR_910_93_EXECUTOR_H
#define COLL_ALLGATHER_V_RING_FOR_910_93_EXECUTOR_H
#include "coll_all_gather_ring_for_910_93_executor.h"
namespace hccl {
class CollAllGatherVRingFor91093Executor : public CollAllGatherRingFor91093Executor {
public:
    CollAllGatherVRingFor91093Executor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherVRingFor91093Executor() override = default;

private:
    // override grand
    bool IsSmallData(const u64 size) override;

    // override parent
    u64 CalcDstMemOffset(const OpParam &param, u32 perDataSize, u64 inputMemSize) const override;
    HcomCollOpInfo GetHcomCollOpInfo(const OpParam &param, const ExecMem &execMem) const override;
    std::vector<Slice> PrepareSlicesL2(const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize,
        u64 inputMemSize) const override;
    std::vector<Slice> PrepareSlicesL1(const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize,
        u64 inputMemSize) const override;
    HcclResult PrepareSlicesL0(std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param,
        const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo,
        u32 perDataSize, u64 inputMemSize) override;
    HcclResult PrepareUserMemSlices(std::vector<std::vector<Slice>> &userMemSlices,
        const std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize,
        u64 inputMemSize) override;
};

} // namespace hccl

#endif