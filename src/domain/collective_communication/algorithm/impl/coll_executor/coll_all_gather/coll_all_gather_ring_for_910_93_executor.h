/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALLGATHER_RING_FOR_910_93_EXECUTOR_H
#define COLL_ALLGATHER_RING_FOR_910_93_EXECUTOR_H
#include "coll_all_gather_executor.h"
namespace hccl {
class CollAllGatherRingFor91093Executor : public CollAllGatherExecutor {
public:
    explicit CollAllGatherRingFor91093Executor(const HcclDispatcher dispatcher, std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherRingFor91093Executor() = default;

private:
    /* *************** 资源计算 *************** */
    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult CalcTransportMemType(TransportMemType &inputType, TransportMemType &outputType);

    /* *************** 算法编排 *************** */
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize) override;
    bool IsDataSplitForRdmaSdmaConcurrent(const u64 curSize) override;

    virtual u64 CalcDstMemOffset(const OpParam &param, u32 perDataSize, u64 inputMemSize) const;
    virtual HcomCollOpInfo GetHcomCollOpInfo(const OpParam &param, const ExecMem &execMem) const;
    virtual std::vector<Slice> PrepareSlicesL2(const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize) const;
    virtual std::vector<Slice> PrepareSlicesL1(const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize) const;
    virtual HcclResult PrepareSlicesL0(std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param,
        const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo,
        u32 perDataSize, u64 inputMemSize);
    virtual HcclResult PrepareUserMemSlices(std::vector<std::vector<Slice>> &userMemSlices,
        const std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param, const SubCommInfo &level2CommInfo,
        const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize);

    virtual HcclResult RunIntraSeverAllGather(const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
        const u64 count, const HcclDataType &dataType,
        const std::vector<std::vector<Slice>> &multRingsSliceZero, const Stream &stream,
        s32 profStage, const u64 baseOffset = 0, const HcomCollOpInfo *opInfo = nullptr,
        const std::vector<std::vector<Slice>> &multRingsUserMemSlice = std::vector<std::vector<Slice>> (0));
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;
    HcclResult Getlevel1CommRank(SubCommInfo& level1CommInfo) override;
    HcclResult SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize) override;
};

} // namespace hccl

#endif