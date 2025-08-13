/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H
#define ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H  
 
#include <cmath>
#include <algorithm>
#include "alg_template_base_pub.h"
#include "asymmetric_hierarchical_concatenate_base_pub.h"
#include "comm_ahc_base_pub.h"
#include "device_capacity.h"
 
namespace hccl {
 
class AHCAlgTemplateBase : public AlgTemplateBase {
public:
    explicit AHCAlgTemplateBase(const HcclDispatcher dispatcher);
    ~AHCAlgTemplateBase() override;

    HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo *opInfo = nullptr) override;

    HcclResult Prepare(u64 totalCount, const std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
        std::map<AHCConcOpType, TemplateType> &ahcAlgOption, bool extendFlag = false,
        AHCExtendPreparePara extendPara = AHCExtendPreparePara()) override;

    HcclResult GetFftsPhase(u32 &fftsPhase) const;
    HcclResult SetFftsPhase(u32 &fftsPhase);
 
    u64 reduceAttr_;
protected:
    virtual HcclResult CommAHCInfoInit();
    virtual HcclResult DisposeSubGroups(const u32 rank);
 
    HcclResult PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    HcclResult PrepareAlgTemplate (std::unique_ptr<AlgTemplateBase> &tempAlg, const std::vector<Slice> &slices,
        AHCOpType opType);
    HcclResult RunInstance(const u32 rank, const std::vector<LINK> &links, const std::vector<Slice> &slices,
        std::unique_ptr<AlgTemplateBase> &tempAlg, AHCOpType opType);
    HcclResult MemcpyForSingleOp(const u32 rank, AHCOpType opType);
    
    u32 rankSize_;
    std::unique_ptr<CommAHCBaseInfo> commAHCBaseInfo_;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption_;
    std::vector<std::vector<u32>> level0SubGroups_;
    std::vector<std::vector<u32>> level1SubGroups_;
    std::vector<std::vector<std::vector<u32>>> globalSubGroups_;
    u64 totalCount_; // 完整数据量大小，用于判断是否为hugeData
    bool extendFlag_;
    AHCExtendPreparePara ahcExtendPreparePara_;
private:
    u32 fftsPhase_; // 表示 AHC 算法内部的多个子图阶段，0为默认phase，算法内部从1开始
};
 
class AllGatherAHCBase : public AHCAlgTemplateBase {
public:
    explicit AllGatherAHCBase(const HcclDispatcher dispatcher);
    ~AllGatherAHCBase() override;
 
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
private:
    HcclResult RunIntraAllGather(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);
    virtual HcclResult RunInterAllGather(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo) = 0;
};
 
class AllReduceAHCBase : public AHCAlgTemplateBase {
public:
    explicit AllReduceAHCBase(const HcclDispatcher dispatcher);
    ~AllReduceAHCBase() override;
 
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
private:
    HcclResult RunIntraReduceScatter(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);
    HcclResult RunIntraAllGather(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);
    virtual HcclResult RunInterAllReduce(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo) = 0;
};
 
class ReduceScatterAHCBase : public AHCAlgTemplateBase {
public:
    explicit ReduceScatterAHCBase(const HcclDispatcher dispatcher);
    ~ReduceScatterAHCBase() override;
 
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
private:
    HcclResult RunIntraReduceScatter(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo);
    virtual HcclResult RunInterReduceScatter(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo) = 0;
};
} // hccl

#endif /* ASYMMETRIC_HIERARCHICAL_CONCATENATE_ALG_TEMPLATE_BASE_PUB_H */