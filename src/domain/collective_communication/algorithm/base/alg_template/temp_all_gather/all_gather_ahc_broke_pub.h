/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef ALL_GATHER_AHC_BROKE_PUB_H
#define ALL_GATHER_AHC_BROKE_PUB_H
 
#include "asymmetric_hierarchical_concatenate_alg_template_base_pub.h"
 
namespace hccl {
class AllGatherAHCBroke : public AllGatherAHCBase {
public:
    explicit AllGatherAHCBroke(const HcclDispatcher dispatcher);
 
    ~AllGatherAHCBroke() override;
 
private:
    HcclResult DisposeSubGroups(const u32 rank) override;
    HcclResult CommAHCInfoInit() override;
    HcclResult RunInterAllGather(const u32 rank, const std::vector<LINK> &links,
        const std::unique_ptr<CommAHCBaseInfo> &commAHCBaseInfo) override;
};
} // namespace hccl
#endif /* ALL_GATHER_AHC_BROKE_PUB_H */