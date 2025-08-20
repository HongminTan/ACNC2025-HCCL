/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALL_GATHER_SIO_MESH_HYBRID_EXECUTOR_H
#define COLL_ALL_GATHER_SIO_MESH_HYBRID_EXECUTOR_H

#include "coll_all_gather_executor.h"

namespace hccl {
class CollAllGatherSioMeshHybridExecutor : public CollAllGatherExecutor {
public:
    CollAllGatherSioMeshHybridExecutor(const HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher> &topoMatcher);
    ~CollAllGatherSioMeshHybridExecutor() = default;

    HcclResult CalcStreamNum(u32& streamNum) override;
    HcclResult CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport) override;
    HcclResult KernelRun(const OpParam &param, ExecMem &execMem) override;

private:
    HcclResult CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
        std::vector<LevelNSubCommTransport>& opTransport);
    u64 CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize);
};

} // namespace hccl
#endif /* COLL_ALL_GATHER_SIO_MESH_HYBRID_EXECUTOR_H */
