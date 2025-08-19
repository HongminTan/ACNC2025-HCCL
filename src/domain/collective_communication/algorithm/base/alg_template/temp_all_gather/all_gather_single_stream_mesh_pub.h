/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLGATHER_SINGLE_STREAM_MESH_H_PUB_H
#define ALLGATHER_SINGLE_STREAM_MESH_H_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {
class AllGatherSingleStreamMesh : public AlgTemplateBase {
public:
    explicit AllGatherSingleStreamMesh(const HcclDispatcher dispatcher);
    ~AllGatherSingleStreamMesh() override;
    //任务下发的具体执行逻辑，由KernelRun调度
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;
    
private:
    HcclResult RunAllgather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices, const std::vector<LINK> &links);
    
};
}  // namespace hccl

#endif /* ALLGATHER_SINGLE_STREAM_MESH_H_PUB_H */