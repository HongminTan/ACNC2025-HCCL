/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALL_GATHER_GRAPH_PIPELINE_PUB_H
#define ALL_GATHER_GRAPH_PIPELINE_PUB_H

#include "alg_template_base_pub.h"

namespace hccl {

class AllGatherGraphPipeline : public AlgTemplateBase {
public:
    explicit AllGatherGraphPipeline(const HcclDispatcher dispatcher);
    ~AllGatherGraphPipeline() override;

    HcclResult Prepare(HcomCollOpInfo *opInfo, u32 userRank, u64 &count, DeviceMem &inputMem, DeviceMem &outputMem,
        SubCommInfo &level0CommInfo, SubCommInfo &level1CommInfo, Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain,
        std::vector<std::shared_ptr<LocalNotify>> &notifySub) override;

    HcclResult RunAsync() override;

protected:
private:
    HcclResult MainRecordSub();
    HcclResult SubWaitMain();
    HcclResult MainWaitSub();
    HcclResult SubRecordMain();

    HcomCollOpInfo *opInfo_;
    u64 memSliceCount_ = 0;
    u32 userRank_ = 0;

    void *usrInMemAddr_ = nullptr;
    void *usrOutMemAddr_ = nullptr;
    std::vector<DeviceMem> dmaMem_;

    std::vector<Stream> subStream_;

    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    u32 intraRankSize_ = 0;
    u32 interRankSize_ = 0;
    u32 intraRankId_ = 0;
    u32 interRankId_ = 0;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;
};
}  // namespace hccl

#endif /* ALL_GATHER_GRAPH_PIPELINE_PUB_H */