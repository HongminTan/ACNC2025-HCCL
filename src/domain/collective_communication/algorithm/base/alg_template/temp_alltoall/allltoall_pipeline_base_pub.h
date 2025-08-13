/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALLTOALL_PIPELINE_BASE_PUB_H
#define ALLTOALL_PIPELINE_BASE_PUB_H

#include <cstring>
#include <vector>
#include <memory>
#include <list>
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "alg_template_base_pub.h"

namespace hccl {

// 定义 alltoall pipeline 系列算法的一些公共实现，和整体的抽象行为
class AlltoallPipelineBase : public AlgTemplateBase {
public:
    explicit AlltoallPipelineBase(const HcclDispatcher dispatcher);
    virtual ~AlltoallPipelineBase();

    // 适配新CollExecutor接口
    virtual HcclResult Prepare(u32 userRank, A2aPipelineMemory A2aPipelineMemory,
        const SubCommInfo &level0CommInfo, const SubCommInfo &level1CommInfo,
        Stream &mainStream, std::vector<Stream> &subStream,
        std::vector<std::shared_ptr<LocalNotify>> &notifyMain, std::vector<std::shared_ptr<LocalNotify>> &notifySub,
        std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo,
        HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) override;

    HcclResult RunAsync() override;
    HcclResult GetNslbAdjInfo(const u32 rank, const u32 rankSize,
                              const std::vector<LINK> &links, AdjInfo& nslbAdjInfo) override;

protected:
    virtual HcclResult DeviceMemMapping() = 0;
    virtual HcclResult CheckResourceValid();
    virtual HcclResult PreProcess() = 0;
    virtual HcclResult PostProcess() = 0;
    virtual u32 CalcInterNumSteps() = 0;
    virtual HcclResult PipelineSend(u32 step, bool isLastStep) = 0;

    std::string GetCurrClassName();
    std::string GetStreamIndexString();
    HcclResult NotifyInterStreamStart();
    HcclResult WaitInterStreamFinish();
    HcclResult NotifyIntraStreamStart();
    HcclResult WaitIntraStreamFinish();

    std::vector<SendRecvInfo> *allMeshAggregationSendRecvInfo_{nullptr};
    SendRecvInfo localSendRecvInfo_;
    HcclWorkflowMode workMode_;

    DeviceMem cclIn_;
    DeviceMem cclOut_;

    u32 groupRankSize_ = 0;
    u32 intraRankSize_ = 0;
    u32 interRankSize_ = 0;
 
    u32 userRank_ = 0;
    u32 intraRankId_ = 0;
    u32 interRankId_ = 0;
 
    u32 meshRankStart_ = 0;
    u32 meshRankEnd_ = 0;

    Stream mainStream_;
    std::vector<Stream> subStream_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifyMain_;
    std::vector<std::shared_ptr<LocalNotify>> streamNotifySub_;

    std::vector<LINK> intraLinks_;
    std::vector<LINK> interLinks_;

    // 为了各种场景代码通用，需要对内存做一定的映射, 在一些情况下 interTransportRecv_ 和 intraTransportSend_ 会是同一块
    DeviceMem interTransportSend_;
    DeviceMem interTransportRecv_;
    DeviceMem intraTransportSend_;
    DeviceMem intraTransportRecv_;
    std::unordered_map<u32, std::vector<DeviceMem>> intraNeighBoorMemory_;

    //              SDMA流       发送数据长度 接收数据长度 接收数据本地偏移
    std::unordered_map<u32, std::vector<u64>> intraStreamInfo_;
};
}  // namespace hccl

#endif /* ALLTOALL_PIPELINE_BASE_PUB_H */