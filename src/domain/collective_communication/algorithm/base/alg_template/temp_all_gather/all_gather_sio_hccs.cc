/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_sio_hccs.h"
#include "alg_template_register.h"
 
namespace hccl {
AllGatherSioHccs::AllGatherSioHccs(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher) {
}
 
AllGatherSioHccs::~AllGatherSioHccs() {}

HcclResult AllGatherSioHccs::Prepare(SubCommInfo &outerCommInfoHccs, SubCommInfo &outerCommInfoSio, DeviceMem &usrInMem,
    DeviceMem &usrOutMem, u32 totalCount, const HcclDataType dataType, const Stream &mainStream,
    std::vector<Stream> &meshStreams, std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 userRank)
{
    inputMem_ = usrInMem;
    outputMem_ = usrOutMem;
    stream_ = mainStream;
    meshStreams_ = meshStreams;
    meshSignal_ = meshSignal;
    meshSignalAux_ = meshSignalAux;
    userRank_ = userRank;
    dataType_ = dataType;
    dataBytes_ = totalCount * SIZE_TABLE[dataType];
    outerCommInfoHccs_ = outerCommInfoHccs;
    outerCommInfoSio_ = outerCommInfoSio;
    return HCCL_SUCCESS;
}
 
// 主流所有从流
HcclResult AllGatherSioHccs::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, meshSignalAux_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, meshSignalAux_[streamIndex],
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}
 
HcclResult AllGatherSioHccs::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, meshSignal_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, meshSignal_[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherSioHccs::RunInterDie(const u32 dieRankId, const std::vector<LINK> &links, const u32 srcDMAMemSliceId)
{
    // 检查链接是否为空
    CHK_SMART_PTR_NULL(links[dieRankId]);

    // 获取远程内存指针
    void* remDMAMemPtr = nullptr;
    CHK_RET(links[dieRankId]->GetRemoteMem(UserMemType::INPUT_MEM,  &remDMAMemPtr));

    // 确定需要传输的数据部分（上半部分或下半部分）
    u64 dataPartOffset = (dieRankId % intraRankSize_) * dataBytes_;
    u64 dataPartSize = dataBytes_ / 2;

    DeviceMem locDieDst;
    DeviceMem srcDieMem;

    // 定义本地目标内存和远程源内存
    if (srcDMAMemSliceId == 0) {
        locDieDst = dmaMem_[1].range(dataPartOffset, dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr), dataPartSize);
    } else {
        locDieDst = dmaMem_[1].range(dataPartOffset + dataPartSize, dataBytes_ - dataPartSize);
        srcDieMem = DeviceMem::create(static_cast<u8*>(remDMAMemPtr) + dataPartSize, dataBytes_ - dataPartSize);
    }

    HCCL_INFO("RunInterDie: dieRankId[%d], dataPartOffset[%ld], dataPartSize[%ld]",
        dieRankId, dataPartOffset, dataPartSize);

    // 执行异步内存复制
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, srcDieMem, meshStreams_[srcDMAMemSliceId]));

    // 更新 notifyIdx_
    notifyIdx_++;

    // 发送和等待确认信号
    CHK_RET(links[dieRankId]->Post(notifyIdx_, meshStreams_[srcDMAMemSliceId])); // AckRecord
    CHK_RET(links[dieRankId]->Wait(notifyIdx_, meshStreams_[srcDMAMemSliceId])); // AckWait

    return HCCL_SUCCESS;
}
 
// allgather的入口函数
HcclResult AllGatherSioHccs::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    /*rank0:
    从userin拷贝到userout上半部分
    userout下半部分的上半部分通过sio读取rank1的userin上半部分
    userout下半部分的下半部分通过hccs读取rank1的userin下半部分
    */

    /*rank1:
    从userin拷贝到userout下半部分
    userout上半部分的上半部分通过sio读取rank0的userin上半部分
    userout上半部分的下半部分通过hccs读取rank0的userin下半部分
    */
    intraRankSize_ = rankSize;
    u32 dieRankId = (rank + 1) % rankSize;
 
    // dmaMem0部分userin，dmaMem1部分userout
    DeviceMem dmaMem0 = DeviceMem::create(inputMem_.ptr(), dataBytes_);
    DeviceMem dmaMem1 = DeviceMem::create(outputMem_.ptr(), dataBytes_ * intraRankSize_);
    DeviceMem locDieDst = dmaMem1.range(dataBytes_ * rank, dataBytes_);
 
    dmaMem_.push_back(dmaMem0);//userin
    dmaMem_.push_back(dmaMem1);//userout
 
    // 主流启动从流
    CHK_RET(NotifySubStreamStart());
    // step 0操作 : 所有卡本地数据从userIn-->userout
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, locDieDst, dmaMem0, stream_));
 
    // step 1 : die间 && device间并行收发
 
    // 数据搬运及后同步
    u32 srcDMAMemSliceId = 0;

    // 本地userout 读取die间 userin by sio
    CHK_RET(RunInterDie(dieRankId, outerCommInfoHccs_.links, srcDMAMemSliceId));
 
    notifyIdx_++;
 
    // 本地userout 读取die间 userin by hccs
    srcDMAMemSliceId++;
    CHK_RET(RunInterDie(dieRankId, outerCommInfoSio_.links, srcDMAMemSliceId));
 
    CHK_RET(WaitSubStreamFinish());
 
    HCCL_INFO("[AllGatherSioHccs][RunAsync]AllGatherSioHccs finished groupRankId[%u] ", userRank_);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_SIO_HCCS, AllGatherSioHccs);
}