/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_sio_mesh_hybrid.h"
#include "externalinput_pub.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherSioMeshHybrid::AllGatherSioMeshHybrid(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{
}

AllGatherSioMeshHybrid::~AllGatherSioMeshHybrid() {}

HcclResult AllGatherSioMeshHybrid::Prepare(std::vector<Stream> &meshStreams,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignal, std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
    u32 userRank, HcomCollOpInfo *opInfo, u32 interRank, u32 interRankSize)
{
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    interRank_ = interRank;
    interRankSize_ = interRankSize;
    userRank_ = userRank;
    return HCCL_SUCCESS;
}

// 主流启动所有从流
HcclResult AllGatherSioMeshHybrid::NotifySubStreamStart()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex], 
            INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

HcclResult AllGatherSioMeshHybrid::WaitSubStreamFinish()
{
    for (u32 streamIndex = 0; streamIndex < meshStreams_.size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[streamIndex], INVALID_VALUE_STAGE));
    }
    return HCCL_SUCCESS;
}

// SIO通信：适用于相邻rank间的高速通信 (偶数rank与偶数+1rank)
HcclResult AllGatherSioMeshHybrid::RunSioAllGather(const std::vector<LINK> &links, 
    const std::vector<Slice> &outputSlices, const std::vector<Slice> &inputSlices)
{
    // 确定SIO配对的rank
    u32 sioPartnerRank = (interRank_ % 2 == 0) ? interRank_ + 1 : interRank_ - 1;
    
    if (sioPartnerRank >= interRankSize_ || sioPartnerRank >= links.size()) {
        return HCCL_SUCCESS; // 没有SIO配对，跳过
    }

    CHK_SMART_PTR_NULL(links[sioPartnerRank]);
    
    // 使用第一个子流进行SIO通信
    Stream sioStream = meshStreams_[0];
    
    // 获取远程内存指针
    void* remoteMemPtr = nullptr;
    CHK_RET(links[sioPartnerRank]->GetRemoteMem(UserMemType::INPUT_MEM, &remoteMemPtr));
    
    // SIO通信：直接内存访问
    u64 sliceSize = inputSlices[sioPartnerRank].size;
    DeviceMem dstMem = outputMem_.range(outputSlices[sioPartnerRank].offset, sliceSize);
    DeviceMem srcMem = DeviceMem::create(static_cast<u8*>(remoteMemPtr), sliceSize);
    
    HCCL_DEBUG("rank[%u] SIO copy from partner rank[%u] size[%llu]", 
        interRank_, sioPartnerRank, sliceSize);
    
    // 执行异步内存复制
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, sioStream));
    
    // SIO同步
    CHK_RET(links[sioPartnerRank]->Post(1, sioStream));
    CHK_RET(links[sioPartnerRank]->Wait(1, sioStream));
    
    return HCCL_SUCCESS;
}

// Mesh通信：适用于非相邻rank间的UB通信
HcclResult AllGatherSioMeshHybrid::RunMeshAllGather(const std::vector<LINK> &links, 
    const std::vector<Slice> &outputSlices, const std::vector<Slice> &inputSlices)
{
    // 确定需要进行Mesh通信的rank列表（排除自己和SIO配对）
    std::vector<u32> meshRanks;
    u32 sioPartnerRank = (interRank_ % 2 == 0) ? interRank_ + 1 : interRank_ - 1;
    
    for (u32 peerRank = 0; peerRank < interRankSize_; peerRank++) {
        if (peerRank != interRank_ && peerRank != sioPartnerRank) {
            meshRanks.push_back(peerRank);
        }
    }
    
    // 使用多流进行Mesh通信
    for (u32 i = 0; i < meshRanks.size(); i++) {
        u32 peerRank = meshRanks[i];
        if (peerRank >= links.size()) continue;
        
        CHK_SMART_PTR_NULL(links[peerRank]);
        
        // 选择合适的流，避免与SIO流冲突
        u32 streamIndex = (i % (meshStreams_.size() - 1)) + 1; // 从第2个流开始使用
        Stream meshStream = (streamIndex < meshStreams_.size()) ? meshStreams_[streamIndex] : stream_;
        
        // 非对称握手协议避免死锁
        if (interRank_ < peerRank) {
            CHK_RET(links[peerRank]->TxAck(meshStream));
            CHK_RET(links[peerRank]->RxAck(meshStream));
        } else {
            CHK_RET(links[peerRank]->RxAck(meshStream));
            CHK_RET(links[peerRank]->TxAck(meshStream));
        }
        
        // 发送本rank的数据到peer
        Slice mySlice = outputSlices[interRank_];
        CHK_RET(links[peerRank]->TxAsync(UserMemType::OUTPUT_MEM,
            mySlice.offset + baseOffset_, outputMem_.range(mySlice.offset, mySlice.size).ptr(), 
            mySlice.size, meshStream));
        
        // 接收peer的数据
        Slice peerSlice = outputSlices[peerRank];
        CHK_RET(links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM,
            peerSlice.offset + baseOffset_, outputMem_.range(peerSlice.offset, peerSlice.size).ptr(), 
            peerSlice.size, meshStream));
        
        HCCL_DEBUG("rank[%u] Mesh exchange with rank[%u] using stream[%u]", 
            interRank_, peerRank, streamIndex);
    }
    
    // 等待所有Mesh通信完成
    for (u32 i = 0; i < meshRanks.size(); i++) {
        u32 peerRank = meshRanks[i];
        if (peerRank >= links.size()) continue;
        
        u32 streamIndex = (i % (meshStreams_.size() - 1)) + 1;
        Stream meshStream = (streamIndex < meshStreams_.size()) ? meshStreams_[streamIndex] : stream_;
        
        CHK_RET(links[peerRank]->TxWaitDone(meshStream));
        CHK_RET(links[peerRank]->RxWaitDone(meshStream));
    }
    
    return HCCL_SUCCESS;
}

// 混合SIO+Mesh AllGather的主入口函数
HcclResult AllGatherSioMeshHybrid::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherSioMeshHybrid run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", 
        rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    interRank_ = rank;
    interRankSize_ = rankSize;

    if (interRankSize_ == 1) {
        if (inputMem_ != outputMem_) {
            HCCL_DEBUG("rank[%u] mem copy async from input to output", rank);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherSioMeshHybrid][RunAsync]rank[%u] linksize error", rank);
        return HCCL_E_INTERNAL;
    }
    
    // 计算所需的子流数量：SIO需要1个流，Mesh需要若干流
    u32 sioStreamNum = 1;
    u32 meshStreamNum = (interRankSize_ > 2) ? interRankSize_ - 3 : 0; // 排除自己、SIO配对和一个主流
    u32 totalSubStreamNum = sioStreamNum + meshStreamNum;
    
    if (meshStreams_.size() < totalSubStreamNum || (*meshSignal_).size() < totalSubStreamNum ||
        (*meshSignalAux_).size() < totalSubStreamNum) {
        HCCL_ERROR("[AllGatherSioMeshHybrid][RunAsync] stream size error: "
            "rank[%u] totalrank:%u requiredStreams[%u] availableStreams[%llu] signalsize[%llu]",
            rank, rankSize, totalSubStreamNum, meshStreams_.size(), (*meshSignal_).size());
        return HCCL_E_PARA;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherSioMeshHybrid][RunAsync] unitSize is zero");
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(interRankSize_);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < interRankSize_; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = 0;  // Input data is always at offset 0 for each rank
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", 
                rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }

    // Step 1: 如果input和output不一样，先把input的数据拷贝到output的对应位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(0, slices_[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    // Step 2: 启动子流
    CHK_RET(NotifySubStreamStart());

    // Step 3: 并行执行SIO和Mesh通信
    // SIO通信在第一个子流上进行
    CHK_RET(RunSioAllGather(links, slices_, inputSlices));
    
    // Mesh通信在剩余的子流上进行
    CHK_RET(RunMeshAllGather(links, slices_, inputSlices));

    // Step 4: 等待所有子流完成
    CHK_RET(WaitSubStreamFinish());

    CHK_RET(AlgTemplateBase::ExecEmptyTask(inputMem_, outputMem_, stream_, dispatcher_));
    HCCL_INFO("AllGatherSioMeshHybrid finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_SIO_MESH_HYBRID, AllGatherSioMeshHybrid);
} // namespace hccl
