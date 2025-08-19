/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "all_gather_single_stream_mesh.h"
#include "alg_template_register.h"

namespace hccl {
AllGatherSingleStreamMesh::AllGatherSingleStreamMesh(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher)
{
}

AllGatherSingleStreamMesh::~AllGatherSingleStreamMesh()
{
}

//任务下发的具体执行逻辑
HcclResult AllGatherSingleStreamMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{

    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllGatherSingleStreamMesh run_async rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]", \
              rank, rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    if (rankSize == 1) {
        if (inputMem_ != outputMem_) {
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, outputMem_, inputMem_, stream_));
        }
        return HCCL_SUCCESS;
    }

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllGatherSingleStreamMesh][RunAsync]rank[%u] linkSize is less than rankSize", rank);
        return HCCL_E_INTERNAL;
    }

    u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherSingleStreamMesh][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }

    std::vector<Slice> inputSlices(slices_);
    if (slices_.size() == 0) {
        slices_.resize(rankSize);
        inputSlices.resize(rankSize);

        u64 sliceSize = count_ * unitSize;
        for (u32 i = 0; i < rankSize; i++) {
            slices_[i].size = sliceSize;
            slices_[i].offset = sliceSize * i;
            inputSlices[i].size = sliceSize;
            inputSlices[i].offset = 0;  // Input data is always at offset 0 for each rank
            HCCL_DEBUG("rank[%u], slices[%u].offset=%llu, slices[%u].size=%llu", \
                       rank, i, slices_[i].offset, i, slices_[i].size);
        }
    }

    // 如果input和output不一样，则先把input的数据拷贝到output的对应位置
    if (inputMem_ != outputMem_) {
        DeviceMem dst = outputMem_.range(slices_[rank].offset, slices_[rank].size);
        DeviceMem src = inputMem_.range(0, slices_[rank].size);  // Input is always at offset 0
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    }

    CHK_RET(RunAllgather(rank, rankSize, slices_, links));

    // if (barrierSwitchOn_) {
    //     // For non-ring algorithms, barrier is handled in algorithm implementation
    // }

    HCCL_INFO("AllGatherSingleStreamMesh finished: rank[%u] end", rank);
    return HCCL_SUCCESS;
}

// RunAllgather实现了Mesh AllGather算法的核心逻辑 由RunAsync调用
HcclResult AllGatherSingleStreamMesh::RunAllgather(u32 rank, u32 rankSize, const std::vector<Slice> &outputSlices, const std::vector<LINK> &links)
{
    if (outputSlices.size() < rankSize) {
        HCCL_ERROR("[Run][Allgather]rank[%u] OutputSlice Size is less than rank size", rank);
        return HCCL_E_INTERNAL;
    }
    HcclResult ret = HCCL_SUCCESS;

        u32 unitSize = DataUnitSize(dataType_);
    if (unitSize == 0) {
        HCCL_ERROR("[AllGatherSingleStreamMesh][RunAsync]unitSize is zero");
        return HCCL_E_INTERNAL;
    }
    u64 sliceSize = count_ * unitSize;

    //数据发送
    for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {
            if (peerRank == rank || peerRank >= links.size()) continue;
        
            // 非对称握手协议（基于rank ID顺序避免死锁）
            if (rank < peerRank) {
                    CHK_RET(links[peerRank]->TxAck(stream_));  
                    CHK_RET(links[peerRank]->RxAck(stream_));  
            } else {
                    // Higher rank ID: respond to handshake
                    CHK_RET(links[peerRank]->RxAck(stream_)); 
                    CHK_RET(links[peerRank]->TxAck(stream_));  
            }
        
            // 发送本rank的数据到peer的对应slice位置
            Slice mySlice = outputSlices[rank];
            CHK_RET(links[peerRank]->TxAsync(UserMemType::OUTPUT_MEM,
                mySlice.offset + baseOffset_, outputMem_.range(mySlice.offset, mySlice.size).ptr(), mySlice.size, stream_));
        
            //  接收peer rank的数据到本rank的对应slice位置  
            Slice peerSlice = outputSlices[peerRank];
            CHK_RET(links[peerRank]->RxAsync(UserMemType::OUTPUT_MEM,
                peerSlice.offset + baseOffset_, outputMem_.range(peerSlice.offset, peerSlice.size).ptr(), peerSlice.size, stream_));
    }
    
    // 确保所有发送和接收操作完成
    for (u32 peerRank = 0; peerRank < rankSize; peerRank++) {
            if (peerRank == rank || peerRank >= links.size()) continue;
        
            // 等待发送完成
            CHK_RET(links[peerRank]->TxWaitDone(stream_));
        
            // 等待接收完成
            CHK_RET(links[peerRank]->RxWaitDone(stream_));
    }

    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_GATHER_SINGLE_STREAM_MESH, AllGatherSingleStreamMesh);
}  // namespace hccl