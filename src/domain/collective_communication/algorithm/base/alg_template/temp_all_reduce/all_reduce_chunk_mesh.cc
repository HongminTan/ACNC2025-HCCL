/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_template_register.h"
#include "all_reduce_chunk_mesh.h"

namespace hccl {
AllReduceChunkMesh::AllReduceChunkMesh(const HcclDispatcher dispatcher) : AlgTemplateBase(dispatcher)
{
}

AllReduceChunkMesh::~AllReduceChunkMesh()
{
}

HcclResult AllReduceChunkMesh::Prepare(u64 reduceAttrBitMap, std::vector<Stream> &meshStreams,
    std::vector<std::shared_ptr<LocalNotify>> &meshSignal, std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
    u32 interRank, u32 interRankSize, u32 userRank, HcomCollOpInfo *opInfo)
{
    reduceAttr_ = reduceAttrBitMap;
    localRank_ = interRank;
    localRankSize_ = interRankSize;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignal_ = &meshSignal;
    meshSignalAux_ = &meshSignalAux;
    opInfo_ = opInfo;
    return HCCL_SUCCESS;
}
HcclResult AllReduceChunkMesh::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAux_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAux_)[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAux_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_, (*meshSignalAux_)[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignal_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignal_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignal_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, (*meshSignal_)[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

// 将数据均分，最小单位是128
HcclResult AllReduceChunkMesh::PrepareSlice(
    u64 dataCount, u32 unitSize, u32 sliceNum, std::vector<Slice> &dataSlice)
{
    Slice temp;
    u64 totalSize = dataCount * unitSize;
    dataSlice.clear();
    dataSlice.reserve(sliceNum);
    if (sliceNum == 0) {
        HCCL_ERROR("[Prepare][SliceData]data slice prepare, sliceNum is 0");
        return HCCL_E_PARA;
    }
    u64 sizePerSlice = (totalSize + sliceNum - 1) / sliceNum; /* 1是为了向上取整 */
    sizePerSlice = RoundUpWithDivisor(sizePerSlice, HCCL_MIN_SLICE_ALIGN);
    u64 residueSize = totalSize;
    u32 i = 0;
    while (residueSize > 0) {
        u64 sliceSize = sizePerSlice < residueSize ? sizePerSlice : residueSize;
        temp.size = sliceSize;
        temp.offset = totalSize - residueSize;
        i++;
        if (sliceSize <= 0) {
            HCCL_ERROR("[Prepare][SliceData]data_slice_prepare sliceSize[%llu]", sliceSize);
            return HCCL_E_PARA;
        }
        residueSize -= sliceSize;
        dataSlice.push_back(temp);
    }
    while (i < sliceNum) {
        temp.size = 0;
        temp.offset = totalSize;
        i++;
        dataSlice.push_back(temp);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::PrepareAllreduceSliceData()
{
    u32 unitSize = SIZE_TABLE[dataType_];
    HcclResult ret = HCCL_SUCCESS;
    CHK_RET(PrepareSlice(count_, unitSize, localRankSize_, slices_));
    for (u32 rank = 0; rank < localRankSize_; rank++) {
        std::vector<Slice> dataSegsSlice;
        ret = PrepareSlice(slices_[rank].size / unitSize, unitSize, localRankSize_ - 1, dataSegsSlice);
        sliceMap[rank] = dataSegsSlice;
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceChunkMesh][PrepareSlice]rank[%u] failed", rank), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HcclResult ret = HCCL_SUCCESS;
    CHK_SMART_PTR_NULL(dispatcher_);
    CHK_PTR_NULL(stream_.ptr());
    HCCL_INFO("AllReduceChunkMesh run: rank[%u] ranksize[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank,
        rankSize,
        inputMem_.ptr(),
        outputMem_.ptr(),
        count_);

    if (links.size() < rankSize) {
        HCCL_ERROR("[AllReduceChunkMesh][RunAsync]rank[%u] linksize[%llu] is less than rankSize[%u]",
            rank,
            links.size(),
            rankSize);
        return HCCL_E_INTERNAL;
    }

    // 如果ranksize为1, inline reduce和普通跨片reduce操作一致，从input->output
    if (rankSize == 1) {
        if (opInfo_->inputAddr != opInfo_->outputAddr) {
            DeviceMem userMemIn = DeviceMem::create(opInfo_->inputAddr, count_ * DataUnitSize(dataType_));
            DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * DataUnitSize(dataType_));
            ret = HcclD2DMemcpyAsync(dispatcher_, userMemOut, userMemIn, stream_);
            CHK_PRT_RET(
                ret != HCCL_SUCCESS, HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] memcpy async failed", rank), ret);
        }
        return ret;
    }

    ret = PrepareAllreduceSliceData();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in PrepareSliceData step",
            rank,
            count_),
        ret);

    ret = RunReduceScatter(rank, rankSize, links);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in reducescater "
                   "step",
            rank,
            count_),
        ret);

    ret = RunAllGather(rank, rankSize, links);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[AllReduceRing][RunAsync]rank[%u] count[%llu] failed in allgather "
                "step",
            rank,
            count_),
        ret);

    HCCL_INFO("AllReduceChunkMesh finished: rank[%u] ranksize[%u]", rank, rankSize);
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::RunReduceScatter(u32 rank, u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("ReduceScatterMeshAtomicOpbase run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank,
        rankSize,
        inputMem_.ptr(),
        outputMem_.ptr(),
        count_);

    // 数据准备
    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem commMemOut = outputMem_;

    DeviceMem src;
    DeviceMem dst;

    src = DeviceMem::create(static_cast<char *>(opInfo_->inputAddr), count_ * unitSize);

    if (commMemOut.ptr() == opInfo_-> outputAddr) {
        // 图模式
        src = src.range(slices_[rank].offset, slices_[rank].size);
        dst = commMemOut.range(slices_[rank].offset, slices_[rank].size);
    } else {
        // 单算子
        src = src.range(0, count_ * unitSize);
        dst = commMemOut.range(0, count_ * unitSize);
    }
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));

    DeviceMem emptySrc = commMemOut.range(0, 0);
    DeviceMem emptyDst = commMemOut.range(0, 0);

    // 主从流之前加空拷贝 防止成环
    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = (round + rank) % rankSize;
        Stream &subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    for (u32 round = 1; round < rankSize; round++) {
        // 主从流同步

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

        CHK_RET(SubRecordMain());
        CHK_RET(MainWaitSub());

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

        CHK_RET(MainRecordSub());
        CHK_RET(SubWaitMain());

        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

        // 跨片reduceinline写
        for (u32 peer = 1; peer < rankSize; peer++) {
            u32 gap = (peer + round) > rankSize ? (peer + round - 1)%(rankSize -1):(peer + round - 1);
            u32 dstRank = (gap + rank) % rankSize;
            Stream &subStream = (peer == localRankSize_ - 1) ? stream_ : meshStreams_[peer - 1];
            u32 dstSlice = peer - 1;
            void *remMemPtr = nullptr;

            if (commMemOut.ptr() == opInfo_-> outputAddr) {
                CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::INPUT_MEM, &remMemPtr));
            } else {
                CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
            }

            src = DeviceMem::create(
                static_cast<char *>(remMemPtr) + slices_[rank].offset + sliceMap[rank][dstSlice].offset,
                sliceMap[rank][dstSlice].size);
            dst = commMemOut.range(
                slices_[rank].offset + sliceMap[rank][dstSlice].offset, sliceMap[rank][dstSlice].size);
            CHK_RET(HcclReduceAsync(dispatcher_, static_cast<void *>(src.ptr()),
                sliceMap[rank][dstSlice].size / unitSize,
                dataType_,
                reductionOp_,
                subStream,
                static_cast<void *>(dst.ptr()),
                links[dstRank]->GetRemoteRank(),
                links[dstRank]->GetLinkType(), INLINE_REDUCE_BIT));
        }
    }

    for (u32 round = 1; round < rankSize; round++) {
        u32 gap = (round - 1) == 0 ? (rankSize - 1) : (round - 1);
        u32 dstRank = (rank + gap) % rankSize;
        Stream &subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));
    return HCCL_SUCCESS;
}

HcclResult AllReduceChunkMesh::RunAllGather(u32 rank, u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("AllGatherMesh run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]",
        rank,
        rankSize,
        inputMem_.ptr(),
        outputMem_.ptr(),
        count_);
    u32 unitSize = DataUnitSize(dataType_);

    DeviceMem userMemOut = DeviceMem::create(opInfo_->outputAddr, count_ * unitSize);
    DeviceMem commMemOut = DeviceMem::create(outputMem_.ptr(), outputMem_.size());

    DeviceMem emptySrc = userMemOut.range(0, 0);
    DeviceMem emptyDst = commMemOut.range(0, 0);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream &subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        CHK_RET(links[dstRank]->TxAck(subStream));
        CHK_RET(links[dstRank]->RxAck(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    DeviceMem src;
    DeviceMem dst;
    if (opInfo_->outputAddr != outputMem_.ptr()) {
        dst = userMemOut.range(slices_[rank].offset, slices_[rank].size);
        src = commMemOut.range(slices_[rank].offset, slices_[rank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, meshStreams_[meshStreams_.size()-1]));
    }

    for (u32 round = 1; round < rankSize; round++) {
        u32 dstRank = BackwardRank(rank, rankSize, round);
        Stream &subStream = (round == localRankSize_ - 1) ? stream_ : meshStreams_[round - 1];
        void *remMemPtr = nullptr;
        CHK_RET(links[dstRank]->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        src = DeviceMem::create(static_cast<char *>(remMemPtr) + slices_[dstRank].offset, slices_[dstRank].size);
        dst = userMemOut.range(slices_[dstRank].offset, slices_[dstRank].size);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, subStream,
            links[dstRank]->GetRemoteRank(), links[dstRank]->GetLinkType()));

        CHK_RET(links[dstRank]->TxDataSignal(subStream));
        CHK_RET(links[dstRank]->RxDataSignal(subStream));
    }

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, emptyDst, emptySrc, stream_));

    HCCL_INFO("[AllGatherMesh] finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_REDUCE_CHUNK_MESH, AllReduceChunkMesh);
}  // namespace hccl
