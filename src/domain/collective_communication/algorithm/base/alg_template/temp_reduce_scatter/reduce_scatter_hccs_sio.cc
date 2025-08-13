/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "reduce_scatter_hccs_sio.h"
#include "alg_template_register.h"
#include "externalinput_pub.h"

namespace hccl {
using namespace std;

ReduceScatterHccsSio::ReduceScatterHccsSio(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{}

ReduceScatterHccsSio::~ReduceScatterHccsSio() {}

HcclResult ReduceScatterHccsSio::Prepare(DeviceMem &inputMem, DeviceMem &outputMem, DeviceMem &scratchMem,
                                            const u64 count, const HcclDataType dataType, const Stream &stream,
                                            const HcclReduceOp reductionOp, const u32 root,
                                            const u64 baseOffset,
                                            const u64 reduceAttrBitMap, std::vector<Stream> &meshStreams,
                                            std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
                                            std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux,
                                            u32 userRank, SubCommInfo subCommInfoHccs, SubCommInfo subCommInfoSio)
{
    reduceAttr_ = reduceAttrBitMap;
    userRank_ = userRank;
    meshStreams_ = meshStreams;
    meshSignalPtr_ = &meshSignal;
    meshSignalAuxPtr_ = &meshSignalAux;
    subCommInfoHccs_ = subCommInfoHccs;
    subCommInfoSio_ = subCommInfoSio;
    std::vector<Slice> slices;
    return AlgTemplateBase::Prepare(inputMem, outputMem, scratchMem, count, dataType, stream, reductionOp,
        root, slices, baseOffset);
}

HcclResult ReduceScatterHccsSio::MainRecordSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalAuxPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Post(stream_, dispatcher_, (*meshSignalAuxPtr_)[signalIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHccsSio::SubWaitMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalAuxPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Wait(meshStreams_[streamIndex], dispatcher_,
            (*meshSignalAuxPtr_)[streamIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHccsSio::MainWaitSub()
{
    for (u32 signalIndex = 0; signalIndex < meshSignalPtr_->size(); signalIndex++) {
        CHK_RET(LocalNotify::Wait(stream_, dispatcher_, (*meshSignalPtr_)[signalIndex], profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHccsSio::SubRecordMain()
{
    for (u32 streamIndex = 0; streamIndex < meshSignalPtr_->size(); streamIndex++) {
        CHK_RET(LocalNotify::Post(meshStreams_[streamIndex], dispatcher_, (*meshSignalPtr_)[streamIndex],
            profilerInput_.stage));
    }
    return HCCL_SUCCESS;
}

HcclResult ReduceScatterHccsSio::RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links)
{
    HCCL_INFO("ReduceScatterHccsSio run: rank[%u] totalrank[%u] inputMem[%p] outputMem[%p] count[%llu]", rank,
        rankSize, inputMem_.ptr(), outputMem_.ptr(), count_);

    // 数据准备
    u32 unitSize = SIZE_TABLE[dataType_];
    
    DeviceMem src;
    DeviceMem dst;

    u32 dstRank = rank + (rank % 2 == 0 ? (1) : (-1));
    src = DeviceMem::create(static_cast<u8 *>(inputMem_.ptr()) + rank * count_ * unitSize, count_ * unitSize);
    dst = DeviceMem::create(static_cast<u8 *>(outputMem_.ptr()), count_ * unitSize);

    CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, stream_));
    
    CHK_RET(MainRecordSub());
    CHK_RET(SubWaitMain());

    // 每个stream只负责一个对端的交互
    
        const LINK &dstLinkHccs = subCommInfoHccs_.links[dstRank];
        const LINK &dstLinkSio = subCommInfoSio_.links[dstRank];
        CHK_RET(dstLinkHccs->TxAck(meshStreams_[0]));
        CHK_RET(dstLinkHccs->RxAck(meshStreams_[0]));
        CHK_RET(dstLinkSio->TxAck(meshStreams_[1]));
        CHK_RET(dstLinkSio->RxAck(meshStreams_[1]));
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    CHK_RET(SubWaitMain());
    CHK_RET(MainRecordSub());

    // inline执行notice reduce
        
        // 本rank要发数据
        void *remMemPtr = nullptr;
        // 获取远端的commoutMem
        CHK_RET(dstLinkHccs->GetRemoteMem(UserMemType::OUTPUT_MEM, &remMemPtr));
        static u32 HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR = 2;
        dst = DeviceMem::create(static_cast<u8 *>(remMemPtr), count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR * unitSize);
        src = DeviceMem::create(static_cast<u8 *>(inputMem_.ptr()) + dstRank * count_ * unitSize,
            count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR * unitSize);
        // HCCL_ERROR("ReduceScatterHccsSio before hcclReduceAsync, src addr[%p], dst addr[%p], count[%llu], dataType[%s]",
        // static_cast<void *>(src.ptr()), static_cast<void *>(dst.ptr()), count_ / 2, GetDataTypeEnumStr(dataType_).c_str());
        //通过hccs链路写给对端的usrout
        CHK_RET(HcclReduceAsync(dispatcher_,
            static_cast<void *>(src.ptr()),
            count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR,
            dataType_,
            reductionOp_,
            meshStreams_[0],
            static_cast<void *>(dst.ptr()),
            dstLinkHccs->GetRemoteRank(),
            dstLinkHccs->GetLinkType(),
            INLINE_REDUCE_BIT));

        //totalCount_ - totalCount_ / DIVISOR_NUM_TWO
        dst = DeviceMem::create(static_cast<u8 *>(remMemPtr) + count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR * unitSize,
                (count_ - count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR) * unitSize);
        src = DeviceMem::create(static_cast<u8 *>(inputMem_.ptr()) + dstRank * count_ * unitSize +
                                    count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR * unitSize,
            (count_ - count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR) * unitSize);
        // 通过sio链路写给对端的usrout
        CHK_RET(HcclReduceAsync(dispatcher_,
            static_cast<void *>(src.ptr()),
            count_ - count_ / HCCL_REDUCE_SCATTER_SIO_SPLIT_FACTOR,
            dataType_,
            reductionOp_,
            meshStreams_[1],
            static_cast<void *>(dst.ptr()),
            dstLinkSio->GetRemoteRank(),
            dstLinkSio->GetLinkType(),
            INLINE_REDUCE_BIT));

        CHK_RET(dstLinkHccs->TxDataSignal(meshStreams_[0]));
        CHK_RET(dstLinkHccs->RxDataSignal(meshStreams_[0]));
        CHK_RET(dstLinkSio->TxDataSignal(meshStreams_[1]));
        CHK_RET(dstLinkSio->RxDataSignal(meshStreams_[1]));
    
    CHK_RET(SubRecordMain());
    CHK_RET(MainWaitSub());

    HCCL_INFO("ReduceScatterHccsSio finished: rank[%u]", rank);
    return HCCL_SUCCESS;
}
REGISTER_TEMPLATE(TemplateType::TEMPLATE_REDUCESCATTER_HCCS_SIO, ReduceScatterHccsSio);
}