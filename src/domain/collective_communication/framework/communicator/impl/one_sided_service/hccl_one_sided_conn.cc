/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_conn.h"
#include "sal_pub.h"

namespace hccl {
using namespace std;

HcclOneSidedConn::HcclOneSidedConn(const HcclNetDevCtx &netDevCtx, const HcclRankLinkInfo &localRankInfo,
    const HcclRankLinkInfo &remoteRankInfo, std::unique_ptr<HcclSocketManager> &socketManager,
    std::unique_ptr<NotifyPool> &notifyPool, const HcclDispatcher &dispatcher,
    const bool &useRdma, u32 sdid, u32 serverId, u32 trafficClass, u32 serviceLevel)
    : localRankInfo_(localRankInfo), socketManager_(socketManager),  notifyPool_(notifyPool)
{
    netDevCtx_ = netDevCtx;
    remoteRankInfo_ = remoteRankInfo;
    useRdma_ = useRdma;
    TransportMem::AttrInfo attrInfo = {localRankInfo.userRank, remoteRankInfo.userRank, sdid, serverId, trafficClass, serviceLevel};
    if (useRdma) {
        transportMemPtr_ = TransportMem::Create(TransportMem::TpType::ROCE, notifyPool, netDevCtx, 
            dispatcher, attrInfo);
    } else {
        transportMemPtr_ = TransportMem::Create(TransportMem::TpType::IPC, notifyPool, netDevCtx,
            dispatcher, attrInfo);
    }
    CHK_SMART_PTR_RET_NULL(transportMemPtr_);
}

HcclOneSidedConn::~HcclOneSidedConn()
{
}

HcclResult HcclOneSidedConn::Connect(const std::string &commIdentifier, s32 timeoutSec)
{
    // 创建socket用于交换数据
    std::string newTag;
    if (localRankInfo_.userRank < remoteRankInfo_.userRank) {
        // 本端为SERVER，对端为CLIENT
        newTag = string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_" +
            string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_" + commIdentifier;
    } else {
        newTag = string(remoteRankInfo_.ip.GetReadableIP()) + "_" + to_string(remoteRankInfo_.port) + "_" +
            string(localRankInfo_.ip.GetReadableIP()) + "_" + to_string(localRankInfo_.port) + "_" + commIdentifier;
    }
    HCCL_DEBUG("[HcclOneSidedConn][Connect]socket tag:%s", newTag.c_str());
    std::vector<std::shared_ptr<HcclSocket>> connectSockets;
    CHK_RET(socketManager_->CreateSingleLinkSocket(newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true));
    CHK_RET(transportMemPtr_->SetDataSocket(connectSockets[0]));
    socket_ = connectSockets[0];

    if (useRdma_) {
        // 创建socket用于QP建链
        newTag += "_QP";
        CHK_RET(socketManager_->CreateSingleLinkSocket(newTag, netDevCtx_, remoteRankInfo_, connectSockets, true, true));
        CHK_RET(transportMemPtr_->SetSocket(connectSockets[0]));
        // Transport建链：notify资源创建+QP建链
        CHK_RET(transportMemPtr_->Connect(timeoutSec));
    }
    
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeIpcProcessInfo(const ProcessInfo &localProcess, ProcessInfo &remoteProcess)
{
    HCCL_DEBUG("[HcclOneSidedConn][ExchangeIpcProcessInfo] localRank[%u] exchange process info", localRankInfo_.userRank);
    if (socket_->GetLocalRole() == HcclSocketRole::SOCKET_ROLE_CLIENT) {
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
    } else {
        CHK_RET(socket_->Send(&localProcess, sizeof(ProcessInfo)));
        CHK_RET(socket_->Recv(&remoteProcess, sizeof(ProcessInfo)));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedConn::ExchangeMemDesc(const HcclMemDescs &localMemDescs, HcclMemDescs &remoteMemDescs,
    u32 &actualNumOfRemote)
{
    TransportMem::RmaMemDesc *localMemDescArray = static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(localMemDescs.array));
    TransportMem::RmaMemDescs localRmaMemDescs = {localMemDescArray, localMemDescs.arrayLength};
    TransportMem::RmaMemDesc *remoteMemDescArray = static_cast<TransportMem::RmaMemDesc *>(static_cast<void *>(remoteMemDescs.array));
    TransportMem::RmaMemDescs remoteRmaMemDescs = {remoteMemDescArray, remoteMemDescs.arrayLength};

    return transportMemPtr_->ExchangeMemDesc(
        localRmaMemDescs, remoteRmaMemDescs, actualNumOfRemote);
}

void HcclOneSidedConn::EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem)
{
    const TransportMem::RmaMemDesc *remoteRmaMemDesc = static_cast<const TransportMem::RmaMemDesc *>(static_cast<const void *>(remoteMemDesc.desc));
    TransportMem::RmaMem remoteRmaMem;
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->EnableMemAccess(*remoteRmaMemDesc, remoteRmaMem),
        "[HcclOneSidedConn][EnableMemAccess] Enable memory access failed.");
    remoteMem.type = static_cast<HcclMemType>(remoteRmaMem.type);
    remoteMem.addr = remoteRmaMem.addr;
    remoteMem.size = remoteRmaMem.size;
}

void HcclOneSidedConn::DisableMemAccess(const HcclMemDesc &remoteMemDesc)
{
    const TransportMem::RmaMemDesc *remoteRmaMemDesc = static_cast<const TransportMem::RmaMemDesc *>(static_cast<const void *>(remoteMemDesc.desc));
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->DisableMemAccess(*remoteRmaMemDesc),
                           "[HcclOneSidedConn][DisableMemAccess] disable memory access failed.");
}

void HcclOneSidedConn::BatchWrite(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchWrite] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchWrite] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        TransportMem::RmaOpMem remoteMem = {oneSideDescs[i].remoteAddr, byteSize};
        TransportMem::RmaOpMem localMem = {oneSideDescs[i].localAddr, byteSize};
        EXCEPTION_THROW_IF_ERR(transportMemPtr_->Write(remoteMem, localMem, stream),
            "[HcclOneSidedConn][BatchWrite] transportMem WriteAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchWrite] AddOpFence failed.");
}

void HcclOneSidedConn::BatchRead(const HcclOneSideOpDesc* oneSideDescs, u32 descNum, const rtStream_t& stream)
{
    for (u32 i = 0; i < descNum; i++) {
        if (oneSideDescs[i].count == 0) {
            HCCL_WARNING("[HcclOneSidedConn][BatchRead] Desc item[%u] count is 0.", i);
        }
        u32 unitSize;
        EXCEPTION_THROW_IF_ERR(SalGetDataTypeSize(oneSideDescs[i].dataType, unitSize),
            "[HcclOneSidedConn][BatchRead] Get dataType size failed!");
        u64 byteSize = oneSideDescs[i].count * unitSize;
        TransportMem::RmaOpMem remoteMem = {oneSideDescs[i].remoteAddr, byteSize};
        TransportMem::RmaOpMem localMem = {oneSideDescs[i].localAddr, byteSize};
        EXCEPTION_THROW_IF_ERR(transportMemPtr_->Read(localMem, remoteMem, stream),
            "[HcclOneSidedConn][BatchRead] transportMem ReadAsync failed.");
    }
    EXCEPTION_THROW_IF_ERR(transportMemPtr_->AddOpFence(stream), "[HcclOneSidedConn][BatchRead] AddOpFence failed.");
}

}