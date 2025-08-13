/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_service.h"
#include "device_capacity.h"
#include "sal_pub.h"

namespace hccl {
using namespace std;

HcclOneSidedService::HcclOneSidedService(unique_ptr<HcclSocketManager> &socketManager,
    unique_ptr<NotifyPool> &notifyPool)
    : IHcclOneSidedService(socketManager, notifyPool)
{
}

HcclOneSidedService::~HcclOneSidedService()
{
    HCCL_RUN_INFO("[~HcclOneSidedService] localRankId[%u] has registedMemCnt[%u] mem didn't dereg",
                localRankInfo_.userRank, registedMemCnt_);
    HcclResult ret = HCCL_SUCCESS;
    for (auto it = desc2HcclBufMapIpc_.begin(); it != desc2HcclBufMapIpc_.end(); ++it) {
        HcclBuf &buf = it->second;
        do {
            ret = HcclMemDereg(&buf);  // 需循环调用DeregMem来去注册内存(因为存在一块内存多次Reg的情况)
            // 失败场景记录log即可，接着处理后面的mem
            CHK_PRT_CONT(((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)),
                HCCL_ERROR("[~HcclOneSidedService] DeregMem IPC localRankId[%u] addr[%p] size[%lu] failed",
                    localRankInfo_.userRank, buf.addr, buf.len));
        } while (ret == HCCL_E_AGAIN);
    }

    for (auto it = desc2HcclBufMapRoce_.begin(); it != desc2HcclBufMapRoce_.end(); ++it) {
        HcclBuf &buf = it->second;
        do {
            ret = HcclMemDereg(&buf);  // 需循环调用DeregMem来去注册内存(因为存在一块内存多次Reg的情况)
            // 失败场景记录log即可，接着处理后面的mem
            CHK_PRT_CONT(((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)),
                HCCL_ERROR("[~HcclOneSidedService] DeregMem ROCE localRankId[%u] addr[%p] size[%lu] failed",
                    localRankInfo_.userRank, buf.addr, buf.len));
        } while (ret == HCCL_E_AGAIN);
    }
}

HcclResult HcclOneSidedService::IsUsedRdma(RankId remoteRankId, bool &useRdma)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));

    RankInfo_t localRankInfo = (rankTable_->rankList).at(localRankInfo_.userRank);
    RankInfo_t remoteRankInfo = (rankTable_->rankList).at(remoteRankId);
    if (deviceType == DevType::DEV_TYPE_910B) {
        // 外部使能RDMA，或者节点间通信
        if (GetExternalInputIntraRoceSwitch() || localRankInfo.serverId != remoteRankInfo.serverId) {
            useRdma = true;
            return HCCL_SUCCESS;
        }

        // 同一节点的 PCIe 连接判断
        s32 localDeviceId = localRankInfo_.devicePhyId;
        s32 remoteDeviceId = remoteRankInfo.deviceInfo.devicePhyId;
        LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
        CHK_RET(hrtGetPairDeviceLinkType(static_cast<u32>(localDeviceId), static_cast<u32>(remoteDeviceId), linkType));
        if (linkType != LinkTypeInServer::HCCS_TYPE) {
            HCCL_ERROR("[HcclOneSidedService][IsUsedRdma]localDeviceId: %d, remoteDeviceId: %d, linkType %u is not supported",
                localDeviceId, remoteDeviceId, linkType);
            return HCCL_E_NOT_SUPPORT;
        }

        // 节点内通信，默认不使用 RDMA
        useRdma = false;
        return HCCL_SUCCESS;
    } else if (deviceType == DevType::DEV_TYPE_910_93) {
        if (GetExternalInputIntraRoceSwitch() || localRankInfo.superPodId != remoteRankInfo.superPodId) {
            useRdma = true;
            return HCCL_SUCCESS;
        }

        useRdma = false;
        return HCCL_SUCCESS;
    }

    // 其他情况默认使用 RDMA
    useRdma = true;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::GetIsUsedRdma(RankId remoteRankId, bool &useRdma)
{
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    } else {
        useRdma = isUsedRdmaMap_[remoteRankId];
    }

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ReMapMem(HcclMem *memInfoArray, u64 arraySize)
{
    HcclResult ret = HCCL_SUCCESS;
    if (netDevRdmaCtx_) {  // 非roce场景不进行remap，返回success
        ret = HcclMemRemap(netDevRdmaCtx_, memInfoArray, arraySize);
    } else {
        HCCL_RUN_INFO("[HcclOneSidedService][ReMapMem] doesn't support remap ipc mem, just return success");
    }
    return ret;
}

HcclResult HcclOneSidedService::RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId,
    HcclMemDesc &localMemDesc)
{
    constexpr u32 maxRegistedMem = 256;
    if (registedMemCnt_ >= maxRegistedMem) {
        HCCL_ERROR("[HcclOneSidedService][RegMem]The number of registered memory "\
            "exceeds the upper limit[%u]", maxRegistedMem);
        return HCCL_E_UNAVAIL;
    }

    bool useRdma = true;
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    }
    useRdma = isUsedRdmaMap_[remoteRankId];

    HcclMem localMem{type, addr, size};
    HcclBuf buf;
    HcclResult ret = HcclMemReg(useRdma ? netDevRdmaCtx_ : netDevIpcCtx_, &localMem, &buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) {  // HCCL_E_AGAIN:调用HcclMemReg前，内存已注册过
        return ret;
    }
    bool firstReg = (ret == HCCL_SUCCESS);

    char *desc = nullptr;
    uint64_t descLen = 0;
    ret = HcclMemGetDesc(&buf, &desc, &descLen);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclOneSidedService][RegMem] get mem desc fialed, ret[%d]", ret);
        throw logic_error("[HcclOneSidedService][RegMem] get mem desc fialed");
    }

    HcclMemDescData *ptr = static_cast<HcclMemDescData *>(static_cast<void *>(localMemDesc.desc));
    ptr->localRankId = localRankInfo_.userRank;
    ptr->remoteRankId = remoteRankId;
    memset_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, 0, HCCL_MEM_DESC_STR_LEN);
    if (memcpy_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, desc, descLen + 1) != EOK) {
        HCCL_ERROR("[HcclOneSidedService][RegMem] memcpy_s memDesc failed");
        return HCCL_E_INTERNAL;
    }

    if (firstReg) {
        registedMemCnt_++;
        std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
        if (useRdma) {
            desc2HcclBufMapRoce_.emplace(descStr, buf);
        } else {
            desc2HcclBufMapIpc_.emplace(descStr, buf);
        }
    }
    HCCL_DEBUG("[HcclOneSidedService][RegMem] localRankId[%u] remoteRankId[%u] size[%lu] useRdma[%d] "
        "desc2HcclBufMap[%u] registedMemCnt[%u]",
        ptr->localRankId, ptr->remoteRankId, size, useRdma,
        useRdma ? desc2HcclBufMapRoce_.size() : desc2HcclBufMapIpc_.size(), registedMemCnt_);
    return HCCL_SUCCESS;
}

HcclBuf *HcclOneSidedService::GetHcclBufByDesc(std::string &descStr, bool useRdma)
{
    HcclBuf *buf = nullptr;
    if (useRdma) {
        auto iter = desc2HcclBufMapRoce_.find(descStr);
        if (iter == desc2HcclBufMapRoce_.end()) {
            HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]Roce memory is not registered, please register first.");
            return nullptr;
        }
        buf = &(iter->second);
    } else {
        auto iter = desc2HcclBufMapIpc_.find(descStr);
        if (iter == desc2HcclBufMapIpc_.end()) {
            HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]Ipc memory is not registered, please register first.");
            return nullptr;
        }
        buf = &(iter->second);
    }
    return buf;
}

HcclResult HcclOneSidedService::DeregMem(const HcclMemDesc &localMemDesc)
{
    const HcclMemDescData *ptr = static_cast<const HcclMemDescData *>(static_cast<const void *>(localMemDesc.desc));
    u32 remoteRankId = ptr->remoteRankId;
    if (registedMemCnt_ == 0) {
        HCCL_ERROR("[HcclOneSidedService][DeregMem]The number of registered memory is 0, please register first.");
        return HCCL_E_NOT_FOUND;
    }

    bool useRdma = true;
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    }
    useRdma = isUsedRdmaMap_[remoteRankId];

    std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
    HcclBuf *buf = GetHcclBufByDesc(descStr, useRdma);
    CHK_PRT_RET(buf == nullptr, HCCL_ERROR("[HcclOneSidedService][DeregMem] GetHcclBufByDesc failed."), HCCL_E_INTERNAL);
    HcclResult ret = HcclMemDereg(buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) {  // 调用DeregMem后，去注册的内存还需继续使用（即有多次注册
        return ret;
    }

    HCCL_DEBUG("[HcclOneSidedService][DeregMem] localRankId[%u] remoteRankId[%u] size[%lu] useRdma[%d] "
        "desc2HcclBufMap[%u] registedMemCnt[%u]",
        ptr->localRankId, ptr->remoteRankId, buf->len, useRdma,
        useRdma ? desc2HcclBufMapRoce_.size() : desc2HcclBufMapIpc_.size(), registedMemCnt_);

    if (ret == HCCL_SUCCESS) {
        registedMemCnt_--;
        if (useRdma) {
            desc2HcclBufMapRoce_.erase(descStr);
        } else {
            desc2HcclBufMapIpc_.erase(descStr);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::SetupRemoteRankInfo(RankId remoteRankId, HcclRankLinkInfo &remoteRankInfo)
{
    // 检查 rankId 是否有效
    CHK_PRT_RET(rankTable_->rankList.size() <= remoteRankId,
        HCCL_ERROR("[HcclOneSidedService][SetupRemoteRankInfo] the size of rankList is less than remoteRankId[%u].",
            remoteRankId), HCCL_E_NOT_FOUND);

    RankInfo_t tempRankInfo = rankTable_->rankList.at(remoteRankId);
    remoteRankInfo.userRank = tempRankInfo.rankId;
    remoteRankInfo.devicePhyId = tempRankInfo.deviceInfo.devicePhyId;

    // 检查 deviceIp 是否为空
    CHK_PRT_RET(tempRankInfo.deviceInfo.deviceIp.empty(),
        HCCL_ERROR("[HcclOneSidedService][SetupRemoteRankInfo] deviceIp is empty. RemoteRankId is [%u]",
            remoteRankId), HCCL_E_NOT_FOUND);
    remoteRankInfo.ip = tempRankInfo.deviceInfo.deviceIp[0];

    if (isUsedRdmaMap_.find(remoteRankId) != isUsedRdmaMap_.end() && !isUsedRdmaMap_[remoteRankId]) {
        bool useSuperPodMode = false;
        CHK_RET(IsSuperPodMode(useSuperPodMode));

        HcclIpAddress localVnicIp = HcclIpAddress(localRankInfo_.devicePhyId);
        HcclIpAddress remoteVnicIp = HcclIpAddress(remoteRankInfo.devicePhyId);
        RankInfo_t tRankInfo = rankTable_->rankList.at(localRankInfo_.userRank);

        if (useSuperPodMode) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_SDID,
                tRankInfo.superDeviceId, localVnicIp));
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_SDID,
                tempRankInfo.superDeviceId, remoteVnicIp));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                localRankInfo_.devicePhyId, localVnicIp));
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID,
                remoteRankInfo.devicePhyId, remoteVnicIp));
        }

        localRankVnicInfo_.ip = localVnicIp;
        remoteRankInfo.ip = remoteVnicIp;
    }
    remoteRankInfo.port = tempRankInfo.deviceInfo.port == 0 || tempRankInfo.deviceInfo.port == HCCL_INVALID_PORT ?
        HETEROG_CCL_PORT : tempRankInfo.deviceInfo.port;
    remoteRankInfo.socketsPerLink = 1;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::CreateConnection(RankId remoteRankId, const HcclRankLinkInfo &remoteRankInfo,
    std::shared_ptr<HcclOneSidedConn> &tempConn)
{
    HcclNetDevCtx *ctx = isUsedRdmaMap_.at(remoteRankId) ? &netDevRdmaCtx_ : &netDevIpcCtx_;
    HcclRankLinkInfo *rankInfo = isUsedRdmaMap_.at(remoteRankId) ? &localRankInfo_ : &localRankVnicInfo_;
    u32 sdid = isUsedRdmaMap_.at(remoteRankId) ? 0 : rankTable_->rankList.at(localRankInfo_.userRank).superDeviceId;
    u32 serverId = isUsedRdmaMap_.at(remoteRankId) ? 0 : rankTable_->rankList.at(localRankInfo_.userRank).serverIdx;
    EXECEPTION_CATCH(tempConn = std::make_shared<HcclOneSidedConn>(*ctx, *rankInfo, remoteRankInfo,
        socketManager_, notifyPool_, dispatcher_,
        isUsedRdmaMap_[remoteRankId], sdid, serverId, trafficClass_, serviceLevel_), return HCCL_E_PTR);

    CHK_SMART_PTR_NULL(tempConn);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::Grant(const HcclMemDesc &localMemDesc, const ProcessInfo &remoteProcess)
{
    const HcclMemDescData *ptr = static_cast<const HcclMemDescData *>(static_cast<const void *>(localMemDesc.desc));
    std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
    HCCL_DEBUG("[HcclOneSidedService][Grant] desc[%s] length[%u]", descStr.c_str(), descStr.length());
    HcclBuf *buf = GetHcclBufByDesc(descStr, false);
    if (buf == nullptr) {
        return HCCL_E_INTERNAL;
    }

    HcclMemGrantInfo grantInfo = {remoteProcess.sdid, static_cast<int32_t>(remoteProcess.pid)};
    HcclResult ret = HcclMemGrant(buf, &grantInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclOneSidedService][Grant] Grant error"), ret);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ExchangeMemDesc(RankId remoteRankId, const HcclMemDescs &localMemDescs,
    HcclMemDescs &remoteMemDescs, u32 &actualNumOfRemote, const std::string &commIdentifier, s32 timeoutSec)
{
    std::shared_ptr<HcclOneSidedConn> tempConn;
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HcclRankLinkInfo remoteRankInfo;
        CHK_RET(SetupRemoteRankInfo(remoteRankId, remoteRankInfo));
        CHK_RET(CreateConnection(remoteRankId, remoteRankInfo, tempConn));
        CHK_RET(tempConn->Connect(commIdentifier, timeoutSec));
        oneSidedConns_.emplace(remoteRankId, tempConn);
    } else {
        tempConn = it->second;
    }

    // HCCS下进行权限授予
    if (!isUsedRdmaMap_[remoteRankId]) {
        u32 pid;
        SalGetBareTgid(&pid);
        RankId localRankId = localRankInfo_.userRank;
        u32 sid = rankTable_->rankList.at(localRankId).superDeviceId;
        u32 serverId = rankTable_->rankList.at(localRankId).serverIdx;

        // 收发进程信息
        ProcessInfo localProcess = {pid, sid, serverId};
        ProcessInfo remoteProcess = {0};
        CHK_RET(tempConn->ExchangeIpcProcessInfo(localProcess, remoteProcess));
        remoteProcess.sdid = localProcess.serverId == remoteProcess.serverId ? INVALID_INT : remoteProcess.sdid;

        for (u32 i = 0; i < localMemDescs.arrayLength; ++i) {
            CHK_RET(Grant(localMemDescs.array[i], remoteProcess));
        }
    }

    return tempConn->ExchangeMemDesc(localMemDescs, remoteMemDescs, actualNumOfRemote);
}

void HcclOneSidedService::EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem)
{
    const TransportMem::RmaMemDesc* ptr = reinterpret_cast<const TransportMem::RmaMemDesc*>(remoteMemDesc.desc);
    u32 remoteRank = ptr->localRankId;
    if (oneSidedConns_.find(remoteRank) == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclOneSidedService][EnableMemAccess]connection not found, remoteRank[%u], "\
            "please exchange mem desc to create connection first.", remoteRank);
        throw logic_error("[HcclOneSidedService][EnableMemAccess]connection not found.");
    }
    oneSidedConns_.at(remoteRank)->EnableMemAccess(remoteMemDesc, remoteMem);
}

void HcclOneSidedService::DisableMemAccess(const HcclMemDesc &remoteMemDesc)
{
    const TransportMem::RmaMemDesc* ptr = reinterpret_cast<const TransportMem::RmaMemDesc*>(remoteMemDesc.desc);
    u32 remoteRank = ptr->localRankId;
    if (oneSidedConns_.find(remoteRank) == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclOneSidedService][DisableMemAccess]connection not found by remoteRankId[%u], "\
            "please exchange mem desc to create connection first.", remoteRank);
        throw logic_error("[HcclOneSidedService][DisableMemAccess]connection not found.");
    }
    oneSidedConns_.at(remoteRank)->DisableMemAccess(remoteMemDesc);
}

void HcclOneSidedService::BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum,
    const rtStream_t &stream)
{
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchPut] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    it->second->BatchWrite(desc, descNum, stream);
}

void HcclOneSidedService::BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum,
    const rtStream_t &stream)
{
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchGet] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    it->second->BatchRead(desc, descNum, stream);
}
}