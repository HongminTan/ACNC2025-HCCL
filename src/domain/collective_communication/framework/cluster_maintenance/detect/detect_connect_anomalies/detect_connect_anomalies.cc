/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_communicator.h"
#include "detect_connect_anomalies.h"
#include "hccl_socket_manager.h"
#include "hccl_socket.h"
#include "env_config.h"
#include <chrono>
#include <thread>


namespace hccl {
DetectConnectionAnomalies &DetectConnectionAnomalies::GetInstance(s32 deviceLogicID)
{
    static DetectConnectionAnomalies dca[MAX_MODULE_DEVICE_NUM];
    if (static_cast<u32>(deviceLogicID) >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING("[DetectConnectionAnomalies][GetInstance]deviceLogicID[%d] is invalid", deviceLogicID);
        return dca[0];
    }
    return dca[deviceLogicID];
}

// 创建单例，保存所有RankInfoList中的Ip地址
void DetectConnectionAnomalies::Init(std::vector<RankInfo> &rankInfos, bool isNeedNic)
{
    if (isNeedNic) {
        isNeedNic_ =  isNeedNic;
    }
    // 直接用set保存，省掉查重
    for (auto &rankInfo : rankInfos) {
        if (!rankInfo.nicIp[0].IsInvalid()) {
            uniqueIps_.insert(rankInfo.nicIp[0]);
        }

        if (!rankInfo.deviceVnicIp.IsInvalid()) {
            uniqueIps_.insert(rankInfo.deviceVnicIp);
        }
    }
    return;
}

// 添加ipQueue
void DetectConnectionAnomalies::AddIpQueue(RankInfo &localRankInfo, RankInfo &remoteRankInfo, NicType nicType)
{
    // 检查是否需要进行连接异常检测
    if (GetExternalInputDfsConnectionFaultDetctionTime() == 0 || !threadExit_) {
        HCCL_RUN_INFO("[Add][IpQueue] GetExternalInputDfsConnectionFaultDetctionTime is 0, no need to detect");
        return;
    }

    // 检查设备类型是否支持
    if (localRankInfo.deviceType != DevType::DEV_TYPE_910_93 && localRankInfo.deviceType != DevType::DEV_TYPE_910B) {
        HCCL_WARNING("[AddIpQueue] not support deviceType[%d]", localRankInfo.deviceType);
        return;
    }

    // 使用 unique_lock 来管理互斥锁
    std::unique_lock<std::mutex> lock(ipNictypeQueueMutex_);
    if (nicType == NicType::VNIC_TYPE) {
        if (!localRankInfo.deviceVnicIp.IsInvalid()) {
            auto ip = ipMap_.find(remoteRankInfo.deviceVnicIp);
            if (ip == ipMap_.end()) {
                ipMap_.insert(std::make_pair(remoteRankInfo.deviceVnicIp, localRankInfo.deviceVnicIp));
                HCCL_INFO("[Add][IpQueue]localIp[%s], remoteIp[%s]", localRankInfo.deviceVnicIp.GetReadableAddress(),
                    remoteRankInfo.deviceVnicIp.GetReadableAddress());
                ipNictypeQueue_.push(std::make_pair(std::make_pair(localRankInfo, remoteRankInfo), nicType));
            }
        }
    } else if (nicType == NicType::DEVICE_NIC_TYPE) {
        auto ip = ipMap_.find(remoteRankInfo.nicIp[0]);
        if (ip == ipMap_.end()) {
            HCCL_INFO("[Add][IpQueue]localIp[%s], remoteIp[%s]", localRankInfo.nicIp[0].GetReadableAddress(),
                remoteRankInfo.nicIp[0].GetReadableAddress());
            ipMap_.insert(std::make_pair(remoteRankInfo.nicIp[0], localRankInfo.nicIp[0]));
            ipNictypeQueue_.push(std::make_pair(std::make_pair(localRankInfo, remoteRankInfo), nicType));
        }
    }
    lock.unlock();
    // 计算等待时间
    auto waitTime = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime()) +
        std::chrono::seconds(broadCastTime);
    
    auto startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= waitTime) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 每次休眠100毫秒
    }

    HCCL_INFO("[Add][IpQueue] ipNictypeQueue size[%d]", ipNictypeQueue_.size());
    return;
}

// 心跳线程调用
HcclResult DetectConnectionAnomalies::GetIpQueue(std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> socketMap,
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap)
{
    if (ipNictypeQueue_.empty()) {
        return HCCL_SUCCESS;
    }

    HCCL_RUN_INFO("[GetIpQueue] ipNictypeQueue_ size[%d], start to detect", ipNictypeQueue_.size());
    std::unique_lock<std::mutex> lock(ipNictypeQueueMutex_);
    while (!ipNictypeQueue_.empty() && threadExit_) {
        socketMap_ = socketMap;
        netDevCtxMap_ = netDevCtxMap;
        auto& front = ipNictypeQueue_.front();
        NicType nicType = front.second;
        RankInfo localRankInfo = front.first.first;
        RankInfo remoteRankInfo = front.first.second;
        HCCL_INFO("[GetIpQueue] localIp[%s], remoteIp[%s]", localRankInfo.deviceVnicIp.GetReadableIP(),
            remoteRankInfo.deviceVnicIp.GetReadableIP());
        if (CreateServers(localRankInfo, remoteRankInfo, nicType) != HCCL_SUCCESS ||
            CreateClients(nicType, linkClientThreads_, localRankInfo, remoteRankInfo) != HCCL_SUCCESS) {
            ipNictypeQueue_.pop();
            HCCL_ERROR("[GetIpQueue]CreateServers or CreateClients fail");
            return HCCL_E_INTERNAL;
        }
        ipNictypeQueue_.pop();
    }
    // client端需要的线程join
    HCCL_INFO("[GetIpQueue] compeleted[%d]", ipNictypeQueue_.size());
    lock.unlock();
    return HCCL_SUCCESS;
}


HcclResult DetectConnectionAnomalies::CreateDetectVnicLinks(HcclIpAddress localIpAddr, HcclIpAddress remoteIpAddr)
{
    SetThreadName("Hccl_Detect");
    CHK_PRT_RET(socketMap_.find(localIpAddr) == socketMap_.end(),
        HCCL_ERROR("[vnic]ip[%s] is not in socketMap", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);
    u32 port = socketMap_[localIpAddr]->GetLocalPort();
    CHK_PRT_RET(netDevCtxMap_.find(localIpAddr) == netDevCtxMap_.end(),
        HCCL_ERROR("[vnic]ip[%s] is not in socketMap", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);
    HcclNetDevCtx nicCtx = netDevCtxMap_[localIpAddr];
    auto timeOut = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());

    std::string tag = GetTag(localIpAddr);
    EXECEPTION_CATCH((socketServer_ = std::make_shared<HcclSocket>(nicCtx, port)), return HCCL_E_PTR);
    // 接收异常
    HCCL_INFO("[CreateDetectVnicLinks]tag[%s], localIpAddr[%s], remoteIpAddr[%u]", tag.c_str(),
        localIpAddr.GetReadableIP(),  remoteIpAddr.GetReadableIP());
    CHK_RET(socketServer_->Init());
    CHK_RET(socketServer_->Listen());
    CHK_RET(AddWhiteList(localIpAddr)); // 添加白名单

    u32 acceptTimeOut = 1;
    std::shared_ptr<HcclSocket> acceptSuccessSocket;
    auto startTime = std::chrono::steady_clock::now();
    while ((std::chrono::steady_clock::now() - startTime) < std::chrono::seconds(timeOut)) {
        HcclResult ret = socketServer_->Accept(tag, acceptSuccessSocket, acceptTimeOut);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[CreateDetectVnicLinks] accept success, localIpAddr[%s], acceptSuccessSocket[%p]",
                localIpAddr.GetReadableIP(), acceptSuccessSocket.get());
            listenVnicVec_.push_back(acceptSuccessSocket);
        }
    }

    // 循环发送
    startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= std::chrono::seconds(broadCastTime)) {
        for (auto &socket : listenVnicVec_) {
            std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
            for (auto &recvError : recvErrorInfoMap_) {
                auto it = sendErrorInfoIpSet_.find(recvError.first);
                if (it == sendErrorInfoIpSet_.end()) {
                    CHK_RET(socket->Send(&recvError.second, sizeof(recvError.second)));
                    sendErrorInfoIpSet_.insert(recvError.first);
                }
            }
            lock.unlock();
        }
    }
    return HCCL_SUCCESS;
}


HcclResult DetectConnectionAnomalies::CreateDetectNicLinks(HcclIpAddress localIpAddr, HcclIpAddress remoteIpAddr)
{
    CHK_PRT_RET(socketMap_.find(localIpAddr) == socketMap_.end(),
        HCCL_ERROR("[CreateDetectNicLinks]ip[%s] is not in socketMap", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);
    u32 port = socketMap_[localIpAddr]->GetLocalPort();
    CHK_PRT_RET(netDevCtxMap_.find(localIpAddr) == netDevCtxMap_.end(),
        HCCL_ERROR("[CreateDetectNicLinks]ip[%s] is not in netDevCtxMap", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);

    HcclNetDevCtx nicCtx = netDevCtxMap_[localIpAddr];
    std::string tag = GetTag(localIpAddr);
    EXECEPTION_CATCH((nicSocket_ = std::make_shared<HcclSocket>(nicCtx, port)), return HCCL_E_PTR);
    HCCL_INFO("[CreateDetectNicLinks]tag[%s], localIpAddr[%s], remoteIpAddr[%u]", tag.c_str(),
        localIpAddr.GetReadableIP(), remoteIpAddr.GetReadableIP());
    CHK_RET(nicSocket_->Init());
    CHK_RET(nicSocket_->Listen());
    CHK_RET(AddWhiteList(localIpAddr)); // 添加白名单

    auto acceptTimeOut = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());

    u32 acceptTimeOutAccept = 1;
    std::shared_ptr<HcclSocket> acceptSuccessSocket;
    auto startTime = std::chrono::steady_clock::now();
    HcclResult ret;
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= acceptTimeOut) {
        ret = nicSocket_->Accept(tag, acceptSuccessSocket, acceptTimeOutAccept);
        if (ret == HCCL_SUCCESS) {
            HCCL_INFO("[CreateDetectVnicLinks] accept success, localIpAddr[%s], acceptSuccessSocket[%p]",
                localIpAddr.GetReadableIP(), acceptSuccessSocket.get());
            listenNicVec_.push_back(acceptSuccessSocket);
        }
    }
    
    // 循环发送
    startTime = std::chrono::steady_clock::now();
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) <= std::chrono::seconds(broadCastTime)) {
        for (auto &socket : listenNicVec_) {
            std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
            for (auto &recvError : recvErrorInfoMap_) {
                auto it = sendErrorInfoIpSet_.find(recvError.first);
                if (it == sendErrorInfoIpSet_.end()) {
                    detctRes_ = true;// 记录是否找到报错节点
                    CHK_RET(socket->Send(&recvError.second, sizeof(recvError.second)));
                    sendErrorInfoIpSet_.insert(recvError.first);
                }
            }
            lock.unlock();
        }
    }
    HCCL_INFO("[CreateDetectNicLinks] socketServer_ [%p]", socketServer_.get());
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateServers(RankInfo loaclRankInfo, RankInfo remoteRankInfo, NicType nicType)
{
    if (threadExit_) {
        HcclIpAddress localIp;
        HcclIpAddress remopteIp;
        if (!isCreateLink_) {
            if (nicType == NicType::VNIC_TYPE) {
                localIp = loaclRankInfo.deviceVnicIp;
                remopteIp = remoteRankInfo.deviceVnicIp;
                HCCL_INFO("[CreateServers]localIpAddr[%s], remoteIpAddr[%s], nicType[%d]",
                    localIp.GetReadableIP(), remopteIp.GetReadableIP(), nicType);
                detectThread_.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateDetectVnicLinks, this,
                    localIp, remopteIp));
                CHK_SMART_PTR_NULL(detectThread_);
            }
            isCreateLink_ = true;
        }

        // 多机场景，且vnic失败时
        if (isNeedNic_) {
            localIp = loaclRankInfo.nicIp[0];
            remopteIp = remoteRankInfo.nicIp[0];
            HCCL_INFO("[CreateServers]nicIp[%s], remoteIp[%s], nicType[%d]", localIp.GetReadableIP(),
                remopteIp.GetReadableIP(), nicType);
            // 这里得用nicIp，否则添加白名单无效
        if (!isCreateNicLink_) {
                detectNicThread_.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateDetectNicLinks,
                    this, localIp, remopteIp));
                CHK_SMART_PTR_NULL(detectNicThread_);
                isCreateNicLink_ = true;
            }
        }
    }
    return HCCL_SUCCESS;
}

std::string DetectConnectionAnomalies::GetTag(HcclIpAddress &Ip, int i)
{
    return std::string(Ip.GetReadableIP()) + "_detect_" + std::to_string(i);
}

HcclResult DetectConnectionAnomalies::processWhiteList(const HcclIpAddress &ipAddr, HcclIpAddress &localIpAddr)
{
    if (ipAddr.IsInvalid()) {
        return HCCL_SUCCESS;
    }
    
    // 白名单查重
    auto item = std::find(whiteListVec_.begin(), whiteListVec_.end(), ipAddr);
    if (item != whiteListVec_.end()) {
        return HCCL_SUCCESS;
    }

    std::vector<struct SocketWlistInfo> wlistInfosVec;
    SocketWlistInfo wlistInfo;
    wlistInfo.connLimit = 1;
    wlistInfo.remoteIp.addr = ipAddr.GetBinaryAddress().addr; // 这里传递二进制 IP 会有什么问题

    std::string tag = GetTag(localIpAddr);
    CHK_SAFETY_FUNC_RET(memcpy_s(&wlistInfo.tag[0], sizeof(wlistInfo.tag), tag.c_str(), tag.size() + 1));
    wlistInfosVec.push_back(wlistInfo);

    CHK_PRT_RET(socketMap_.find(localIpAddr) == socketMap_.end(),
        HCCL_ERROR("[processWhiteList]ip[%s] is not in socketMap_", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);
    HCCL_INFO("[AddNicWhiteList]tag[%s], localIpAddr[%s], remote_ip.addr[%u]", tag.c_str(),
        ipAddr.GetReadableIP(), wlistInfo.remoteIp.addr);
    HcclResult ret = socketMap_[localIpAddr]->AddWhiteList(wlistInfosVec);
    
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[processWhiteList]ip[%s] AddWhiteList fail", ipAddr.GetReadableIP()),
        HCCL_E_NOT_FOUND);
    whiteListVec_.push_back(ipAddr); // 保存白名单，方便查重
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::AddWhiteList(HcclIpAddress &localIpAddr)
{
    // 合并处理NIC和VNIC的IP地址
    for (auto &ip : uniqueIps_) {
        // 处理IP地址
        CHK_RET(processWhiteList(ip, localIpAddr));
    }

    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::ConstructErrorInfo(std::shared_ptr<HcclSocket> &tempSocket,
    RankInfo &localRankInfo, RankInfo &remoteRankInfo)
{
    detect_info  detectInfo{};
    detectInfo.detectResult = true;
    detectInfo.localDeviceId = localRankInfo.devicePhyId;

    // 获取本地设备IP并复制到错误信息中(直接获取，因为vnic场景从localRankInfo获得的IP可能是无效的)
    std::string localDeviceIp = tempSocket->GetLocalIp().GetReadableIP();
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.localDeviceIp, DEST_MAX_LEN, localDeviceIp.c_str(), localDeviceIp.size()));

    // 获取本地主机IP并复制到错误信息中
    std::string hostIp = localRankInfo.hostIp.GetReadableIP();
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.localHostIp, DEST_MAX_LEN, hostIp.c_str(), hostIp.size()));

    // 复制本地SuperPod ID到错误信息中
    std::string superPodId = localRankInfo.superPodId;
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.localSuperPodId, DEST_MAX_LEN, superPodId.c_str(), superPodId.size()));

    detectInfo.remoteDeviceId = remoteRankInfo.devicePhyId;
    // 获取远程设备IP并复制到错误信息中
    std::string remoteDeviceIp = tempSocket->GetRemoteIp().GetReadableIP();
    CHK_SAFETY_FUNC_RET(
        memcpy_s(detectInfo.remoteDeviceIp, DEST_MAX_LEN, remoteDeviceIp.c_str(), remoteDeviceIp.size()));
    
    // 获取远程主机IP并复制到错误信息中
    std::string remoteHostIp = remoteRankInfo.hostIp.GetReadableIP();
    CHK_SAFETY_FUNC_RET(
        memcpy_s(detectInfo.remoteHostIp, DEST_MAX_LEN, remoteHostIp.c_str(), remoteHostIp.size()));

    // 复制远程SuperPod ID到错误信息中
    std::string remoteSuperPodId =  remoteRankInfo.superPodId;
    CHK_SAFETY_FUNC_RET(memcpy_s(detectInfo.remoteSuperPodId, DEST_MAX_LEN, remoteSuperPodId.c_str(),
        remoteSuperPodId.size()));

    std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
    std::string ip = localDeviceIp + "-" + remoteDeviceIp;
    recvErrorInfoMap_.emplace(ip, detectInfo);
    lock.unlock();

    HCCL_ERROR("[ConstructErrorInfo]detectResult[%d], localDeviceIp[%s], localDeviceId[%d], localHostIp[%s],"
        "localSuperPodId[%s], remoteDeviceIp[%s], remoteSuperPodId[%s], remoteHostIp[%s], remoteDeviceId[%d]",
        detectInfo.detectResult, detectInfo.localDeviceIp, detectInfo.localDeviceId, detectInfo.localHostIp,
        detectInfo.localSuperPodId, detectInfo.remoteDeviceIp, detectInfo.remoteSuperPodId, detectInfo.remoteHostIp,
        detectInfo.remoteDeviceId);
    // 保存错误信息
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::GetStatus(std::shared_ptr<HcclSocket> &tempSocket,
    RankInfo &localRankInfo, RankInfo &remoteRankInfo)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime());
    // 等待时间不大于超时时间
    HcclSocketStatus status = HcclSocketStatus::SOCKET_INIT;

    while ((std::chrono::steady_clock::now() - startTime) < timeout) {
        status = tempSocket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_OK) {
            HCCL_INFO("[Detect][ConnectionAnomalies]GetStatus success, remoteIpAddr[%s]",
                tempSocket->GetRemoteIp().GetReadableIP());
            return HCCL_SUCCESS;
        }
        usleep(ONE_MILLISECOND_OF_USLEEP); // 休眠1毫秒
    }
    std::unique_lock<std::mutex> lock(ipConstuctMutex_);
    CHK_RET(ConstructErrorInfo(tempSocket, localRankInfo, remoteRankInfo));
    lock.unlock();
    return HCCL_E_TIMEOUT;
}

HcclResult DetectConnectionAnomalies::Connect(RankInfo &localRankInfo, RankInfo &remoteRankInfo,
    std::shared_ptr<HcclSocket> &tempSocket, NicType nicType)
{
    HcclSocketRole role = HcclSocketRole::SOCKET_ROLE_CLIENT;
    HcclIpAddress localIpAddr = (nicType == NicType::VNIC_TYPE) ? localRankInfo.deviceVnicIp :
        localRankInfo.nicIp[0];
    CHK_PRT_RET(netDevCtxMap_.find(localIpAddr) == netDevCtxMap_.end(),
        HCCL_ERROR("ip[%s] is not in netDevCtxMap_", localIpAddr.GetReadableIP()), HCCL_E_NOT_FOUND);
    HcclNetDevCtx nicCtx = netDevCtxMap_[localIpAddr];
    // hccp多进程场景，todo能不能从业务传出port
    u32 vnicPort = localRankInfo.deviceVnicPort;
    u32 nicPort = localRankInfo.deviceNicPort;
    u32 port = nicType == NicType::VNIC_TYPE ? vnicPort : nicPort;
    HcclIpAddress remoteIpAddr = (nicType == NicType::VNIC_TYPE) ? remoteRankInfo.deviceVnicIp :
        remoteRankInfo.nicIp[0];
    std::string tag = GetTag(remoteIpAddr);
    HCCL_INFO("[Connect]tag[%s], port[%u], nicCtx[%p], remoteIpAddr[%s], role[%d]", tag.c_str(),
        port, nicCtx, remoteIpAddr.GetReadableIP(), role);
    EXECEPTION_CATCH((tempSocket = std::make_shared<HcclSocket>(tag, nicCtx, remoteIpAddr, port, role)),
        return HCCL_E_PTR);
    CHK_RET(tempSocket->Init());

    HcclResult ret = tempSocket->Connect();
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[Detect][ConnectionAnomalies] connect fail,"
        "localIp[%s], remoteIp[%s]",
        localIpAddr.GetReadableIP(), remoteIpAddr.GetReadableIP()),
        HCCL_E_INTERNAL);
    HCCL_INFO("[Connect] Connect success  localIp[%s], remoteIp[%s]", localIpAddr.GetReadableIP(),
        remoteIpAddr.GetReadableIP());
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateClient(NicType nicType, RankInfo localRankInfo, RankInfo remoteRankInfo)
{
    SetThreadName("Hccl_Detect");
    std::shared_ptr<HcclSocket> tempSocket;

    CHK_RET(Connect(localRankInfo, remoteRankInfo, tempSocket, nicType));
    HcclResult ret = GetStatus(tempSocket, localRankInfo, remoteRankInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[CreateClientConnect]GetStatus fail, ret[%d]", ret);
        return ret;
    }

    detect_info detectInfo{};
    u32 expectSize = sizeof(detect_info);
    u64 recvSize = 0;
    void *recvBuffer = reinterpret_cast<void *>(&detectInfo);

    auto waitTime = std::chrono::seconds(GetExternalInputDfsConnectionFaultDetctionTime()) +
        std::chrono::seconds(broadCastTime);
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("[CreateClient] start to recv");
    while (threadExit_ && (std::chrono::steady_clock::now() - startTime) < waitTime) {
        u64 compSize = 0;
        ret = tempSocket->IRecv(recvBuffer, expectSize, compSize);
        if (ret == HCCL_SUCCESS && compSize > 0) {
            recvBuffer = reinterpret_cast<u8 *>(recvBuffer) + compSize; // 偏移
            recvSize += compSize;
            HCCL_DEBUG("[CreateClient]CreateClients expectSize[%u], recvSize[%llu]", expectSize, recvSize);
            // 循环接收
            if (recvSize == static_cast<u64>(expectSize)) {
                recvSize = 0;
                std::string loaclIp(detectInfo.localDeviceIp);
                std::string remopteIp(detectInfo.remoteDeviceIp);
                std::string ip = loaclIp + "-" + remopteIp;
                auto it  = recvErrorInfoMap_.find(ip);
                if (it == recvErrorInfoMap_.end()) {
                    detctRes_ = true;// 记录是否找到报错节点
                    std::unique_lock<std::mutex> lock(readRecvErrtInfo_);
                    recvErrorInfoMap_.emplace(ip, detectInfo);
                    HCCL_ERROR(
                        "[CreateClient]detectResult[%d], localDeviceIp[%s], localDeviceId[%d], localHostIp[%s],"
                        "localSuperPodId[%s], remoteDeviceIp[%s], remoteSuperPodId[%s], remoteHostIp[%s], remoteDeviceId[%d]",
                        detectInfo.detectResult, detectInfo.localDeviceIp, detectInfo.localDeviceId, detectInfo.localHostIp,
                        detectInfo.localSuperPodId, detectInfo.remoteDeviceIp, detectInfo.remoteSuperPodId, detectInfo.remoteHostIp,
                        detectInfo.remoteDeviceId);
                    lock.unlock();
                }
                continue;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1)); // 休眠1毫秒
    }
    return HCCL_SUCCESS;
}

HcclResult DetectConnectionAnomalies::CreateClients(NicType nicType,
    std::vector<std::unique_ptr<std::thread>> &linkClientThreads, RankInfo localRankInfo, RankInfo remoteRankInfo)
{
    std::unique_ptr<std::thread> linkClientThread;
    linkClientThread.reset(new (std::nothrow) std::thread(&DetectConnectionAnomalies::CreateClient, this, nicType,
        localRankInfo, remoteRankInfo));
    linkClientThreads.emplace_back(std::move(linkClientThread));
    return HCCL_SUCCESS;
}

void DetectConnectionAnomalies::ThreadDestory()
{
    HCCL_DEBUG("[DetectConnectionAnomalies]Destory");
    threadExit_ = false;

    if (count_ == 0 && detctRes_) {
        HCCL_RUN_INFO("[DetectConnectionAnomalies]No link abnormality detected");
        count_++;
    }
    for (u32 index = 0; index < linkClientThreads_.size(); index++) {
        if (linkClientThreads_[index] == nullptr || !linkClientThreads_[index]->joinable()) {
            continue;
        }
        linkClientThreads_[index]->join(); // 等待线程执行完毕
    }
    linkClientThreads_.clear();

    if (detectThread_ != nullptr && detectThread_->joinable()) {
        detectThread_->join();
    }
    if (detectNicThread_ != nullptr && detectNicThread_->joinable()) {
        detectNicThread_->join();
    }
}

DetectConnectionAnomalies::~DetectConnectionAnomalies()
{
    ThreadDestory();
}  
} // namespace hccl