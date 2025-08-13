/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_DETECT_CONNECT_ANOMALIES_H
#define HCCL_DETECT_CONNECT_ANOMALIES_H
#include "hccl_socket_manager.h"
#include "hccl_socket.h"
#include "hccl_ip_address.h"
#include "common.h"
#include <queue>
#include <map>
#include <unordered_set>
#include <mutex>
#include <thread>

namespace hccl {
// todo 本端和对端的都得保存，并打印
constexpr size_t DEST_MAX_LEN = 128;
struct detect_info {
    bool detectResult = false;
    char localHostIp[DEST_MAX_LEN]{};
    char remoteHostIp[DEST_MAX_LEN]{};

    s32 localDeviceId = 0XFFFFFFFF;
    s32 remoteDeviceId = 0XFFFFFFFF;

    char localDeviceIp[DEST_MAX_LEN]{};
    char remoteDeviceIp[DEST_MAX_LEN]{};

    char localSuperPodId[DEST_MAX_LEN]{};
    char remoteSuperPodId[DEST_MAX_LEN]{};
};

class DetectConnectionAnomalies {
public:
    static DetectConnectionAnomalies &GetInstance(s32 deviceLogicID);
    void Init(std::vector<RankInfo> &rankInfos, bool isNeedNic);
    void AddIpQueue(RankInfo &localRankInfo, RankInfo &remoteRankInfo, NicType nicType);
    HcclResult GetIpQueue(std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> socketMap,
        std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap);
    std::string GetTag(HcclIpAddress &Ip, int i = 0);
    HcclResult AddWhiteList(HcclIpAddress &localIpAddr);
    HcclResult GetStatus(std::shared_ptr<HcclSocket> &tempSocket, RankInfo &localRankInfo,
        RankInfo &remoteRankInfo);
    HcclResult Connect(RankInfo &localRankInfo, RankInfo &remoteRankInfo,
        std::shared_ptr<HcclSocket> &tempSocket, NicType nicType);
    HcclResult CreateDetectVnicLinks(HcclIpAddress localIpAddr, HcclIpAddress remoteIpAddr);
    HcclResult CreateDetectNicLinks(HcclIpAddress localIpAddr, HcclIpAddress remoteIpAdd);
    HcclResult CreateClients(NicType nicType,
        std::vector<std::unique_ptr<std::thread>> &linkClientThreads, RankInfo localRankInfo, RankInfo remoteRankInfo);
    HcclResult CreateServers(RankInfo localRankInfo, RankInfo remoteRankInfo, NicType nicType);
    HcclResult ConstructErrorInfo(std::shared_ptr<HcclSocket> &tempSocket, RankInfo &localRankInfo, RankInfo &remoteRankInfo);
    HcclResult CreateClient(NicType nicType, RankInfo localRankInfo, RankInfo remoteRankInfo);
    HcclResult processWhiteList(const HcclIpAddress &ipAddr, HcclIpAddress &localIpAddr);
    void ThreadDestory();

private:
    DetectConnectionAnomalies() = default;
    ~DetectConnectionAnomalies();
    int broadCastTime = 10; // 故障广播时间
    std::set<HcclIpAddress> uniqueIps_;
    bool threadExit_ = true;
    bool isNeedNic_ = false;
    std::mutex ipNictypeQueueMutex_;
    std::mutex ipConstuctMutex_;
    std::mutex readDetectInfo_;
    std::vector<HcclIpAddress> whiteListVec_;
    std::shared_ptr<HcclSocket> socketServer_ = nullptr;
    std::shared_ptr<HcclSocket> nicSocket_ = nullptr;
    std::map<HcclIpAddress, HcclIpAddress> ipMap_;
    std::map<HcclIpAddress, std::shared_ptr<HcclSocket>> socketMap_;
    std::map<HcclIpAddress, HcclNetDevCtx> netDevCtxMap_;
    std::vector<std::shared_ptr<HcclSocket>> listenNicVec_;
    std::vector<std::shared_ptr<HcclSocket>> listenVnicVec_;
    std::queue<std::pair<std::pair<RankInfo, RankInfo>, NicType>> ipNictypeQueue_;
    std::unique_ptr<std::thread> detectThread_ = nullptr;
    std::unique_ptr<std::thread> detectNicThread_ = nullptr;

    // 发送完成后添加，发送前查重
    std::unordered_set<std::string> sendErrorInfoIpSet_;
    // 接收到添加，接收前查重
    std::unordered_map<std::string, detect_info> recvErrorInfoMap_;
    std::mutex readRecvErrtInfo_;

    bool isCreateLink_ = false;
    bool isCreateNicLink_  = false;
    std::atomic<bool> detctRes_{false}; // 记录是否探测到异常
    std::vector<std::unique_ptr<std::thread>> linkClientThreads_; // 保存client拉起的线程
    std::atomic<int> count_{0};
};
}

#endif // HCCL_DETECT_CONNECT_ANOMALIES_H
