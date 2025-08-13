/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPOINFO_EXCHANGE_BASE_H
#define TOPOINFO_EXCHANGE_BASE_H

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <nlohmann/json.hpp>
#include "topoinfo_struct.h"
#include "comm.h"
#include "hccl_socket.h"
#include "hccl_network_pub.h"

namespace hccl {
constexpr u32 MAX_AGENT_BUF_SIZE = 256; // masterInfo方式: superPodId(128) + hostIp + deviceId
constexpr s32 TOPO_SERVERIP_OFFSET_OF_RANKID = 32;
constexpr int BIT_NUM_PER_BYTE = 8;
constexpr u32 TOPO_GROUPLEADER_PORT_OFFSET = 16 ; //TopoDetect GroupLeader监听端口偏移值
constexpr u32 TOPO_HIERARCHICAL_ENABLE_THRESHOLD = 32768 ; //TopoDetect 分层阈值
constexpr u32 TOPO_MAX_GROUP_SIZE = 2048 ;

enum class TopoDetectResult {
    TOPO_DETECT_SUCCESS = 0,
    TOPO_CONNECT_FAILED = 1,
};

enum class BroadcastStage {
    Idle = 0,
    Started = 1,
    Completed = 2
};

constexpr int MAX_WAIT_BROADCAST_SECONDS = 10;
constexpr u32 WAIT_ERROR_BROADCAST_TIME = 20;
extern std::atomic<BroadcastStage> g_broadcastStage;
extern std::mutex g_broadcast_stage_mutex;
extern std::condition_variable g_broadcast_stage_cv;

class TopoInfoExchangeBase {
public:
    TopoInfoExchangeBase();
    virtual ~TopoInfoExchangeBase();
protected:
    HcclResult DisconnectSocket(std::shared_ptr<HcclSocket> socket) const;
    HcclResult BlockReceive(std::shared_ptr<HcclSocket> socket, char *buff, u32 size) const;
    HcclResult SendClusterInfoMsg(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo,
                                    const std::string buffer, const u32 msgLen);
    HcclResult RecvClusterInfoMsg(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo);
    HcclResult SendClusterInfo(std::shared_ptr<HcclSocket> socket, const RankTable_t &clusterInfo);
    HcclResult RecvClusterInfo(std::shared_ptr<HcclSocket> socket, RankTable_t &clusterInfo);
    HcclResult RecvClusterJson(std::shared_ptr<HcclSocket> socket, nlohmann::json &jClusterJson);
    HcclResult RecvGrpLeaderInfoMsg(std::shared_ptr<HcclSocket> socket, GroupLeader_t &LeaderInfo);
    HcclResult parseJsonBuff(const char buff[], u32 buffLen, nlohmann::json& buffJson) const;
    HcclResult Json2Struct(const nlohmann::json& jClusterJson, RankTable_t &clusterInfo) const;
    HcclResult Json2GrpLeader(const nlohmann::json& jClusterJson, GroupLeader_t &GrpLeaderInfo) const;
    HcclResult Struct2Json(const RankTable_t &clusterInfo, nlohmann::json& ClusterJson);
    HcclResult SetClusterDeploy(const nlohmann::json& jClusterJson, RankTable_t &clusterInfo) const;
    HcclResult GrpLeader2Json(const GroupLeader_t &GrpLeaderInfo, nlohmann::json& GroupLeaderJson);
    HcclResult TransformRankListToJson(const RankTable_t &clusterInfo, nlohmann::json& rankListJson) const;
    u32 currentStep_; // topo detect 分为多个step， 用以校验server和agent的step是否一致。
    bool isByMasterInfo_ = false;
    u32 identifierNum_;
private:
    HcclResult GetCommonTopoInfo(RankTable_t &rankTable, const RankTable_t &orginRankTable);
    HcclResult SortRankList(RankTable_t &rankTable);
    friend class TopoInfoExchangeDispather;
};
}  // namespace hccl
#endif /* TOPOINFO_EXCHANGE_BASE_H */
