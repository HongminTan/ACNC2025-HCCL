/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ONE_SIDED_SERVICE_H
#define HCCL_ONE_SIDED_SERVICE_H

#include "i_hccl_one_sided_service.h"
#include "hccl_one_sided_conn.h"
#include "hccl_common.h"
#include "externalinput_pub.h"
#include "hccl_mem.h"

using HcclBatchData = struct HcclBatchDataDef {
    HcclComm comm;
    HcclCMDType cmdType;
    u32 remoteRank;
    HcclOneSideOpDesc* desc;
    u32 descNum;
    rtStream_t stream;
};

namespace hccl {
constexpr size_t HCCL_MEM_DESC_STR_LEN = HCCL_MEM_DESC_LENGTH + 1 - (sizeof(u32) * 2);

class HcclOneSidedService : public IHcclOneSidedService {
public:
    using RankId = u32;
    using ProcessInfo = HcclOneSidedConn::ProcessInfo;

    struct HcclMemDescData {
        u32 localRankId;
        u32 remoteRankId;
        char memDesc[HCCL_MEM_DESC_STR_LEN];
    };

    HcclOneSidedService(std::unique_ptr<HcclSocketManager> &socketManager,
        std::unique_ptr<NotifyPool> &notifyPool);

    // 父类Config()等已经完成必要参数的配置
    HcclOneSidedService() = default;
    ~HcclOneSidedService() override;

    HcclResult ReMapMem(HcclMem *memInfoArray, u64 arraySize);
    HcclResult RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc &localMemDesc);
    HcclResult DeregMem(const HcclMemDesc &localMemDesc);
    // 可能返回超时
    HcclResult ExchangeMemDesc(RankId remoteRankId, const HcclMemDescs &localMemDescs,
        HcclMemDescs &remoteMemDescs, u32 &actualNumOfRemote, const std::string &commIdentifier, s32 timeoutSec);

    void EnableMemAccess(const HcclMemDesc &remoteMemDesc, HcclMem &remoteMem);
    void DisableMemAccess(const HcclMemDesc &remoteMemDesc);

    void BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t &stream);
    void BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t &stream);

    HcclResult GetIsUsedRdma(RankId remoteRankId, bool &useRdma);

private:
    u32 registedMemCnt_{0};
    HcclResult IsUsedRdma(RankId remoteRankId, bool &useRdma);

    HcclResult SetupRemoteRankInfo(RankId remoteRankId, HcclRankLinkInfo &remoteRankInfo);
    HcclResult CreateConnection(RankId remoteRankId, const HcclRankLinkInfo &remoteRankInfo,
        std::shared_ptr<HcclOneSidedConn> &tempConn);
    HcclResult Grant(const HcclMemDesc &localMemDesc, const ProcessInfo &remoteProcess);
    HcclBuf *GetHcclBufByDesc(std::string &descStr, bool useRdma);

    std::unordered_map<RankId, std::shared_ptr<HcclOneSidedConn>> oneSidedConns_{};
    std::unordered_map<RankId, bool> isUsedRdmaMap_;
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapIpc_{};
    std::unordered_map<std::string, HcclBuf> desc2HcclBufMapRoce_{};
};
}

#endif