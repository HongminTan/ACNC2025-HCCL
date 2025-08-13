/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CALC_AHC_TRANSPORT_REQ_BASE_H
#define CALC_AHC_TRANSPORT_REQ_BASE_H

#include "comm_ahc_base_pub.h"
#include "calc_transport_req_base.h"
 
namespace hccl {
class CalcAHCTransportReqBase : public CalcTransportReqBase {
public:
    explicit CalcAHCTransportReqBase(std::vector<std::vector<u32>> &subCommPlaneVector,
        std::vector<bool> &isBridgeVector, u32 userRank, std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
        std::map<AHCConcOpType, TemplateType> &ahcAlgOption);
 
    ~CalcAHCTransportReqBase() override;
 
    HcclResult CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
        TransportMemType outputMemType, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, u32 subUserRankRoot = INVALID_VALUE_RANKID) override;
 
    virtual HcclResult CalcDstRanks(u32 rank, std::set<u32> &dstRanks, u32 ringIndex);
 
    AHCOpType opType_; // 当前处理的算子类型
protected:
    std::unique_ptr<CommAHCBaseInfo> commAHCBaseInfo_;
    std::vector<std::vector<u32>> subGroups_;
    std::vector<std::vector<std::vector<u32>>> globalSubGroups_;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption_;
private:
    virtual HcclResult DisposeSubGroups(u32 rank);
    virtual HcclResult CommAHCInfoInit(std::vector<std::vector<u32>> &subGroups);
};
} // namespace hccl
#endif /* CALC_AHC_TRANSPORT_REQ_BASE_H */