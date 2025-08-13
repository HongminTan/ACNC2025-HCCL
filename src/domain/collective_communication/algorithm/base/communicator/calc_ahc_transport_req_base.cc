/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "calc_ahc_transport_req_base.h"

namespace hccl {
CalcAHCTransportReqBase::CalcAHCTransportReqBase(std::vector<std::vector<u32>> &subCommPlaneVector,
    std::vector<bool> &isBridgeVector, u32 userRank, std::vector<std::vector<std::vector<u32>>> &globalSubGroups,
    std::map<AHCConcOpType, TemplateType> &ahcAlgOption)
    : CalcTransportReqBase(subCommPlaneVector, isBridgeVector, userRank), globalSubGroups_(globalSubGroups), ahcAlgOption_(ahcAlgOption)
{
}

CalcAHCTransportReqBase::~CalcAHCTransportReqBase()
{
}

HcclResult CalcAHCTransportReqBase::DisposeSubGroups(u32 rank)
{
    (void)rank;
    return HCCL_SUCCESS;
}
 
HcclResult CalcAHCTransportReqBase::CalcDstRanks(u32 rank, std::set<u32> &dstRanks, u32 ringIndex)
{
    (void)rank;
    (void)dstRanks;
    (void)ringIndex;
    return HCCL_SUCCESS;
}

HcclResult CalcAHCTransportReqBase::CommAHCInfoInit(std::vector<std::vector<u32>> &subGroups)
{
    (void) subGroups;
    return HCCL_SUCCESS;
}

HcclResult CalcAHCTransportReqBase::CalcTransportRequest(const std::string &tag, TransportMemType inputMemType,
        TransportMemType outputMemType, const CommParaInfo &commParaInfo,
        std::vector<SingleSubCommTransport> &commTransport, u32 subUserRankRoot)
{
    (void)subUserRankRoot;
    u32 ringSize = subCommPlaneVector_.size();
    commTransport.resize(ringSize);
    if (tag.find("AllReduce", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_ALLREDUCE;
    } else if(tag.find("ReduceScatter", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_REDUCE_SCATTER;
    } else if (tag.find("AllGather", 0) != std::string::npos) {
        opType_ = AHCOpType::AHC_OP_TYPE_ALLGATHER;
    }

    for (u32 ringIndex = 0; ringIndex < ringSize; ringIndex++) {
        if (commParaInfo.commPlane == COMM_LEVEL1_AHC && !isBridgeVector_[ringIndex]) {
            continue; // 跳出本次循环
        }

        u32 rank = GetSubCollectiveRank(subCommPlaneVector_[ringIndex]);
        if (rank == INVALID_VALUE_RANKID) {
            continue;
        }

        u32 rankSize = subCommPlaneVector_[ringIndex].size();
        SingleSubCommTransport &subCommTransport = commTransport[ringIndex];
        subCommTransport.transportRequests.resize(rankSize);
        // 只有一张卡时不需要建链
        if (rankSize == HCCL_RANK_SIZE_EQ_ONE) {
            HCCL_INFO("[CalcAHCTransportReqBase] comm base needn't to create links, rankSize_[%u].", rankSize);
            return HCCL_SUCCESS;
        }

        std::set<u32> dstRanks;
        CHK_RET(CalcDstRanks(rank, dstRanks, ringIndex));

        // 建链
        for (u32 dstRank : dstRanks) {
            CHK_PRT_RET(dstRank >= rankSize,
                HCCL_ERROR("[CalcAHCTransportReqBase][CalcTransportRequest] dstRank [%u] exceed rankSize [%u]  error", 
                dstRank, rankSize ), HCCL_E_INTERNAL);

            if (dstRank != rank) {
                TransportRequest &tmpTransport = subCommTransport.transportRequests[dstRank];
                tmpTransport.isValid = true;
                tmpTransport.localUserRank  = userRank_;
                tmpTransport.remoteUserRank = subCommPlaneVector_[ringIndex][dstRank];
                tmpTransport.inputMemType = inputMemType;
                tmpTransport.outputMemType = outputMemType;
                HCCL_INFO("[CalcAHCTransportReqBase] param_.tag[%s] ringIndex[%u], localRank[%u], " \
                    "remoteRank[%u], inputMemType[%d], outputMemType[%d]", tag.c_str(), ringIndex, userRank_,
                    tmpTransport.remoteUserRank, inputMemType, outputMemType);
            }
        }
    }
    return HCCL_SUCCESS;
}

}  // namespace hccl