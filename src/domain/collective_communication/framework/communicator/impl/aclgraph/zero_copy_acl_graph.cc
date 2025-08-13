/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "zero_copy_acl_graph.h"
#include "stream_utils.h"
namespace hccl {
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
ZeroCopyAclGraph::ZeroCopyAclGraph() : tagResourceIndex_(0), retryEnable_(false)
{
    algoSet_.insert(HcclCMDType::HCCL_CMD_BROADCAST);
    algoSet_.insert(HcclCMDType::HCCL_CMD_ALLREDUCE);
    algoSet_.insert(HcclCMDType::HCCL_CMD_REDUCE);
    algoSet_.insert(HcclCMDType::HCCL_CMD_ALLTOALL);
    algoSet_.insert(HcclCMDType::HCCL_CMD_ALLTOALLV);
    algoSet_.insert(HcclCMDType::HCCL_CMD_SCATTER);
    algoSet_.insert(HcclCMDType::HCCL_CMD_REDUCE_SCATTER);
    algoSet_.insert(HcclCMDType::HCCL_CMD_SEND);
    algoSet_.insert(HcclCMDType::HCCL_CMD_RECEIVE);
    algoSet_.insert(HcclCMDType::HCCL_CMD_ALLGATHER);
}

ZeroCopyAclGraph::~ZeroCopyAclGraph()
{}

std::string ZeroCopyAclGraph::GetTagPrefix()
{
    std::stringstream ss;
    ss << std::hex << std::uppercase << (tagResourceIndex_++);
    return ss.str();
}

void ZeroCopyAclGraph::SetRetryEnable(bool retryEnable)
{
    this->retryEnable_ = retryEnable;
}

bool ZeroCopyAclGraph::SetAclGraphZeroCopyMode(
    DevType deviceType, HcclCMDType opType, OpParam &opParam, HcclAlg *impl, u64 bufferSize)
{
    bool isInGraphCaputureZeroCopy = false;
    rtModel_t rtModel = nullptr;
    bool isCapture = false;

    if (deviceType != DevType::DEV_TYPE_910_93) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl doen't support graph zero copy mode. current "
                  "device is %d not DEV_TYPE_910_93",
            deviceType);
        return false;
    }
    if (opParam.isZeroCopy) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl can't support graph zero copy mode and operator "
                  "zero copy at the same time.");
        return false;
    }

    GetStreamCaptureInfo(opParam.stream.ptr(), rtModel, isCapture);
    if (isCapture) {
        isInGraphCaputureZeroCopy = SetGraphMode(opType, opParam, impl, bufferSize);
    }
    return isInGraphCaputureZeroCopy;
}

bool ZeroCopyAclGraph::SetGraphMode(HcclCMDType opType, OpParam &opParam, HcclAlg *impl, u64 bufferSize)
{
    if (!opParam.aicpuUnfoldMode || GetExternalInputHcclAivMode()) {
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl can't support graph zero copy "
                  "mode. Only support on aicpu mode aicpuUnfoldMode %d aiv %d",
            opParam.aicpuUnfoldMode,
            GetExternalInputHcclAivMode());
        return false;
    }
    if (IsAlgoSupportAclGraphZeroCopyMode(opType, opParam, impl, bufferSize)) {
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        HCCL_INFO("[ZeroCopyAclGraph][SetAclGraphZeroCopyMode] Hccl set op %d workflow mode to "
                  "HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB "
                  "graph zero copy mode.",
            opType);
        return true;
    }
    return false;
}

bool ZeroCopyAclGraph::AlgoCheck(OpParam &opParam, std::unique_ptr<CollAlgOperator> &algo, u64 bufferSize)
{
    std::string algName;
    std::string newTag;
    if (opParam.aicpuUnfoldMode) {
        // 用于inplace支持重执行判断
        algo->SetRetryEnable(retryEnable_);
    }
    HcclResult res = algo->SelectAlg(opParam.tag, opParam, algName, newTag);
    if (res != HCCL_SUCCESS) {
        HCCL_INFO("[ZeroCopyAclGraph][AlgoCheck] Select algo failed. result =%x", res);
        return false;
    }

    AlgResourceRequest resRequest;
    HcclResult ret = algo->CalcResRequest(algName, opParam, resRequest);
    if (ret == HCCL_SUCCESS) {
        if (IsScratchMemorySupportAclGraphZeroCopyMode(opParam, bufferSize, resRequest.scratchMemSize)) {
            opParam.tag = opParam.tag + GetTagPrefix();
            HCCL_INFO("[ZeroCopyAclGraph][AlgoCheck] scratch support.");
            return true;
        }
        HCCL_INFO("[ZeroCopyAclGraph][AlgoCheck] scratch support failed.");
    } else {
        HCCL_INFO("[ZeroCopyAclGraph][AlgoCheck] op %d calcResRequest failed.", opParam.opType);
    }
    return false;
}

bool ZeroCopyAclGraph::IsAlgoSupportAclGraphZeroCopyMode(
    HcclCMDType opType, OpParam &opParam, HcclAlg *impl, u64 bufferSize)
{
    if (algoSet_.find(opType) != algoSet_.end()) {
        HcclWorkflowMode oldMode = GetWorkflowMode();
        SetWorkflowMode(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        std::unique_ptr<CollAlgOperator> algo = impl->GetAlgOperator(opType);
        if (algo == nullptr) {
            HCCL_INFO("[ZeroCopyAclGraph][IsAlgoSupportAclGraphZeroCopyMode] GetAlgo failed.");
            return false;
        }
        if (AlgoCheck(opParam, algo, bufferSize)) {
            return true;
        }
        HCCL_INFO("[ZeroCopyAclGraph][IsAlgoSupportAclGraphZeroCopyMode] algo check failed.");
        SetWorkflowMode(oldMode);
    }

    return false;
}

bool ZeroCopyAclGraph::IsScratchMemorySupportAclGraphZeroCopyMode(
    const OpParam &opParam, u64 bufferSize, u64 scratchMemSize)
{
    if (scratchMemSize <= bufferSize) {
        HCCL_INFO("[ZeroCopyAclGraph] OP %d support acl graph zero copy. scratchmemsize=%ul cclbuffer size=%ul",
            opParam.opType,
            scratchMemSize,
            bufferSize);
        return true;
    }
    HCCL_INFO("[ZeroCopyAclGraph] OP %d doesn't support acl graph zero copy. scratchmemsize=%ul cclbuffer size=%ul",
        opParam.opType,
        scratchMemSize,
        bufferSize);
    return false;
}

#endif
}  // namespace hccl