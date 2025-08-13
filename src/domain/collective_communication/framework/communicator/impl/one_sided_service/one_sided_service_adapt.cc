/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_services.h"
#include "exception_handler.h"
#include "hccl_one_sided_service.h"
#include "hccl_comm_pub.h"
#include "hccl_communicator.h"
#include "i_hccl_one_sided_service.h"
#include "adapter_prof.h"
#include "param_check_pub.h"
#include "profiler_base_pub.h"
#include "externalinput_pub.h"
#include "profiling_manager_pub.h"
#include "adapter_rts_common.h"
using namespace hccl;
using namespace std;

constexpr u64 ONE_SIDE_DEVICE_MEM_MAX_SIZE = 64llu * 1024 * 1024 * 1024;  // device侧支持内存注册大小上限为64GB
constexpr u64 ONE_SIDE_HOST_MEM_MAX_SIZE = 1024llu * 1024 * 1024 * 1024;  // host侧支持内存注册大小上限为1TB
constexpr u64 ONE_SIDE_HOST_MEM_ZERO = 0;

HcclResult HcclOneSidedSetIfProfile()
{
    bool ifOpbase = (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
    bool state = ProfilingManagerPub::GetAllState();
    SetIfProfile((!ifOpbase) || (!state));
    return HCCL_SUCCESS;
}

void HcclOneSidedResetIfProfile()
{
    SetIfProfile(true);
}

static HcclResult AddDescTraceInfo(hccl::hcclComm* hcclComm, HcclOneSideOpDesc* desc, u32 descNum, const std::string& tag)
{
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    // trace日志逐个记录描述符信息
    for (u32 i = 0; i < descNum; i++) {
        CHK_PTR_NULL((desc + i)->localAddr);
        CHK_PTR_NULL((desc + i)->remoteAddr);
        CHK_RET(HcomCheckCount((desc + i)->count));
        CHK_RET(HcomCheckDataType((desc + i)->dataType));
        s32 ret = snprintf_s(stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
                             "[%s] HcclOneSideOpDesc[%d] : localAddr[%p], remoteAddr[%p], count[%llu], dataType[%d].",
                             __func__, i, (desc + i)->localAddr, (desc + i)->remoteAddr, (desc + i)->count, (desc + i)->dataType);
        CHK_PRT_CONT(ret == -1, HCCL_WARNING("Failed to build log info, tag[%s].", tag.c_str()));
        std::string logInfo(stackLogBuffer);
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclRemapRegistedMemory(HcclComm *comm, HcclMem *memInfoArray, u64 commSize, u64 arraySize)
{
    RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
    std::vector<std::string>({"HcclRemapRegistedMemory", "comm", "nullptr", "please check comm"}));
    CHK_PTR_NULL(comm);
    RPT_INPUT_ERR(memInfoArray == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclRemapRegistedMemory", "memInfoArray", "nullptr", "please check memInfoArray"}));
    CHK_PTR_NULL(memInfoArray);

    RPT_INPUT_ERR(commSize <= ONE_SIDE_HOST_MEM_ZERO, "EI0003", std::vector<std::string>(
        {"ccl_op", "parameter", "value", "tips"}), std::vector<std::string>({"HcclRemapRegistedMemory", "commSize", \
        "less than or equal to 0", "please check commSize"}));
    CHK_PRT_RET(commSize <= ONE_SIDE_HOST_MEM_ZERO, HCCL_ERROR("[HcclRemapRegistedMemory]commSize[%llu] is invalid, "\
        "please check commSize", commSize), HCCL_E_PARA);
    RPT_INPUT_ERR(arraySize <= ONE_SIDE_HOST_MEM_ZERO, "EI0003", std::vector<std::string>(
        {"ccl_op", "parameter", "value", "tips"}), std::vector<std::string>({"HcclRemapRegistedMemory", "arraySize", \
        "less than or equal to 0", "please check arraySize"}));
    CHK_PRT_RET(arraySize <= ONE_SIDE_HOST_MEM_ZERO, HCCL_ERROR("[HcclRemapRegistedMemory]arraySize[%llu] is invalid, "\
        "please check arraySize", arraySize), HCCL_E_PARA);

    IHcclOneSidedService *service = nullptr;
    for (u64 i = 0; i < commSize; i++) {
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm[i]);
        CHK_PTR_NULL(hcclComm);
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        CHK_RET(reinterpret_cast<HcclOneSidedService*>(service)->ReMapMem(memInfoArray, arraySize));
    }

    return HCCL_SUCCESS;
}

static HcclResult CallOneSideMsprofReportHostApi(hccl::hcclComm* hcclComm, HcclCMDType cmdType, uint64_t beginTime, u64 count,
                                                 HcclDataType dataType, std::string tag)
{
    if (GetIfProfile()) {
        AlgType algType;
        CHK_RET(hcclComm->GetAlgType(algType, cmdType));
        uint64_t groupName = hrtMsprofGetHashId(hcclComm->GetIdentifier().c_str(), hcclComm->GetIdentifier().length());
        CHK_RET_AND_PRINT_IDE(ProfilingManagerPub::CallMsprofReportHostApi(cmdType, beginTime, count, dataType, algType,
                                                                           groupName), tag.c_str());
    }
    return HCCL_SUCCESS;
}

// HcclCommInitClusterInfoMem在open_hccl中
HcclResult HcclRegisterMem(HcclComm comm, u32 remoteRank, int type,
                           void* addr, u64 size, HcclMemDesc* desc)
{
    EXCEPTION_HANDLE_BEGIN
        // 参数校验和适配
        RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclRegisterMem", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(addr == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclRegisterMem", "addr", "nullptr", "please check addr"}));
        CHK_PTR_NULL(addr);
        RPT_INPUT_ERR(desc == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclRegisterMem", "memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(desc);

        u32 localRank = INVALID_VALUE_RANKID;
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        CHK_RET(hcclComm->GetUserRank(localRank));
        std::string commIdentifier = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s], remoteRank[%u], memType[%d], memAddr[%p], memSize[%llu], memDescPtr[%p]",
                      __func__, commIdentifier.c_str(), remoteRank, type, addr, size, desc);
        CHK_PRT_RET(remoteRank == localRank, HCCL_WARNING("remoteRank[%u] is equal to localRank[%u], no need to "\
        "register memeory, return HcclRegisterMem success", remoteRank, localRank), HCCL_SUCCESS);

        CHK_PRT_RET(type != HCCL_MEM_TYPE_DEVICE && type != HCCL_MEM_TYPE_HOST,
                    HCCL_ERROR("[HcclRegisterMem]memoryType[%d] must be device or host, please check type", type), HCCL_E_PARA);
        CHK_PRT_RET(size <= ONE_SIDE_HOST_MEM_ZERO, HCCL_ERROR("[HcclRegisterMem]memory size[%llu] is invalid, "\
        "please check memory size", size), HCCL_E_PARA);
        CHK_PRT_RET(type == HCCL_MEM_TYPE_DEVICE && size > ONE_SIDE_DEVICE_MEM_MAX_SIZE,
                    HCCL_ERROR("[HcclRegisterMem]memory size[%llu] is too large, please check memory size", size), HCCL_E_PARA);
        CHK_PRT_RET(type == HCCL_MEM_TYPE_HOST && size > ONE_SIDE_HOST_MEM_MAX_SIZE,
                    HCCL_ERROR("[HcclRegisterMem]memory size[%llu] is too large, please check memory size", size), HCCL_E_PARA);

        IHcclOneSidedService *service = nullptr;
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);

        // 校验netDevCtx是否为空
        bool useRdma;
        CHK_RET(reinterpret_cast<HcclOneSidedService*>(service)->GetIsUsedRdma(remoteRank, useRdma));
        HcclNetDevCtx netDevCtx;
        CHK_RET(service->GetNetDevCtx(netDevCtx, useRdma));
        if (netDevCtx == nullptr) {
            HCCL_INFO("[%s]Network resources are not initialized, start to initOneSidedServiceNetDevCtx", __func__);
            CHK_RET(hcclComm->InitOneSidedServiceNetDevCtx(remoteRank));
        }
        CHK_RET(reinterpret_cast<HcclOneSidedService*>(service)->RegMem(addr, size, static_cast<HcclMemType>(type), remoteRank, *desc));

        HCCL_RUN_INFO("%s success:comm[%s], remoteRank[%u], memType[%d], memAddr[%p], memSize[%llu], memDescPtr[%p]",
                      __func__, commIdentifier.c_str(), remoteRank, type, addr, size, desc);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclDeregisterMem(HcclComm comm, HcclMemDesc* desc)
{
    EXCEPTION_HANDLE_BEGIN
        // 参数校验和适配
        RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclDeregisterMem", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(desc == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclDeregisterMem", "memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(desc);

        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        std::string commIdentifier = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s], memDescPtr[%p]", __func__, commIdentifier.c_str(), desc);
        IHcclOneSidedService *service = nullptr;
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        CHK_RET(reinterpret_cast<HcclOneSidedService*>(service)->DeregMem(*desc));

        HCCL_RUN_INFO("%s success:comm[%s], memDescPtr[%p]", __func__, commIdentifier.c_str(), desc);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclExchangeMemDesc(HcclComm comm, u32 remoteRank, HcclMemDescs* local,
                               int timeout, HcclMemDescs* remote, u32* actualNum)
{
    EXCEPTION_HANDLE_BEGIN
        // 参数校验和适配
        RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclExchangeMemDesc", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(local == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclExchangeMemDesc", "local memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(local);
        RPT_INPUT_ERR(remote == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclExchangeMemDesc", "remote memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(remote);
        RPT_INPUT_ERR(actualNum == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclExchangeMemDesc", "actualNum", "nullptr", "please check params"}));
        CHK_PTR_NULL(actualNum);

        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        std::string commIdentifier = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s], remoteRank[%u], localMemDescPtr[%p], timeout[%d], remoteMemDescPtr[%p], "
                      "actualNum[%u]", __func__, commIdentifier.c_str(), remoteRank, local, timeout, remote, *actualNum);
        u32 localRank = INVALID_VALUE_RANKID;
        CHK_RET(hcclComm->GetUserRank(localRank));
        CHK_PRT_RET(remoteRank == localRank, HCCL_WARNING("remoteRank[%u] is equal to localRank[%u], no need to "\
        "register memeory, return HcclRegisterMem success", remoteRank, localRank), HCCL_SUCCESS);

        IHcclOneSidedService *service = nullptr;
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        CHK_RET(reinterpret_cast<HcclOneSidedService *>(service)->ExchangeMemDesc(
                remoteRank, *local, *remote, *actualNum, commIdentifier, timeout));

        HCCL_RUN_INFO("%s success:comm[%s], remoteRank[%u], localMemDescPtr[%p], timeout[%d], remoteMemDescPtr[%p], "
                      "actualNum[%u]", __func__, commIdentifier.c_str(), remoteRank, local, timeout, remote, *actualNum);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclEnableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc, HcclMem* remoteMem)
{
    EXCEPTION_HANDLE_BEGIN
        // 参数校验和适配
        RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclEnableMemAccess", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(remoteMemDesc == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclEnableMemAccess", "remote memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(remoteMemDesc);
        RPT_INPUT_ERR(remoteMem == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclEnableMemAccess", "remoteMem Param error", "nullptr", "please check params"}));
        CHK_PTR_NULL(remoteMem);

        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        std::string commIdentifier = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s], remoteMemDescPtr[%p], remoteMemPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc,
                      remoteMem);
        IHcclOneSidedService *service = nullptr;
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        reinterpret_cast<HcclOneSidedService*>(service)->EnableMemAccess(*remoteMemDesc, *remoteMem);

        HCCL_RUN_INFO("%s success:comm[%s], remoteMemDescPtr[%p], remoteMemPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc,
                      remoteMem);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclDisableMemAccess(HcclComm comm, HcclMemDesc* remoteMemDesc)
{
    EXCEPTION_HANDLE_BEGIN
        // 参数校验和适配
        RPT_INPUT_ERR(comm == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclEnableMemAccess", "comm", "nullptr", "please check comm"}));
        CHK_PTR_NULL(comm);
        RPT_INPUT_ERR(remoteMemDesc == nullptr, "EI0003", std::vector<std::string>({"ccl_op", "parameter", "value", "tips"}),\
        std::vector<std::string>({"HcclEnableMemAccess", "remote memory description", "nullptr", "please check params"}));
        CHK_PTR_NULL(remoteMemDesc);

        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        std::string commIdentifier = hcclComm->GetIdentifier();
        HCCL_RUN_INFO("Entry-%s:comm[%s], remoteMemDescPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc);
        IHcclOneSidedService *service = nullptr;
        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        reinterpret_cast<HcclOneSidedService*>(service)->DisableMemAccess(*remoteMemDesc);

        HCCL_RUN_INFO("%s success:comm[%s], remoteMemDescPtr[%p]", __func__, commIdentifier.c_str(), remoteMemDesc);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

inline static HcclResult HcclBatchParaCheck(HcclBatchData &paraData, std::string &getTag)
{
    // 参数校验和适配
    CHK_PTR_NULL(paraData.comm);
    CHK_PTR_NULL(paraData.stream);
    CHK_PTR_NULL(paraData.desc);
    std::string batchString = (paraData.cmdType == HcclCMDType::HCCL_CMD_BATCH_GET) ? "BatchGet" : "BatchPut";
    CHK_PRT_RET(paraData.descNum == 0, HCCL_WARNING("[%s] the count of HcclOneSideOpDesc is zero.",
                                                    batchString.c_str()), HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(paraData.comm);
    // 同算子复用tag
    u32 localRank = INVALID_VALUE_RANKID;
    CHK_RET(hcclComm->GetGroupRank(localRank));

    const std::string tag = batchString + "_" + std::to_string(localRank) + "_" + std::to_string(paraData.remoteRank)
                            + "_" + hcclComm->GetIdentifier();
    getTag = tag;

    u32 rankSize = INVALID_VALUE_RANKSIZE;
    CHK_RET_AND_PRINT_IDE(hcclComm->GetRankSize(rankSize), tag.c_str());
    CHK_RET(HcomCheckUserRank(rankSize, paraData.remoteRank));
    CHK_PRT_RET(paraData.remoteRank == localRank,
                HCCL_ERROR("[%s] the remoteRank can't be equal to localRank, please check.", batchString.c_str()), HCCL_E_PARA);

    if (GetExternalInputHcclEnableEntryLog()) {
        s32 deviceLogicId = 0;
        s32 streamId = 0;
        CHK_RET(hrtGetDeviceRefresh(&deviceLogicId));
        CHK_RET(hrtGetStreamId(paraData.stream, streamId));
        // 记录接口交互信息日志
        std::string logInfo = "Entry-";
        logInfo.append(batchString);
        logInfo.append(":tag[");
        logInfo.append(tag);
        logInfo.append("], descNum[");
        logInfo.append(std::to_string(paraData.descNum));
        logInfo.append("], streamId[");
        logInfo.append(std::to_string(streamId));
        logInfo.append("], deviceLogicId[");
        logInfo.append(std::to_string(deviceLogicId));
        logInfo.append("].");
        CHK_RET(hcclComm->SaveTraceInfo(logInfo));
        CHK_RET(AddDescTraceInfo(hcclComm, paraData.desc, paraData.descNum, tag));
    }
    return HCCL_SUCCESS;
}

HcclResult HcclBatchPut(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream)
{
    EXCEPTION_HANDLE_BEGIN
        HcclOneSidedSetIfProfile();
        HcclUs startut = TIME_NOW();
        uint64_t beginTime = hrtMsprofSysCycleTime();
        std::string getTag;
        HcclBatchData paraData = {comm, HcclCMDType::HCCL_CMD_BATCH_PUT, remoteRank, desc, descNum, stream};
        CHK_RET(HcclBatchParaCheck(paraData, getTag));

        IHcclOneSidedService *service = nullptr;
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);
        
        HCCL_PROFILER_ADD_TAG(getTag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        HCCL_PROFILER_ADD_STREAM(stream, getTag, 0, AlgType::Reserved());

        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        reinterpret_cast<HcclOneSidedService*>(service)->BatchPut(remoteRank, desc, descNum, stream);

        CHK_RET(CallOneSideMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BATCH_PUT, beginTime, desc->count,
                                               desc->dataType, getTag));
        HcclOneSidedResetIfProfile();
        if (GetExternalInputHcclEnableEntryLog()) {
            HcclUs endut = TIME_NOW();
            std::string endInfo = "HcclBatchPut:success,take time: " +
                                  std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + getTag;
            CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), getTag.c_str());
        }
        HCCL_PROFILER_DEL_TAG(getTag);
        HCCL_PROFILER_DEL_STREAM(stream);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclBatchGet(HcclComm comm, u32 remoteRank, HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream)
{
    EXCEPTION_HANDLE_BEGIN
        HcclOneSidedSetIfProfile();
        HcclUs startut = TIME_NOW();
        uint64_t beginTime = hrtMsprofSysCycleTime();
        std::string getTag;
        HcclBatchData paraData = {comm, HcclCMDType::HCCL_CMD_BATCH_GET, remoteRank, desc, descNum, stream};
        CHK_RET(HcclBatchParaCheck(paraData, getTag));

        IHcclOneSidedService *service = nullptr;
        hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm *>(comm);

        HCCL_PROFILER_ADD_TAG(getTag, hcclComm->GetIdentifier(), HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);
        HCCL_PROFILER_ADD_STREAM(stream, getTag, 0, AlgType::Reserved());

        CHK_RET(hcclComm->GetOneSidedService(&service));
        CHK_PTR_NULL(service);
        reinterpret_cast<HcclOneSidedService*>(service)->BatchGet(remoteRank, desc, descNum, stream);

        CHK_RET(CallOneSideMsprofReportHostApi(hcclComm, HcclCMDType::HCCL_CMD_BATCH_GET, beginTime, desc->count,
                                               desc->dataType, getTag));
        HcclOneSidedResetIfProfile();
        if (GetExternalInputHcclEnableEntryLog()) {
            HcclUs endut = TIME_NOW();
            std::string endInfo = "HcclBatchGet:success,take time: " +
                                  std::to_string(DURATION_US(endut - startut).count()) + " us, tag: " + getTag;
            CHK_RET_AND_PRINT_IDE(hcclComm->SaveTraceInfo(endInfo), getTag.c_str());
        }
        HCCL_PROFILER_DEL_TAG(getTag);
        HCCL_PROFILER_DEL_STREAM(stream);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

// 通信域创建OneSidedService对象的回调函数
HcclResult HcclBuildOneSidedService(std::unique_ptr<IHcclOneSidedService> &service, std::unique_ptr<hccl::HcclSocketManager> &socketManager,
                                    std::unique_ptr<hccl::NotifyPool> &notifyPool)
{
    EXCEPTION_HANDLE_BEGIN
        service = std::make_unique<HcclOneSidedService>(socketManager, notifyPool);
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}