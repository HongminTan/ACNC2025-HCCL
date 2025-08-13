/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_reduce_scatter_hccs_sio_executor.h"
#include "alg_template_register.h"

namespace hccl {

CollReduceScatterHccsSioExecutor::CollReduceScatterHccsSioExecutor(
    const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollReduceScatterExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollReduceScatterHccsSioExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
{
    HcclUs startut = TIME_NOW();
    tag_ = param.tag;
    algResResp_ = &algRes;

    ExecMem execMem;
    execMem.count = param.DataDes.count;
    execMem.inputPtr = param.inputPtr;
    execMem.outputPtr = param.outputPtr;
    execMem.inputMem = algRes.paramInputMem;
    execMem.outputMem = algRes.paramOutputMem;

    HcclResult ret = KernelRun(param, execMem);
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollReduceScatterHccsSioExecutor]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], ReduceScatter executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollReduceScatterVMeshOpbaseExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcCommInfo(
    std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::PARAM_INPUT;
    outputType = TransportMemType::PARAM_OUTPUT;
    HCCL_INFO("[CollReduceScatterHccsSioExecutor][CalcTransportMemType] tag[%s] inputType[%d],"
        " outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_COMBINE_ORDER, CommType::COMM_TAG_HCCS_PLUS_SIO);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollReduceScatterHccsSioExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollReduceScatterHccsSioExecutor][KernelRun] userRank[%u] starts.", topoAttr_.userRank);
    HcclDataType dataType = param.DataDes.dataType;

    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo subCommInfoHccs = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    SubCommInfo subCommInfoSio = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_1);

    u64 reduceAttr = GetReduceAttr(execMem.inputMem, execMem.outputMem, dataType, param.reduceType);
    std::unique_ptr<AlgTemplateBase> TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_REDUCESCATTER_HCCS_SIO, dispatcher_);
    CHK_SMART_PTR_NULL(TempAlg);

    CHK_RET(TempAlg->Prepare(execMem.inputMem, execMem.outputMem, execMem.scratchMem, execMem.count, dataType,
        param.stream, param.reduceType, LEVEL0_BRIDGE_RANK_ID, 0, reduceAttr,
        algResResp_->slaveStreams, algResResp_->notifiesMain, algResResp_->notifiesAux,
        topoAttr_.userRank, subCommInfoHccs, subCommInfoSio));

    CHK_RET(TempAlg->RegisterProfiler(
        (subCommInfoHccs.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + subCommInfoHccs.localRank,
        PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(TempAlg, subCommInfoHccs));

    return HCCL_SUCCESS;
}

REGISTER_EXEC("ReduceScatterHccsSioExecutor",
    ReduceScatterHccsSio, CollReduceScatterHccsSioExecutor);
}