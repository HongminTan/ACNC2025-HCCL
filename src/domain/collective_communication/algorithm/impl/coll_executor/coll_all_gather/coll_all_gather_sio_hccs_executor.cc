/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_sio_hccs_executor.h"
 
namespace hccl {
CollAllGatherSioHccsExecutor::CollAllGatherSioHccsExecutor(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
}

HcclResult CollAllGatherSioHccsExecutor::Orchestrate(OpParam& param, AlgResourceResponse& algRes)
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
        HCCL_ERROR("[CollAllGatherSioHccsExecutor]errNo[0x%016llx] tag[%s] excutor kernel run failed",
            HCCL_ERROR_CODE(ret), param.tag.c_str()), ret);

    HCCL_INFO("tag[%s], AllGather executor orchestrate success, take time [%lld]us.",
        param.tag.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::CalcNotifyNum(u32 streamNum, u32 &notifyNum)
{
    // notify数量是从流的两倍 + 新增带notifyId的notify资源
    notifyNum = 2U * streamNum;
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = topoAttr_.deviceNumPerAggregation + 1U;
    streamNum = totalStreamNum - 1U;
    HCCL_INFO("[CollAllGatherSioHccsExecutor][CalcStreamNum] tag[%s] streamNum[%u]",
        tag_.c_str(), streamNum);
 
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    inputType = TransportMemType::PARAM_INPUT;
    outputType = TransportMemType::PARAM_OUTPUT;
 
    HCCL_INFO("[CollAllGatherSioHccsExecutor][CalcTransportMemType]" \
        "tag[%s] inputType[%d], outputType[%d]", tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType, std::vector<LevelNSubCommTransport>& opTransport)//opTransport  vector<vector<SingleSubCommTransport>>
{
    CommParaInfo commParaLevel0(COMM_COMBINE_ORDER, CommType::COMM_TAG_HCCS_PLUS_SIO);//新建通信平面
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_COMBINE_ORDER], inputType, outputType));
 
    LevelNSubCommTransport &commTransportLevel0 = opTransport[COMM_COMBINE_ORDER];
    for (u32 subCommIndex = 0; subCommIndex < commTransportLevel0.size(); subCommIndex++) {
        for (auto &transportRequest : commTransportLevel0[subCommIndex].transportRequests) {//两个transport分别设置ishccs为false、true
            transportRequest.notifyNum = 4U; //只传递额外的notify个数
            // 根据子通信索引设置isHccs的值
            transportRequest.linkType = (subCommIndex == 0) ? TransportLinkType::HCCS : TransportLinkType::SIO;
            HCCL_INFO("[CollAllGatherSioHccsExecutor][CalcLevel0CommInfo] set extral notifyNum[%u]",
                transportRequest.notifyNum);
        }
    }
    return HCCL_SUCCESS;
}
 
HcclResult CollAllGatherSioHccsExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherSioHccsExecutor][KernelRun] The AllGatherSioHccsExecutor starts.");
 
    // step 1 先获取 comm inner \ comm outer 的value
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_0 + 1));
    SubCommInfo outerCommInfoHccs = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_0);
    CHK_RET(CheckCommSize(COMM_COMBINE_ORDER, COMM_INDEX_1 + 1));
    SubCommInfo outerCommInfoSio = GetSubCommInfo(COMM_COMBINE_ORDER, COMM_INDEX_1);

    // 执行
    std::unique_ptr<AlgTemplateBase> tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
        TemplateType::TEMPLATE_ALL_GATHER_SIO_HCCS, dispatcher_);
    CHK_SMART_PTR_NULL(tempAlg);

    CHK_RET(tempAlg->Prepare(outerCommInfoHccs, outerCommInfoSio, execMem.inputMem, execMem.outputMem,
        execMem.count, param.DataDes.dataType, param.stream, algResResp_->slaveStreams,
        algResResp_->notifiesMain, algResResp_->notifiesAux, topoAttr_.userRank));
 
    CHK_RET(tempAlg->RegisterProfiler((outerCommInfoHccs.localRankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) +
        outerCommInfoHccs.localRank, PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
 
    CHK_RET(RunTemplate(tempAlg, outerCommInfoHccs));
 
    HCCL_INFO("[AllGatherExector][AllGatherSioHccsExecutor] AllGatherSioHccsExecutor ends.");
 
    return HCCL_SUCCESS;
}
 
REGISTER_EXEC("AllGatherSioHccsExecutor", AllGatherSioHccs, CollAllGatherSioHccsExecutor);
 
} // namespace hccl