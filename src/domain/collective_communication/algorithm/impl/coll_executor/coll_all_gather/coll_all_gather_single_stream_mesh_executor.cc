
/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_single_stream_mesh_executor.h"

namespace hccl {
CollAllGatherSingleStreamMeshExecutor::CollAllGatherSingleStreamMeshExecutor(const HcclDispatcher dispatcher,
    std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = false;
}

HcclResult CollAllGatherSingleStreamMeshExecutor::CalcStreamNum(u32& streamNum)
{
    streamNum = 1U;
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSingleStreamMeshExecutor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::CCL_INPUT;
    TransportMemType outputType = TransportMemType::CCL_OUTPUT;
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherSingleStreamMeshExecutor::CalcLevel0CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_MESH);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

//根据 CCL 缓冲区大小和数据单元大小，计算 AllGather 操作在单次循环中可处理的最大元素数量，确保数据传输不超过预分配的缓冲区容量。
//HCCL_MIN_SLICE_ALIGN是 HCCL 定义的最小数据分片对齐值
u64 CollAllGatherSingleStreamMeshExecutor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

HcclResult CollAllGatherSingleStreamMeshExecutor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherSingleStreamMeshExecutor][KernelRun]allgather new start");
    u32 perDataSize = SIZE_TABLE[param.DataDes.dataType];

    // 获取子通信域信息
    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0RankSize = level0CommInfo.localRankSize;
    u32 commIndex = level0CommInfo.localRank;
    CHK_RET(CheckCommSize(COMM_LEVEL1, commIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(COMM_LEVEL1, commIndex);
    u32 serverIndex = level1CommInfo.localRank;


    u64 inputMemSize = execMem.inputMem.size();
    u64 baseOffset = serverIndex * inputMemSize * level0RankSize;
    u64 level0Offset = commIndex * inputMemSize;
    DeviceMem dstMem = execMem.outputMem.range(baseOffset + level0Offset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);
    //  第一步，将数据从input内存拷贝到output内存的对应位置
    HcclResult ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
    CHK_PRT_RET(ret != HCCL_SUCCESS,
        HCCL_ERROR("[CollAllGatherSingleStreamMeshExecutor][KernelRun]all gather 4PmeshHD memcpy Failed, Offset[%llu], Size[%llu].",
        baseOffset + level0Offset, inputMemSize), ret);

    // 第二步，各个AI Server 内  all gather
    std::vector<Slice> dataSegsSlice;                 // 数据分成ranksize份，每份的起始偏移和大小
    std::vector<std::vector<Slice>> multiStreamSlice; // 每个stream使用的数据基于用户buffer的偏移
    u32 sliceNum = level0RankSize;
    CHK_RET(PrepareAllgatherSlice(sliceNum, inputMemSize, dataSegsSlice));

    CHK_RET(ActiveSlaveStreams(param.stream));

    //  抽取当前用于多环all gather 的output内存数据
    DeviceMem currentOutputMem = execMem.outputMem.range(baseOffset, inputMemSize * level0RankSize);
    CHK_SMART_PTR_NULL(currentOutputMem);


    std::unique_ptr<AlgTemplateBase> level0TempAlg;
    level0TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_SINGLE_STREAM_MESH,
                                                                      dispatcher_);
    CHK_SMART_PTR_NULL(level0TempAlg);

    CHK_RET(level0TempAlg->Prepare(currentOutputMem, currentOutputMem, execMem.inputMem,
        execMem.count * level0RankSize, param.DataDes.dataType, param.stream, HCCL_REDUCE_RESERVED,
        LEVEL0_BRIDGE_RANK_ID, dataSegsSlice, baseOffset));
    u32 rankSize = level0RankSize;
    CHK_RET(level0TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + commIndex,
        PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));
    CHK_RET(RunTemplate(level0TempAlg, level0CommInfo));
    HCCL_INFO("all gather mesh HD level0 run success");

    //  第三步， AI server 间  all gather
    u64 hdSize = inputMemSize * level0RankSize;
    u64 hdCount = hdSize / perDataSize;

    std::unique_ptr<AlgTemplateBase> level1TempAlg;
    //level1可以按照注释掉的模块选择已有的算法模板   也可以直接指定
    // if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING || (topoAttr_.isDiffDeviceModule && topoAttr_.serverNum == 1)) {
    //     // 1-单server-SDMA
    //     HCCL_ERROR("1");
    //     level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
    //             TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    //     HCCL_INFO("allgather mesh: using ring algo inter-server.");
    // } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
    //     HCCL_ERROR("2");
    //     level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
    //             TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
    //     HCCL_INFO("allgather mesh: using nhr algo inter-server.");
    // } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR_V1) {
    //     HCCL_ERROR("3");
    //     level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
    //             TemplateType::TEMPLATE_ALL_GATHER_NHRV1, dispatcher_);
    //     HCCL_INFO("allgather mesh: using nhr_v1 algo inter-server.");
    // } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
    //     HCCL_ERROR("4");
    //     level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
    //             TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
    //     HCCL_INFO("allgather mesh: using nonuniform-bruck algo inter-server.");
    // } else {
    //     HCCL_ERROR("5");
    //     level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
    //             //TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
    //             TemplateType::TEMPLATE_ALL_GATHER_NEW, dispatcher_);
    //     HCCL_INFO("allgather mesh: using halving-doubling algo inter-server.");
    // }
    level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING, dispatcher_);
                //TemplateType::TEMPLATE_ALL_GATHER_NEW, dispatcher_);

    CHK_SMART_PTR_NULL(level1TempAlg);

    //  此处虽然带入inputMem作为scratch mem, 但inputMem 不能被使用
    CHK_RET(level1TempAlg->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, hdCount,
        param.DataDes.dataType, param.stream, HcclReduceOp::HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID,
        std::vector<Slice>(COMM_INDEX_0), 0));

    rankSize = level1CommInfo.localRankSize;
    CHK_RET(level1TempAlg->RegisterProfiler((rankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + serverIndex,
        PROF_STAGE_2, HCCL_EXEC_STEP_NOT_SET, param.stream));

    CHK_RET(RunTemplate(level1TempAlg, level1CommInfo));
    HCCL_INFO("all gather mesh HD level1 run success");
    return HCCL_SUCCESS;
}

REGISTER_EXEC("AllGatherSingleStreamMeshExecutor", AllGatherSingleStreamMesh, CollAllGatherSingleStreamMeshExecutor);
} // namespace hccl
