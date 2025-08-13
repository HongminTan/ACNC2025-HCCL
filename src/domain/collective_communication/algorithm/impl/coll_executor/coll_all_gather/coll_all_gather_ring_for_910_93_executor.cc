/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "coll_all_gather_ring_for_910_93_executor.h"

namespace hccl {
CollAllGatherRingFor91093Executor::CollAllGatherRingFor91093Executor(const HcclDispatcher dispatcher,
                                                                   std::unique_ptr<TopoMatcher> &topoMatcher)
    : CollAllGatherExecutor(dispatcher, topoMatcher)
{
    DMAReduceFlag_ = workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE;
}

HcclResult CollAllGatherRingFor91093Executor::CalcStreamNum(u32& streamNum)
{
    u32 totalStreamNum = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
        LEVEL0_PLANE_NUM_IN_NPRING_SINGLE);
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        totalStreamNum *= STREAM_NUM_FOR_DMAREDUCE_ONE_RING;
    }
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB &&
        GetExternalInputEnableRdmaSdmaConcurrent()) {
        totalStreamNum += (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? LEVEL0_PLANE_NUM_IN_NPRING_DOUBLE :
            LEVEL0_PLANE_NUM_IN_NPRING_SINGLE;
    }
    streamNum = totalStreamNum - 1;
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcStreamNum] tag[%s] streamNum_[%u]",
        tag_.c_str(), streamNum);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcCommInfo(std::vector<LevelNSubCommTransport>& opTransport)
{
    TransportMemType inputType = TransportMemType::RESERVED;
    TransportMemType outputType = TransportMemType::RESERVED;
    CHK_RET(CalcTransportMemType(inputType, outputType));
    CHK_RET(CalcLevel0CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel1CommInfo(inputType, outputType, opTransport));
    CHK_RET(CalcLevel2CommInfo(inputType, outputType, opTransport));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcTransportMemType(TransportMemType &inputType,
    TransportMemType &outputType)
{
    if (workflowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        inputType = TransportMemType::CCL_INPUT;
        outputType = TransportMemType::CCL_OUTPUT;
    } else {
        inputType = TransportMemType::PARAM_INPUT;
        outputType = TransportMemType::PARAM_OUTPUT;
    }
    HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcTransportMemType] tag[%s] inputType[%d], outputType[%d]",
        tag_.c_str(), inputType, outputType);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel0CommInfo(TransportMemType inputType,
    TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    CommParaInfo commParaLevel0(COMM_LEVEL0, CommType::COMM_TAG_RING_INNER);
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel0, opTransport[COMM_LEVEL0], inputType, outputType));
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::CalcLevel2CommInfo(TransportMemType inputType, TransportMemType outputType,
    std::vector<LevelNSubCommTransport>& opTransport)
{
    if( algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
        HCCL_INFO("[CollAllGatherRingFor91093Executor][CalcLevel2CommInfo] select AHC bypass level2 comm calulate");        
        return HCCL_SUCCESS;
    }
    
    CommParaInfo commParaLevel2(COMM_LEVEL2, CommType::COMM_TAG_MAX);
    if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_HIERARCHICAL_RING;
        HCCL_INFO("[%s]Calc NHRCommInfo", __func__);
    } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
        commParaLevel2.commType = CommType::COMM_TAG_NONUNIFORM_BRUCK;
        HCCL_INFO("[%s]Calc NBCommInfo", __func__);
    } else {
        commParaLevel2.commType = CommType::COMM_TAG_RING_INNER;
        HCCL_INFO("[%s]Calc RingCommInfo", __func__);
    }
    CHK_RET(CalcCommPlaneInfo(tag_, commParaLevel2, opTransport[COMM_LEVEL2], inputType, outputType));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcLoopMaxCount(const u64 cclBuffSize, const u32 unitSize)
{
    u64 maxCountPerLoop = cclBuffSize / topoAttr_.userRankSize / HCCL_MIN_SLICE_ALIGN
        * HCCL_MIN_SLICE_ALIGN / unitSize;
    return maxCountPerLoop;
}

bool CollAllGatherRingFor91093Executor::IsDataSplitForRdmaSdmaConcurrent(const u64 curSize)
{
    bool isLargeSize = (curSize >= HCCL_SPLIT_SIZE_INTER_SERVER);
    return GetExternalInputEnableRdmaSdmaConcurrent() && (topoAttr_.serverNum > 1) && isLargeSize;
}

HcclResult CollAllGatherRingFor91093Executor::RunIntraSeverAllGather(
    const std::string &tag, DeviceMem &inputMem, DeviceMem &outputMem,
    const u64 count, const HcclDataType &dataType, const std::vector<std::vector<Slice>> &multRingsSliceZero,
    const Stream &stream, s32 profStage, const u64 baseOffset, const HcomCollOpInfo *opInfo,
    const std::vector<std::vector<Slice>> &multRingsUserMemSlice)
{
    CHK_RET(MultiRingAllGather(tag, inputMem, outputMem, count, dataType,
        multRingsSliceZero, stream, profStage, baseOffset, opInfo, multRingsUserMemSlice));
    return HCCL_SUCCESS;
}

u64 CollAllGatherRingFor91093Executor::CalcDstMemOffset(const OpParam &param, u32 perDataSize, u64 inputMemSize) const
{
    return topoAttr_.userRank * inputMemSize;
}

HcomCollOpInfo CollAllGatherRingFor91093Executor::GetHcomCollOpInfo(const OpParam &param, const ExecMem &execMem) const
{
    HcomCollOpInfo opInfo = {
        "", execMem.inputPtr, execMem.outputPtr, param.DataDes.count, param.DataDes.dataType, 0, HCCL_REDUCE_RESERVED,
        param.DataDes.strideCount
    };
    if (!DMAReduceFlag_ && (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)) {
        opInfo.inputAddr = execMem.inputMem.ptr();
        opInfo.outputAddr = execMem.outputMem.ptr();
    }
    return opInfo;
}

std::vector<Slice> CollAllGatherRingFor91093Executor::PrepareSlicesL2(const OpParam &param,
    const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo,
    u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level1ServerIndex = level1CommInfo.localRank;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level2DataSegsSlice;
    for (u32 i = 0; i < level2RankSize; i++) {
        Slice sliceTemp;
        sliceTemp.size = inputMemSize;
        sliceTemp.offset = inputMemSize *
            (i * level1RankSize * level0RankSize + level1ServerIndex * level0RankSize + level0ServerIndex);
        level2DataSegsSlice.push_back(sliceTemp);
    }
    return level2DataSegsSlice;
}

std::vector<Slice> CollAllGatherRingFor91093Executor::PrepareSlicesL1(const OpParam &param,
    const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo,
    u32 perDataSize, u64 inputMemSize) const
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level0ServerIndex = level0CommInfo.localRank;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;
    std::vector<Slice> level1DataSegsSlice;
    for (u32 j = 0; j < level1RankSize; j++) {
        for (u32 i = 0; i < level2RankSize; i++) {
            Slice level1Slice;
            level1Slice.size = inputMemSize;
            level1Slice.offset = inputMemSize *
                (i * level1RankSize * level0RankSize + j * level0RankSize + level0ServerIndex);
            level1DataSegsSlice.push_back(level1Slice);
        }
    }
    return level1DataSegsSlice;
}

HcclResult CollAllGatherRingFor91093Executor::PrepareSlicesL0(std::vector<std::vector<Slice>> &multRingsSlice,
    const OpParam &param, const SubCommInfo &level2CommInfo, const SubCommInfo &level1CommInfo,
    const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    const u32 level0RankSize = level0CommInfo.localRankSize;
    const u32 level1RankSize = level1CommInfo.localRankSize;
    const u32 level2RankSize = level2CommInfo.localRankSize;

    std::vector<Slice> dataSegsSlice;
    CHK_RET(PrepareAllgatherSlice(level0RankSize, inputMemSize, dataSegsSlice));

    // 多环数据切分
    std::vector<std::vector<Slice>> multRingsSliceZero; // 数据基于该rank上环0的偏移
    if (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING &&
        !IsSupportUnifiedMarch(param, topoType_, topoAttr_.serverNum, topoAttr_.superPodNum)) {
        multRingsSliceZero = PrepareMultiRingSlice(dataSegsSlice, param.tag, false, topoAttr_.nicList);
    } else {
        multRingsSliceZero.push_back(dataSegsSlice);
    }
    for (u32 ringIndex = 0; ringIndex < multRingsSliceZero.size(); ringIndex++) {
        std::vector<Slice> level2DataSlice;
        CHK_RET(CalculateLevel2AllgatherSlice(inputMemSize, level0RankSize, level1RankSize, level2RankSize,
            multRingsSliceZero, level2DataSlice, ringIndex));
        multRingsSlice.push_back(level2DataSlice);
    }

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::PrepareUserMemSlices(std::vector<std::vector<Slice>> &userMemSlices,
    const std::vector<std::vector<Slice>> &multRingsSlice, const OpParam &param, const SubCommInfo &level2CommInfo,
    const SubCommInfo &level1CommInfo, const SubCommInfo &level0CommInfo, u32 perDataSize, u64 inputMemSize)
{
    CHK_PRT_RET(0 < param.DataDes.strideCount && param.DataDes.strideCount < param.DataDes.count,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu] is smaller than opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count),
        HCCL_E_PARA);
    HCCL_DEBUG("[CollAllGatherRingFor91093Executor][KernelRun]strideCount[%llu], opCount[%llu]",
        param.DataDes.strideCount, param.DataDes.count);

    if (!DMAReduceFlag_) {
        userMemSlices = multRingsSlice;
        // 图模式，根据strideCount更新slice的offset
        if (param.DataDes.strideCount != 0) {
            CHK_RET(UpdateOffsetBasedOnStrideCount(param, userMemSlices));
        }
    } else {
        for (u32 ringIndex = 0; ringIndex < multRingsSlice.size(); ringIndex++) {
            std::vector<Slice> userMemSlice;
            for (const auto &cclSlice : multRingsSlice[ringIndex]) {
                Slice tmpSlice;
                u64 count = (param.DataDes.strideCount == 0) ? param.DataDes.count : param.DataDes.strideCount;
                tmpSlice.size = cclSlice.size;
                tmpSlice.offset = (cclSlice.offset / inputMemSize) * count * perDataSize +
                    multRingsSlice[ringIndex][0].offset;
                userMemSlice.push_back(tmpSlice);
                HCCL_DEBUG("rank[%u], ringIndex[%u], tmpSlice.offset=[%llu], size=[%llu]",
                    topoAttr_.userRank, ringIndex, tmpSlice.offset, tmpSlice.size);
            }
            userMemSlices.push_back(userMemSlice);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::KernelRun(const OpParam &param, ExecMem &execMem)
{
    HCCL_CONFIG_INFO(HCCL_ALG, "[CollAllGatherRingFor91093Executor][KernelRun] The AllGatherRingExecutor starts. topoType_[%u], agv[%u]",
        topoType_, isAllGatherV_);
    CHK_RET(ActiveSlaveStreams(param.stream));
    const HcclDataType dataType = GetDataType(param);
    u32 perDataSize = 0;
    CHK_RET(SalGetDataTypeSize(dataType, perDataSize));
    CHK_PRT_RET(perDataSize == 0,
        HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]errNo[0x%016llx] datatype[%s] is invalid",
            HCCL_ERROR_CODE(HCCL_E_PARA), GetDataTypeEnumStr(dataType).c_str()), HCCL_E_PARA);

    CHK_RET(CheckCommSize(COMM_LEVEL0, COMM_INDEX_0 + 1));
    SubCommInfo level0CommInfo = GetSubCommInfo(COMM_LEVEL0, COMM_INDEX_0);
    u32 level0ServerIndex = level0CommInfo.localRank;
    
    bool isSelectAHC = (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC ||
        algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE);
    CommPlane commPlaneLevel1 = isSelectAHC ? COMM_LEVEL1_AHC : COMM_LEVEL1;
    CHK_RET(CheckCommSize(commPlaneLevel1, level0ServerIndex + 1));
    SubCommInfo level1CommInfo = GetSubCommInfo(commPlaneLevel1, level0ServerIndex);
    u32 level1RankSize = level1CommInfo.localRankSize;

    SubCommInfo level2CommInfo;
    if (isSelectAHC) {
        level2CommInfo = level1CommInfo;
        level2CommInfo.localRankSize = 1;   // AHC bypass level2
    } else {
        CHK_RET(CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1));
        level2CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);
    }
    const u32 level2RankSize = level2CommInfo.localRankSize;

    //  第一步，将数据从input内存拷贝到output内存的对应位置
    u64 inputMemSize = execMem.inputMem.size();
    u64 dstMemOffset = CalcDstMemOffset(param, perDataSize, inputMemSize);
    DeviceMem dstMem = execMem.outputMem.range(dstMemOffset, inputMemSize);
    CHK_SMART_PTR_NULL(dstMem);

    HcomCollOpInfo opInfo = GetHcomCollOpInfo(param, execMem);
    HcomCollOpInfo *opInfoPtr = (DMAReduceFlag_ || (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING)) ? &opInfo :
        nullptr;

    // 图模式opinfo不为空，但需要将数据从ccl input拷贝到ccl output上
    HcclResult ret = HCCL_SUCCESS;
    if (!DMAReduceFlag_) {
        ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, execMem.inputMem, const_cast<Stream&>(param.stream));
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                        "ring memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
    } else {
        // 先做server间算法，带有消减拷贝场景数据需要从user input取，拷贝到ccl output上
        if (level1RankSize > 1 || level2RankSize > 1) {
            DeviceMem srcMem = DeviceMem::create(static_cast<u8 *>(execMem.inputPtr), inputMemSize);
            ret = HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, const_cast<Stream&>(param.stream));
            CHK_PRT_RET(ret != HCCL_SUCCESS,
                HCCL_ERROR("[CollAllGatherRingFor91093Executor][KernelRun]all gather double "
                    "ring user memcpy Failed, Offset[%llu], Size[%llu]", dstMemOffset, inputMemSize), ret);
        }
    }
    if (level2RankSize > 1) {
        std::unique_ptr<AlgTemplateBase> level2AGExecutor;
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level2AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-superPod.");
        }
        CHK_SMART_PTR_NULL(level2AGExecutor);

        std::vector<Slice> level2DataSegsSlice = PrepareSlicesL2(param, level2CommInfo, level1CommInfo, level0CommInfo,
            perDataSize, inputMemSize);
        CHK_RET(level2AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
            dataType, param.stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level2DataSegsSlice, 0));

        CHK_RET(level2AGExecutor->RegisterProfiler((
            level2RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
            PROF_STAGE_0, HCCL_EXEC_STEP_NOT_SET, param.stream));

        CHK_RET(RunTemplate(level2AGExecutor, level2CommInfo));
        HCCL_INFO("allgather ring [superpod] level2 allgather run successtopoType_[%u], agv[%u]",
            topoType_, isAllGatherV_);
    }
    if (level1RankSize > 1) {
        // 计算slice, 不同超节点相同slice
        std::vector<Slice> level1DataSegsSlice = PrepareSlicesL1(param, level2CommInfo, level1CommInfo, level0CommInfo,
            perDataSize, inputMemSize);

        if (GetExternalInputEnableRdmaSdmaConcurrent() && (inputMemSize >= HCCL_SPLIT_SIZE_INTER_SERVER) 
            && !aicpuUnfoldMode_) {
            u32 syncTrans = (topoType_ == TopoType::TOPO_TYPE_NP_DOUBLE_RING) ? BEST_SPLIT_VALUE_DR :
                BEST_SPLIT_VALUE_SR;
            CHK_RET(Level1AllGatherConcurrent(execMem.inputMem, execMem.outputMem, execMem.count, dataType,
                param.stream, PROF_STAGE_1, level1DataSegsSlice, syncTrans));
        } else {
            std::unique_ptr<AlgTemplateBase> level1AGExecutor;
            if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_RING) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
                HCCL_INFO("allgather ring: using ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NB) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_NHR) {
                level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(
                    TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
                HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-server.");
            } else if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC || algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE) {
                // 获取通信域分组信息
                std::vector<std::vector<std::vector<u32>>> globalSubGroups;
                std::map<AHCConcOpType, TemplateType> ahcAlgOption;
                CHK_RET(topoMatcher_->GetGlobalSubGroups(commPlaneLevel1, globalSubGroups));
                topoMatcher_->GetAHCAlgOption(ahcAlgOption);
                if (algType_.algoLevel1 == AlgTypeLevel1::ALG_LEVEL1_AHC) {
                    level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC, dispatcher_);
                    HCCL_INFO("allgather comm: using ahc algo inter-server.");
                } else {
                    level1AGExecutor = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_GATHER_AHC_BROKE, dispatcher_);
                    HCCL_INFO("allgather comm: using ahc-broke algo inter-server.");
                }
                CHK_SMART_PTR_NULL(level1AGExecutor);
                CHK_RET(level1AGExecutor->Prepare(execMem.count, globalSubGroups, ahcAlgOption));
            } else {
                HCCL_ERROR("allgather ring: unsupported algtype [%s].", AlgTypeToStr(algType_).c_str());
                return HCCL_E_NOT_SUPPORT;
            }
            CHK_SMART_PTR_NULL(level1AGExecutor);
            CHK_RET(level1AGExecutor->Prepare(execMem.outputMem, execMem.outputMem, execMem.inputMem, execMem.count,
                dataType, param.stream, HCCL_REDUCE_RESERVED, INVALID_VALUE_RANKID, level1DataSegsSlice, 0));

            CHK_RET(level1AGExecutor->RegisterProfiler((
                level1RankSize << PROF_RANKSIZE_OFFSET_OF_PLANEID) + level2CommInfo.localRank,
                PROF_STAGE_1, HCCL_EXEC_STEP_NOT_SET, param.stream));

            CHK_RET(RunTemplate(level1AGExecutor, level1CommInfo));
            HCCL_INFO("allgather ring [superpod] level1 allgather run successtopoType_[%u], agv[%u]",
                topoType_, isAllGatherV_);
        }
    }
    // 节点内做all gather ring
    std::vector<std::vector<Slice>> multRingsSlice;
    CHK_RET(PrepareSlicesL0(multRingsSlice, param, level2CommInfo, level1CommInfo, level0CommInfo, perDataSize,
        inputMemSize));

    std::vector<std::vector<Slice>> multRingsUserMemSlice;
    CHK_RET(PrepareUserMemSlices(multRingsUserMemSlice, multRingsSlice, param, level2CommInfo, level1CommInfo,
        level0CommInfo, perDataSize, inputMemSize));

    if (DMAReduceFlag_ && (level1RankSize > 1 || level2RankSize > 1)) {
        // allgather输入放在CCL buffer上，通过设置nullptr指示要从CCL buffer获取输入
        opInfo.inputAddr = nullptr;
    }
    CHK_RET(RunIntraSeverAllGather(param.tag, execMem.inputMem, execMem.outputMem, execMem.count, dataType,
        multRingsSlice, param.stream, PROF_STAGE_2, 0, opInfoPtr, multRingsUserMemSlice));
    HCCL_INFO("allgather ring run successtopoType_[%u], agv[%u]", topoType_, isAllGatherV_);
    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::Getlevel1CommRank(SubCommInfo& level1CommInfo)
{
    HCCL_INFO("[nslbdp] Entry Getlevel1CommRank.");
    if (CheckCommSize(COMM_LEVEL2, COMM_INDEX_0 + 1) != HCCL_SUCCESS) {
        HCCL_INFO("[nslbdp] Getlevel1CommRank size not match.");
        return HCCL_E_UNAVAIL;
    }
    level1CommInfo = GetSubCommInfo(COMM_LEVEL2, COMM_INDEX_0);

    return HCCL_SUCCESS;
}

HcclResult CollAllGatherRingFor91093Executor::SelectTempAlg(std::unique_ptr<AlgTemplateBase> &level1TempAlg, u32 level1RankSize)
{
    HCCL_INFO("[nslbdp] Entry SelectTempAlg, level1RankSize = [%u].", level1RankSize);
    if (level1RankSize > 1) {
        if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NB) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NB, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-bruck algo inter-superPod.");
        } else if (algType_.algoLevel2 == AlgTypeLevel2::ALG_LEVEL2_NHR) {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_NHR, dispatcher_);
            HCCL_INFO("allgather ring: using nonuniform-hierarchical-ring algo inter-superPod.");
        } else {
            level1TempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(
                TemplateType::TEMPLATE_ALL_GATHER_RING, dispatcher_);
            HCCL_INFO("allgather ring: using ring algo inter-superPod.");
        }
        return HCCL_SUCCESS;
    }
    return HCCL_E_UNAVAIL;
}

REGISTER_EXEC("AllGatherRingFor91093Executor", AllGatherRingFor91093, CollAllGatherRingFor91093Executor);

} // namespace hccl
