/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alltoallv_mesh_read_only.h"
#include "log.h"
#include "alg_template_register.h"

namespace hccl {

static const u64 SMALL_SIZE = 262144;               // alltoall 小数据量边界，暂定256k
static const u64 MAX_DATA_BLOCK_SIZE = 16777216;    // 最大块数据大小，防止设置的CCLbuffer过大，导致数据块过大，mesh算法卡死
static const u64 MAX_SUB_NODE = 24;

AlltoAllVMeshReadOnly::AlltoAllVMeshReadOnly(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{
}

AlltoAllVMeshReadOnly::~AlltoAllVMeshReadOnly() {}

u64 AlltoAllVMeshReadOnly::GetGraphModeRemoteMemSize(u32 destRank)
{
    u64 memSize = 0;
    for (const OneSendRecvAddrInfo& readInfo : recvAddrInfo_[destRank]) {
        memSize = std::max(readInfo.remoteOffset + readInfo.remoteLength, memSize);
    }
    return memSize;
}

HcclResult AlltoAllVMeshReadOnly::InitParams(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &scratchPingMem,
    DeviceMem &scratchPongMem, HcclWorkflowMode workMode, Stream &mainStream, std::vector<Stream> &subStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
    u32 userRank, u32 intraRankSize, const std::vector<LINK> &links,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    userInput_ = userInput;
    userOutput_ = userOutput;
    scratchPingMem_ = scratchPingMem;
    scratchPongMem_ = scratchPongMem;
    workMode_ = workMode;
    mainStream_ = mainStream;
    subStreams_ = subStreams;
    meshSignalMainToSub_ = meshSignalMainToSub;
    meshSignalSubToMain_ = meshSignalSubToMain;
    userRank_ = userRank;
    intraRankSize_ = intraRankSize;
    links_ = links;
    allMeshAggregationSendRecvInfoPtr_ = &allMeshAggregationSendRecvInfo;
    return HCCL_SUCCESS;
}

u64 GetGlobalMaxUserInSize(const std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    u64 maxUserIn = 0;
    for (const auto& sendRecvInfo: allMeshAggregationSendRecvInfo) {
        u64 sendLengthSize = sendRecvInfo.sendLength.size();
        u64 sendOffsetSize = sendRecvInfo.sendOffset.size();
        CHK_PRT_RET(sendLengthSize != sendOffsetSize, HCCL_ERROR("invalid sendRecvInfo"), HCCL_E_PARA);
        for (u32 index = 0; index < sendLengthSize; index++) {
            u64 currRankUserIn = sendRecvInfo.sendLength[index] + sendRecvInfo.sendOffset[index];
            maxUserIn = std::max(maxUserIn, currRankUserIn);
        }
    }
    return maxUserIn;
}

HcclResult AlltoAllVMeshReadOnly::Prepare(DeviceMem &userInput, DeviceMem &userOutput, DeviceMem &scratchPingMem,
    DeviceMem &scratchPongMem, StageAlltoAllVAddrInfo &sendAddrInfo, StageAlltoAllVAddrInfo &recvAddrInfo,
    HcclWorkflowMode workMode, Stream &mainStream, std::vector<Stream> &subStreams,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalMainToSub,
    const std::vector<std::shared_ptr<LocalNotify>> &meshSignalSubToMain,
    u32 userRank, u32 intraRankSize, const std::vector<LINK> &links,
    std::vector<SendRecvInfo> &allMeshAggregationSendRecvInfo)
{
    CHK_RET(InitParams(userInput, userOutput, scratchPingMem, scratchPongMem, workMode, mainStream, subStreams,
        meshSignalMainToSub, meshSignalSubToMain, userRank, intraRankSize, links, allMeshAggregationSendRecvInfo));

    CHK_PRT_RET(intraRankSize_ == 0, HCCL_ERROR("[AlltoAllVMeshReadOnly][Prepare]intraRankSize_ is zero."),
        HCCL_E_PARA);
    intraRank_ = userRank_ % intraRankSize_;
    dataBlockSize_ = (scratchPingMem.size() / std::max(1u, intraRankSize_ - 1u));
    if (dataBlockSize_ > HCCL_MIN_SLICE_ALIGN_910B) {
        dataBlockSize_ = (dataBlockSize_ / HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }
    useScratchPingMem_ = true;
    for (u32 dataIndex = 0; dataIndex < intraRankSize_; dataIndex++) {
        sendIndexTrace_[dataIndex] = {0u, 0u};
        recvIndexTrace_[dataIndex] = {0u, 0u};
    }
    sendAddrInfo_.clear();
    for (auto& currRankSendInfo : sendAddrInfo) {
        std::vector<OneSendRecvAddrInfo> dataDescription;
        dataDescription.assign(currRankSendInfo.second.begin(), currRankSendInfo.second.end());
        sendAddrInfo_[currRankSendInfo.first] = dataDescription;
    }
    recvAddrInfo_.clear();
    for (auto& currRankRecvInfo : recvAddrInfo) {
        std::vector<OneSendRecvAddrInfo> dataDescription;
        dataDescription.assign(currRankRecvInfo.second.begin(), currRankRecvInfo.second.end());
        recvAddrInfo_[currRankRecvInfo.first] = dataDescription;
    }
    for (u32 destRank = 0; destRank < intraRankSize_; destRank++) {
        if (destRank == intraRank_) {
            continue;
        }
        const LINK& intraNeighboorTransport = links_[destRank];
        void* remDMAMemPtr = nullptr;
        if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
            DeviceMem remoteScratchPing = DeviceMem::create(static_cast<u8 *>(remDMAMemPtr), scratchPingMem_.size());
            CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::OUTPUT_MEM, &remDMAMemPtr));
            DeviceMem remoteScratchPong = DeviceMem::create(static_cast<u8 *>(remDMAMemPtr), scratchPingMem_.size());
            destRankRemoteMem_[destRank] = {remoteScratchPing, remoteScratchPong};
        } else {
            CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &remDMAMemPtr));
            DeviceMem remoteScratchPing = DeviceMem::create(static_cast<u8 *>(remDMAMemPtr),
                GetGraphModeRemoteMemSize(destRank));
            destRankRemoteMem_[destRank] = {remoteScratchPing, remoteScratchPing};
        }
    }
    return HCCL_SUCCESS;
}

std::string AlltoAllVMeshReadOnly::GetStreamIndexString()
{
    std::string res = "";
    for (auto& info : subStreamSendRecvInfo_) {
        u32 destRank = info.first;
        u32 streamIndex = (destRank > intraRank_ ? destRank - 1 : destRank);
        res += std::to_string(streamIndex) + ", ";
    }
    return res;
}

HcclResult AlltoAllVMeshReadOnly::RunAsync()
{
    u64 maxUserIn = GetGlobalMaxUserInSize(*allMeshAggregationSendRecvInfoPtr_);
    if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE && maxUserIn <= SMALL_SIZE) {
        // 暂定小数据量为userIn <= 256k
        // 单算子模式小数据量，统一搬整个userIn到scratchPing，此时后续步骤和图模式归一
        DeviceMem dst = scratchPingMem_.range(0, userInput_.size());
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, userInput_, mainStream_));
        CHK_RET(RunAlltoall());
    } else if (workMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        // 单算子模式，已处理字节对齐
        CHK_RET(RunAlltoallPingPong());
    } else {
        // 图模式
        CHK_RET(RunAlltoall());
    }
    return HCCL_SUCCESS;
}

// 主流只需要通知当前子步骤需要收发数据的 SDMA 流，减少同步开销
HcclResult AlltoAllVMeshReadOnly::NotifySubStreamStart()
{
    u64 count = 0;
    for (auto& sdmaInfo : subStreamSendRecvInfo_) {
        u32 destRank = sdmaInfo.first;
        u32 streamIndex = (destRank > intraRank_ ? destRank - 1 : destRank);
        CHK_RET(LocalNotify::Post(mainStream_, dispatcher_, meshSignalMainToSub_[streamIndex], INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(subStreams_[streamIndex], dispatcher_, meshSignalMainToSub_[streamIndex],
            INVALID_VALUE_STAGE));
        count++;
        if (count == MAX_SUB_NODE) {
            count = 0;
            CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
        }
    }
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][NotifyIntraStreamStart] userRank [%u] main stream notify sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVMeshReadOnly::WaitSubStreamFinish()
{
    for (auto& sdmaInfo : subStreamSendRecvInfo_) {
        u32 destRank = sdmaInfo.first;
        u32 streamIndex = (destRank > intraRank_ ? destRank - 1 : destRank);
        CHK_RET(LocalNotify::Post(subStreams_[streamIndex], dispatcher_, meshSignalSubToMain_[streamIndex],
            INVALID_VALUE_STAGE));
        CHK_RET(LocalNotify::Wait(mainStream_, dispatcher_, meshSignalSubToMain_[streamIndex], INVALID_VALUE_STAGE));
    }
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][WaitIntraStreamFinish] userRank [%u] main stream wait sdma stream [%s]",
        userRank_, GetStreamIndexString().c_str());
    return HCCL_SUCCESS;
}

u32 AlltoAllVMeshReadOnly::CalcNumSubStep()
{
    sendNumSubStep_.clear();
    recvNumSubStep_.clear();
    u32 numSubStep = 0;
    u64 maxDataBlock = 0;
    for (const SendRecvInfo& info : *allMeshAggregationSendRecvInfoPtr_) {
        for (u64 sendLen : info.sendLength) {
            maxDataBlock = std::max(maxDataBlock, sendLen);
        }
    }
    for (u32 destRank = 0; destRank < intraRankSize_; destRank++) {
        if (destRank == intraRank_) {
            continue;
        }
        u64 totalSendLen = 0;
        const std::vector<OneSendRecvAddrInfo>& currRankSendInfo = sendAddrInfo_[destRank];
        for (const OneSendRecvAddrInfo& dataBlockInfo : currRankSendInfo) {
            u64 addrOffset = (dataBlockInfo.remoteOffset % HCCL_MIN_SLICE_ALIGN_910B);
            u64 sendTail = (totalSendLen % HCCL_MIN_SLICE_ALIGN_910B);
            totalSendLen += (addrOffset >= sendTail ? addrOffset : (addrOffset + HCCL_MIN_SLICE_ALIGN_910B));
            totalSendLen -= sendTail;
            totalSendLen += maxDataBlock;
        }
        u32 currRankSendSubStep = ((totalSendLen + dataBlockSize_ - 1) / dataBlockSize_);
        sendNumSubStep_[destRank] = currRankSendSubStep;
        u64 totalRecvLen = 0;
        const std::vector<OneSendRecvAddrInfo>& currRankRecvInfo = recvAddrInfo_[destRank];
        for (const OneSendRecvAddrInfo& dataBlockInfo : currRankRecvInfo) {
            u64 addrOffset = (dataBlockInfo.localOffset % HCCL_MIN_SLICE_ALIGN_910B);
            u64 recvTail = (totalRecvLen % HCCL_MIN_SLICE_ALIGN_910B);
            totalRecvLen += (addrOffset >= recvTail ? addrOffset : (addrOffset + HCCL_MIN_SLICE_ALIGN_910B));
            totalRecvLen -= recvTail;
            totalRecvLen += maxDataBlock;
        }
        u32 currRankRecvSubStep = ((totalRecvLen + dataBlockSize_ - 1) / dataBlockSize_);
        recvNumSubStep_[destRank] = currRankRecvSubStep;
        numSubStep = std::max(numSubStep, std::max(currRankSendSubStep, currRankRecvSubStep));
    }
    return numSubStep;
}

HcclResult AlltoAllVMeshReadOnly::SendRecvData()
{
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][SendRecvData] userRank [%u] sdma stream [%s] wait main stream",
        userRank_, GetStreamIndexString().c_str());
    for (auto& sdmaInfo : subStreamSendRecvInfo_) {
        u32 destRank = sdmaInfo.first;
        u32 streamIndex = (destRank > intraRank_ ? destRank - 1 : destRank);
        const std::vector<ReadDataBlock>& readInfo = sdmaInfo.second.readInfo;
        Stream& currStream = subStreams_[streamIndex];
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, currStream, dispatcher_));
        const LINK& readTransport = links_[destRank];
        CHK_RET(readTransport->TxAck(currStream));
        CHK_RET(readTransport->RxAck(currStream));
        if (readInfo.size() > 0) {
            for (const ReadDataBlock& readData : readInfo) {
                DeviceMem srcMem = (useScratchPingMem_ ? destRankRemoteMem_[destRank].remoteScratchPingMem :
                    destRankRemoteMem_[destRank].remoteScratchPongMem).range(readData.remoteOffset,
                    readData.recvLen);
                DeviceMem dstMem = userOutput_.range(readData.recvOffset, readData.recvLen);
                CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dstMem, srcMem, currStream,
                    readTransport->GetRemoteRank(), readTransport->GetLinkType()));
                HCCL_DEBUG("[AlltoAllVMeshReadOnly][SendRecvData] userRank [%u], sdma stream [%llu] read "
                    "data from remote [%s] offset [%llu] len [%llu] to local [%llu]", userRank_, streamIndex,
                    useScratchPingMem_ ? "IntraSendPingMem" : "IntraSendPongMem",
                    readData.remoteOffset, readData.recvLen, readData.recvOffset);
            }
        }
        // 当前从mesh内其他卡的cclInput读了数据之后，其接收到的下一块rdma数据在cclIn，反之同理
        CHK_RET(readTransport->TxDataSignal(currStream));
        CHK_RET(readTransport->RxDataSignal(currStream));
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, currStream, dispatcher_));
    }
    HCCL_DEBUG("[AlltoallPipelineMeshPairwiseBcopy][SendRecvData] userRank [%u], sdma stream [%s] notify "
        "main stream", userRank_, GetStreamIndexString().c_str());
    useScratchPingMem_ ^= true;
    return HCCL_SUCCESS;
}

void AlltoAllVMeshReadOnly::UpdateCurrRankSendInfo(u32 destRank, std::vector<SendDataBlock>& sendInfo)
{
    u64 remainScratchLen = dataBlockSize_;
    u64 scratchOffset = (destRank > intraRank_ ? (destRank - 1) : destRank) * dataBlockSize_;
    u32 dataBlockIndex = sendIndexTrace_[destRank].dataIndex;
    u64 dataOffset = sendIndexTrace_[destRank].dataOffset;
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][UpdateCurrRankSendInfo] destRank [%u]", destRank);
    while (remainScratchLen > 0 && dataBlockIndex < sendAddrInfo_[destRank].size()) {
        OneSendRecvAddrInfo sendDataBlock = sendAddrInfo_[destRank][dataBlockIndex];
        u64 currRemoteOffset = sendDataBlock.remoteOffset + dataOffset;
        u64 addrOffset = (currRemoteOffset % HCCL_MIN_SLICE_ALIGN_910B);
        u64 sendTail = (scratchOffset % HCCL_MIN_SLICE_ALIGN_910B);
        u64 emptyLen = (addrOffset >= sendTail ? addrOffset : (addrOffset + HCCL_MIN_SLICE_ALIGN_910B)) -
            sendTail;
        if (emptyLen >= remainScratchLen) {
            break;
        }
        remainScratchLen -= emptyLen;
        scratchOffset += emptyLen;
        u64 currDataRemainLen = sendDataBlock.localLength - dataOffset;
        u64 sendLen = std::min(remainScratchLen, currDataRemainLen);
        u64 userInOffset = sendDataBlock.localOffset + dataOffset;
        sendInfo.push_back({sendLen, userInOffset, scratchOffset});
        scratchOffset += sendLen;
        dataBlockIndex = (currDataRemainLen > remainScratchLen ? dataBlockIndex : (dataBlockIndex + 1));
        dataOffset = (currDataRemainLen > remainScratchLen ? (dataOffset + sendLen) : 0);
        remainScratchLen -= sendLen;
    }
    sendIndexTrace_[destRank] = {dataBlockIndex, dataOffset};
}

void AlltoAllVMeshReadOnly::UpdateCurrRankRecvInfo(u32 destRank, std::vector<ReadDataBlock>& readInfo)
{
    u64 remainScratchLen = dataBlockSize_;
    u64 scratchOffset = (destRank > intraRank_ ? intraRank_ : (intraRank_ - 1)) * dataBlockSize_;
    u32 dataBlockIndex = recvIndexTrace_[destRank].dataIndex;
    u64 dataOffset = recvIndexTrace_[destRank].dataOffset;
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][UpdateCurrRankRecvInfo] destRank [%u]", destRank);
    while (remainScratchLen > 0 && dataBlockIndex < recvAddrInfo_[destRank].size()) {
        OneSendRecvAddrInfo recvDataBlock = recvAddrInfo_[destRank][dataBlockIndex];
        u64 currLocalOffset = recvDataBlock.localOffset + dataOffset;
        u64 addrOffset = (currLocalOffset % HCCL_MIN_SLICE_ALIGN_910B);
        u64 recvTail = (scratchOffset % HCCL_MIN_SLICE_ALIGN_910B);
        u64 emptyLen = (addrOffset >= recvTail ? addrOffset : (addrOffset + HCCL_MIN_SLICE_ALIGN_910B)) -
            recvTail;
        if (emptyLen >= remainScratchLen) {
            break;
        }
        remainScratchLen -= emptyLen;
        scratchOffset += emptyLen;
        u64 currDataRemainLen = recvDataBlock.remoteLength - dataOffset;
        u64 recvLen = std::min(remainScratchLen, currDataRemainLen);
        u64 recvOffset = recvDataBlock.localOffset + dataOffset;
        readInfo.push_back({recvLen, scratchOffset, recvOffset});
        scratchOffset += recvLen;
        dataBlockIndex = (currDataRemainLen > remainScratchLen ? dataBlockIndex : (dataBlockIndex + 1));
        dataOffset = (currDataRemainLen > remainScratchLen ? (dataOffset + recvLen) : 0);
        remainScratchLen -= recvLen;
    }
    recvIndexTrace_[destRank] = {dataBlockIndex, dataOffset};
}

void AlltoAllVMeshReadOnly::UpdateOpBaseSubStreamInfo(u32 step)
{
    subStreamSendRecvInfo_.clear();
    HCCL_DEBUG("[AlltoAllVMeshReadOnly][UpdateSubStreamInfo] userRank [%u], sub step [%llu]", userRank_, step);
    for (u32 destRank = 0; destRank < intraRankSize_; destRank++) {
        if (destRank == intraRank_) {
            continue;
        }
        u32 currDestSendStep = sendNumSubStep_[destRank];
        u32 currDestRecvStep = recvNumSubStep_[destRank];
        std::vector<SendDataBlock> sendInfo;
        if (step < currDestSendStep) {
            UpdateCurrRankSendInfo(destRank, sendInfo);
        }
        std::vector<ReadDataBlock> readInfo;
        if (step < currDestRecvStep) {
            UpdateCurrRankRecvInfo(destRank, readInfo);
        }
        subStreamSendRecvInfo_[destRank] = {sendInfo, readInfo};
    }
}

void AlltoAllVMeshReadOnly::UpdateGraphModeSubStreamInfo()
{
    subStreamSendRecvInfo_.clear();
    for (u32 destRank = 0; destRank < intraRankSize_; destRank++) {
        if (destRank == intraRank_) {
            continue;
        }
        std::vector<SendDataBlock> sendInfo;
        sendInfo.push_back({0, 0, 0});
        std::vector<ReadDataBlock> readInfo;
        std::vector<OneSendRecvAddrInfo> currRankReadInfo = recvAddrInfo_[destRank];
        for (const OneSendRecvAddrInfo& readData : currRankReadInfo) {
            u64 recvLen = readData.localLength;
            u64 remoteOffset = readData.remoteOffset;
            u64 recvOffset = readData.localOffset;
            readInfo.push_back({recvLen, remoteOffset, recvOffset});
        }
        subStreamSendRecvInfo_[destRank] = {sendInfo, readInfo};
    }
}

HcclResult AlltoAllVMeshReadOnly::PrepareIntraData()
{
    DeviceMem scratchMem = (useScratchPingMem_ ? scratchPingMem_ : scratchPongMem_);
    for (auto& sdmaInfo : subStreamSendRecvInfo_) {
        const std::vector<SendDataBlock>& sendInfo = sdmaInfo.second.sendInfo;
        for (const SendDataBlock& data : sendInfo) {
            DeviceMem src = userInput_.range(data.userInOffset, data.sendLen);
            DeviceMem dst = scratchMem.range(data.scratchOffset, data.sendLen);
            CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
            HCCL_DEBUG("[AlltoAllVMeshReadOnly][PrepareIntraData]userRank [%u] copy from userInput [%u] len [%u] to "
                "scratch [%u]", userRank_, data.userInOffset, data.sendLen, data.scratchOffset);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVMeshReadOnly::LocalCopy()
{
    const std::vector<OneSendRecvAddrInfo>& localData = sendAddrInfo_[intraRank_];
    for (const OneSendRecvAddrInfo& data : localData) {
        DeviceMem src = userInput_.range(data.localOffset, data.localLength);
        DeviceMem dst = userOutput_.range(data.remoteOffset, data.remoteLength);
        CHK_RET(HcclD2DMemcpyAsync(dispatcher_, dst, src, mainStream_));
        HCCL_DEBUG("[AlltoAllVMeshReadOnly][LocalCopy]userRank [%u] copy from userInput [%u] len [%u] to "
            "userOutput [%u]", userRank_, data.localOffset, data.localLength, data.remoteOffset);
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVMeshReadOnly::RunAlltoallPingPong()
{
    u32 totalStep = CalcNumSubStep();
    UpdateOpBaseSubStreamInfo(0);
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(PrepareIntraData());
    for (u32 step = 0; step < totalStep; step++) {
        CHK_RET(NotifySubStreamStart());
        CHK_RET(SendRecvData());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
        if (step == (totalStep - 1)) {
            CHK_RET(LocalCopy());
        } else {
            UpdateOpBaseSubStreamInfo(step + 1);
            CHK_RET(PrepareIntraData());
        }
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
        CHK_RET(WaitSubStreamFinish());
        CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    }
    return HCCL_SUCCESS;
}

HcclResult AlltoAllVMeshReadOnly::RunAlltoall()
{
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    UpdateGraphModeSubStreamInfo();
    CHK_RET(NotifySubStreamStart());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(SendRecvData());
    CHK_RET(LocalCopy());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    CHK_RET(WaitSubStreamFinish());
    CHK_RET(ExecEmptyTask(userInput_, userOutput_, mainStream_, dispatcher_));
    return HCCL_SUCCESS;
}

REGISTER_TEMPLATE(TemplateType::TEMPLATE_ALL_2_ALL_V_MESH_READ_ONLY, AlltoAllVMeshReadOnly);
} // namespace hccl