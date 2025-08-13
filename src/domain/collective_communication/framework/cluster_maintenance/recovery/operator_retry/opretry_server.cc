/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "opretry_server.h"
#include "externalinput_pub.h"
#include "heartbeat.h"

namespace hccl {

HcclResult CreateOpRetryServerByState(RetryState state, RetryContext* retryCtx)
{
    HCCL_INFO("[OpRetry][Server]CreateOpRetryServerByState state[%s]", GetReadableState(state));
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    switch (state) {
        case RETRY_STATE_SERVER_RUNNING: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRunning>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STETA_HANDLE_ALL_ERR: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerHandleError>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_SERVER_RETRY_FAIL: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerRetryFail>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_WAIT_LINK_CHECKED:
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerWaitLinkInfo>()), return HCCL_E_PTR);
            break;
        case RETRY_STATE_WAIT_AICPU_STOPED:
        case RETRY_STATE_WAIT_STREAM_STOPED:
        case RETRY_STATE_WAIT_STREAM_CLEARED:
        case RETRY_STATE_WAIT_STOP_TRANSPORT:
        case RETRY_STATE_WAIT_NOTIFY_RESETED:
        case RETRY_STATE_WAIT_RESUME_TRANSPORT:
        case RETRY_STATE_WAIT_CHECK_INFO:
        case RETRY_STATE_WAIT_CAN_RETRY: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerWaitResp>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CHECK_ALL_LINK:
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerCheckAllLink>()), return HCCL_E_PTR);
            break;
        case RETRY_STATE_CMD_RESUME_TRANSPORT: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueChangeLinkAndResume>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CMD_STOP_AICPU:
        case RETRY_STATE_CMD_STOP_STREAM:
        case RETRY_STATE_CMD_CLEAR_STREAM:
        case RETRY_STATE_CMD_STOP_TRANSPORT:
        case RETRY_STATE_CMD_CHECK_LINK:
        case RETRY_STATE_CMD_RESET_NOTIFY:
        case RETRY_STATE_CMD_CHECK:
        case RETRY_STATE_CMD_CAN_RETRY: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueCmd>()), return HCCL_E_PTR);
            break;
        }
        case RETRY_STATE_CHECK_OP: {
            EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerCheckOp>()), return HCCL_E_PTR);
            break;
        }
        // 检查各agennt主动接轨信息，并发送cmd命令
        case RETRY_STATE_CMD_PLAN_SWITCH_NIC : {
            EXECEPTION_CATCH((retryPtr = std::make_shared<SwitchNicServerCheckAllSwitchRanks>()), return HCCL_E_PTR);
            break;
        }
        default: {
            HCCL_ERROR("[OpRetry][Server]CreateOpRetryServerByState failed, state[%s] is invalid",
                GetReadableState(state));
            return HCCL_E_NOT_SUPPORT;
        }
    }
    retryCtx->SetRetryState(state, retryPtr);
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerBase::ProcessError(RetryContext* retryCtx)
{
    HCCL_ERROR("[%s]OpRetryServer run fail, rankId[%u], state[%s]", __func__, retryCtx->rankId_,
        retryCtx->GetReadableCtxState());
    // 状态切换至RETRY_STATE_SERVER_RETRY_FAIL
    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRunning::ProcessEvent(RetryContext* retryCtx)
{
    if (retryCtx->errorRankList_.size() > 0) {
        // 若当前errorRankList_中有未处理的errorRank，则先进行处理
        HCCL_RUN_INFO("[OpRetry][Server]deal rank from errorRankList_, size[%d]", retryCtx->errorRankList_.size());
        CHK_RET(CreateOpRetryServerByState(RETRY_STETA_HANDLE_ALL_ERR, retryCtx));
        return HCCL_SUCCESS;
    }

    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_KEEP_INTERVAL);
    // 轮询接收agent信息
    for (auto &it : retryCtx->serverSockets_) {
        const u32 &agentId = it.first;
        // 若对端已经关闭, 则不再轮询
        if (disableAgent_.find(agentId) != disableAgent_.end()) {
            continue;
        }

        // 记录时间, 检测和对端上一次通信时间是否超过保活时间
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        if (lastRecvTimes_.find(agentId) == lastRecvTimes_.end()) {
            lastRecvTimes_.insert(std::make_pair(agentId, curTime));
        }

        // 轮询接收agent状态机信息
        HcclResult ret = WaitResponse(it.second.socket, it.second.retryInfo);
        if (ret == HCCL_SUCCESS) { // 成功接收到数据
            // 新增逻辑：检查是否需要接收 ActiveSwitchInfo，进入主动接轨校验阶段
            if (it.second.retryInfo.retryState == RETRY_STATE_SEND_SWITCH_INFO) {
                // 1. 接收剩余字段
                ret = RecvActiveSwitchInfo(it.second.socket, agentId, it.second.switchInfo);
                if (ret != HCCL_SUCCESS) {
                    disableAgent_.insert(agentId);
                } else {
                    retryCtx->switchInfoMap_[agentId] = it.second.switchInfo;
                    HCCL_INFO("[SwitchNic][Server] recv first ActiveSwitchInfo from rank[%u] while running", agentId);
                    // 2. 此时 activeInfo 包含完整数据，可用于后续逻辑
                    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_PLAN_SWITCH_NIC, retryCtx));
                    return HCCL_SUCCESS;
                }
            }
            RetryState nextState = RETRY_STATE_SERVER_RUNNING;
            CHK_RET(ParaseErrorCode(retryCtx, it.second, nextState));
            if (nextState != RETRY_STATE_SERVER_RUNNING) {
                // 收到第一个报错后加入errorRankList_中，并切换到RETRY_STETA_HANDLE_ALL_ERR状态
                HCCL_RUN_INFO("[OpRetry][Server]agent[%u] tag[%s] index[%u] find error, insert to errorRankList_", 
                    agentId, it.second.retryInfo.opInfo.opId.tag, it.second.retryInfo.opInfo.opId.index);
                retryCtx->errorRankList_.insert(std::make_pair(agentId, it.second.retryInfo.opInfo.opId));
                CHK_RET(CreateOpRetryServerByState(RETRY_STETA_HANDLE_ALL_ERR, retryCtx));
                return HCCL_SUCCESS;
            }
            lastRecvTimes_[agentId] = curTime;
        } else if (ret == HCCL_E_AGAIN) { // 未接收到数据
            // 校验是否超时
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastRecvTimes_[agentId]);
            if (elapsed > timeout) {
                HCCL_WARNING("[OpRetry][Server]OpRetryServerRunning recv Retry Frame from agentId[%u] timeout",
                    agentId);
                lastRecvTimes_[agentId] = curTime;
            }
        } else { // 接收数据失败
            disableAgent_.insert(agentId);
            HCCL_RUN_INFO("[OpRetry][Server]WaitResponse from agentId[%u] fail, ret[%u]", agentId, ret);
        }
    }

    // 轮询间隔
    SaluSleep(OP_RETRY_RUNNING_POLL_INTERVAL);
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerHandleError::ProcessEvent(RetryContext* retryCtx)
{
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_WAIT_CAN_RETRY_RANK);
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    u32 waitTime = GetExternalInputRetryHoldTime();
    while (true) {
        CHK_PRT_RET(retryCtx->isServerStateWaitResume_, HCCL_RUN_INFO("[OpRetry][Server]switched state form wait handle error to wait resume"), HCCL_SUCCESS);
        // 判断是否超时
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        if (elapsed > timeout) {
            HCCL_ERROR("[OpRetry][Server]aicpu can not retry, opname is inconsistent");
            for (auto &it : retryCtx->serverSockets_) {
                auto tag = std::string(reinterpret_cast<const char*>(it.second.retryInfo.opInfo.opId.tag));
                HCCL_ERROR("[OpRetry][Server]OpRetryHandle retryinfo rank[%u] tag[%s] index[%u] IpInfo[%s]", it.first,
                    tag.c_str(), it.second.retryInfo.opInfo.opId.index, it.second.retryInfo.dfxIpInfo);
            }
            return HCCL_E_TIMEOUT;
        }

        // 轮询接收agent信息,只期望收上开故障信息
        for (auto &it : retryCtx->serverSockets_) {
            const u32 &agentId = it.first;
            if (retryCtx->errorRankList_.find(agentId) != retryCtx->errorRankList_.end()) {
                // 若当前rank已在errorRankList_，则不进行轮训
                continue;
            }
            // 轮询接收agent状态机信息
            HcclResult ret = WaitResponse(it.second.socket, it.second.retryInfo);
            if (ret == HCCL_SUCCESS) { // 成功接收到数据
                RetryState nextState = RETRY_STATE_SERVER_RUNNING;
                CHK_RET(ParaseErrorCode(retryCtx, it.second, nextState));
                if (nextState != RETRY_STATE_SERVER_RUNNING) {
                    // 当前rank报错，收集到errorRankList_后统一处理
                    HCCL_RUN_INFO("[OpRetry][Server]agent[%u] tag[%s] index[%u] find error, insert to errorRankList_", 
                        agentId, it.second.retryInfo.opInfo.opId.tag, it.second.retryInfo.opInfo.opId.index);
                    retryCtx->errorRankList_.insert(std::make_pair(agentId, it.second.retryInfo.opInfo.opId));
                    continue;
                }
            } else if (ret == HCCL_E_AGAIN) {
                // 未收到数据，则发送一个保活数据给agent
                RetryCommandInfo commandInfo;
                commandInfo.command= RETRY_CMD_RUNNING;
                CHK_RET(IssueCommandWithOpId(it.second.socket, commandInfo));
            }
        }

        bool isFoundSendRecv = false;
        std::set <u32> errorRank;
        for (auto iter= retryCtx->errorRankList_.begin(); iter!= retryCtx->errorRankList_.end();++iter) {
            errorRank.insert(iter->first);
        }
        // 对errorRankList_中rank进行遍历
        for (auto rank:errorRank){
            if (retryCtx->errorRankList_[rank].isSendRecv) {
                // 当前报错rank中存在send/recv算子，优先处理send/recv算子
                isFoundSendRecv = true;
                auto curOpId = retryCtx->errorRankList_[rank];
                uint32_t remoteRank = (rank==curOpId.detRank) ? curOpId.srcRank : curOpId.detRank;
                auto &remoteOpId = retryCtx->serverSockets_[remoteRank].retryInfo.opInfo.opId;
                std::string curTag = std::string(reinterpret_cast<const char*>(curOpId.tag));
                std::string remoteTag = std::string(reinterpret_cast<const char*>(remoteOpId.tag));
                //sendrecv没有下边那两字段
                auto remoteSendTag = std::string(reinterpret_cast<const char*>(remoteOpId.bsrInfo[HCCL_SEND].bsrTag));
                auto remoteRecvTag = std::string(reinterpret_cast<const char*>(remoteOpId.bsrInfo[HCCL_RECV].bsrTag));
                HCCL_RUN_INFO("[OpRetry][Server]curRank[%u], tag[%s], index[%u], startTaskComplete[%d]"\
                               "remoteRank[%u], remotetag[%s], remoteindex[%u], remoteStartTaskComplete[%d]"\
                               "Sendtag[%s], sendindex[%u]"\
                               "Recvtag[%s], recvindex[%u]",
                    rank, curTag.c_str(), curOpId.index, curOpId.isBsrTaskStart,
                    remoteRank, remoteTag.c_str(), remoteOpId.index, remoteOpId.isBsrTaskStart,
                    remoteSendTag.c_str(), remoteOpId.bsrInfo[HCCL_SEND].index,
                    remoteRecvTag.c_str(), remoteOpId.bsrInfo[HCCL_RECV].index);
                if (curOpId.opType == HcclCMDType::HCCL_CMD_BATCH_SEND_RECV && !remoteOpId.isBsrTaskStart) {
                	continue;
                }
                // 如果对端也停在同一个send/recv算子，则触发该算子的重执行
                if ((curTag == remoteSendTag && curOpId.index == remoteOpId.bsrInfo[HCCL_SEND].index) || 
                    (curTag == remoteRecvTag && curOpId.index == remoteOpId.bsrInfo[HCCL_RECV].index) ||
                    (curTag == remoteTag && curOpId.index == remoteOpId.index)) {
                    // 从errorRankList_中清除本端和对端rank
                    retryCtx->errorRankList_.erase(rank);
                    if (retryCtx->errorRankList_.find(remoteRank) != retryCtx->errorRankList_.end()) {
                        if (curTag == remoteTag && curOpId.index == remoteOpId.index)
                        {
                            HCCL_RUN_INFO("[OpRetry][Server]delete remoteRank[%u] from errorRankList_", remoteRank);
                            retryCtx->errorRankList_.erase(remoteRank);
                        }
                    }
                    // 触发重执行
                    HCCL_RUN_INFO("[OpRetry][Server]begin to exec retry of tag[%s] from rank[%u] and rank[%u]",
                        curTag.c_str(), rank, remoteRank);
                    retryCtx->needRetryServerRanks_.clear();
                    CHK_PRT(SetNeedRetryServerRank(retryCtx, curOpId));
                    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_STOP_AICPU, retryCtx));
                    return HCCL_SUCCESS;
                }
            }
        }

        if (!isFoundSendRecv) {
            // 如果没有找到send/recv算子，判断所有rank是否停在同一个算子
            bool isAllTagSame = true;
            u32 firstErrorRank = *(errorRank.begin());
            auto curOpId = retryCtx->errorRankList_[firstErrorRank];
            auto curTag = std::string(reinterpret_cast<const char*>(curOpId.tag));
            for (auto &it : retryCtx->serverSockets_) {
                auto remoteTag = std::string(reinterpret_cast<const char*>(it.second.retryInfo.opInfo.opId.tag));
                if (curTag != remoteTag) {
                    isAllTagSame = false;
                    break;
                }
            }
            if (isAllTagSame) {
                retryCtx->errorRankList_.clear();
                // 所有rank停在同一个算子，开始重执行
                HCCL_RUN_INFO("[OpRetry][Server]begin to exec retry of tag[%s] from rank[%u]",
                    curTag.c_str(), firstErrorRank);
                retryCtx->needRetryServerRanks_.clear();
                CHK_PRT(SetNeedRetryServerRank(retryCtx, curOpId));
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_STOP_AICPU, retryCtx));
                return HCCL_SUCCESS;
            }
        }
        errorRank.clear();
        SaluSleep(waitTime * TIME_MS_TO_US);
        HCCL_INFO("[OpRetry][Server]no rank can retry, wait for [%u]ms for collect all error rank", waitTime);
    }
}

HcclResult OpRetryServerHandleError::SetNeedRetryServerRank(RetryContext* retryCtx, const HcclOpIdentifier &opId)
{
    if (opId.isSendRecv) {
        // 在send/recv场景下，仅需对本端和对端进行重执行即可
        if (retryCtx->serverSockets_.find(opId.srcRank) == retryCtx->serverSockets_.end() ||
            retryCtx->serverSockets_.find(opId.detRank) == retryCtx->serverSockets_.end()) {
            HCCL_ERROR("[OpRetry][Server]srcRank[%u] or detRank[%u] isn't in serverSockets_", 
                opId.srcRank, opId.detRank);
            return HCCL_E_INTERNAL;
        }
        retryCtx->needRetryServerRanks_.push_back(opId.srcRank);
        retryCtx->needRetryServerRanks_.push_back(opId.detRank);
        retryCtx->curFaultOpId = opId;
        HCCL_INFO("[OpRetry][Server]set needRetryServerRank[%u] for send/recv success: srcRank=[%u],detRank=[%u],"
            "tag =[%s], streamid =%u",
            retryCtx->needRetryServerRanks_.size(), opId.srcRank, opId.detRank, opId.tag, opId.streamId);
    } else {
        // 其余场景下需要对所有rank进行重执行
        for (auto &it : retryCtx->serverSockets_) {
            retryCtx->curFaultOpId = opId;
            retryCtx->needRetryServerRanks_.push_back(it.first);
        }
        HCCL_DEBUG("[OpRetry][Server]set needRetryServerRank[%u] success", retryCtx->needRetryServerRanks_.size());
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRunning::ParaseErrorCode(RetryContext* retryCtx, HcclAgentRetryInfo &agentInfo, RetryState &nextState)
{
    // 处理接收到的数据
    KfcError errorCode = agentInfo.retryInfo.opInfo.execStatus.kfcError;
    switch (errorCode) {
        case KfcError::kNone: { // 发送保活数据
            //保活数据携带一个空的opid
            RetryCommandInfo commandInfo;
            commandInfo.command= RETRY_CMD_RUNNING;
            CHK_RET(IssueCommandWithOpId(agentInfo.socket, commandInfo));
            break;
        }
        case KfcError::kRdma:
            retryCtx->isRdmaError = true;
        case KfcError::kExecInplace:
        case KfcError::kSdma: { // 处理ERROR
            nextState = RETRY_STATE_CMD_STOP_AICPU;
            HCCL_RUN_INFO("[OpRetry][Server]OpRetryServerRunning recv ErrorCode[%d] from rank[%u]",
                errorCode, agentInfo.retryInfo.rankId);
            break;
        }
        default: { // 不支持的ErrorCode
            HCCL_ERROR("[OpRetry][Server]OpRetryServerRunning recv invalid ErrorCode[%d] from rank[%u]",
                errorCode, agentInfo.retryInfo.rankId);
            break;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerIssueCmd::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    RetryState curState = retryCtx->GetRetryState();
    // 获取下一个状态
    auto itState = RETRY_SERVER_STATE_TRANSFER_LABEL.find(curState);
    CHK_PRT_RET(itState == RETRY_SERVER_STATE_TRANSFER_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd fail, state[%s] is not in RETRY_SERVER_STATE_TRANSFER_LABEL",
            GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState nextState = itState->second;

    // 发送命令
    auto itCommand = RETRY_SERVER_STATE_TO_CMD_LABEL.find(curState);
    CHK_PRT_RET(itCommand == RETRY_SERVER_STATE_TO_CMD_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd fail, state[%s] is not in RETRY_SERVER_STATE_TO_CMD_LABEL",
            GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryCommand command = itCommand->second;
    HCCL_INFO("[OpRetry][Server]OpRetryServerIssueCmd curState[%s], command[%s]", GetReadableState(curState),
        GetReadableCmd(command));

    for (auto rank : retryCtx->needRetryServerRanks_) {
        RetryCommandInfo commandInfo;
        commandInfo.command = command;
        commandInfo.opId = retryCtx->curFaultOpId;
        HCCL_INFO("[OpRetry][Server]IssueCommandWithOpId tag[%s], index[%u], srcRank[%u], detRank[%u], isSendRecv[%d]," 
            "streamid[%u]",
            commandInfo.opId.tag, commandInfo.opId.index, commandInfo.opId.srcRank, 
            commandInfo.opId.detRank,commandInfo.opId.isSendRecv, commandInfo.opId.streamId);
        ret = IssueCommandWithOpId(retryCtx->serverSockets_[rank].socket, commandInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueCmd IssueCommand fail, curState[%s], command[%s]",
            GetReadableState(curState), GetReadableCmd(command)), ret);
    }
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerWaitResp::ProcessEvent(RetryContext* retryCtx)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_SEND_RECV_TIMEOUT);
    RetryState curState = retryCtx->GetRetryState();

    // 获取预期的下一个server状态
    auto serverTransferIt = RETRY_SERVER_STATE_TRANSFER_LABEL.find(curState);
    CHK_PRT_RET(serverTransferIt == RETRY_SERVER_STATE_TRANSFER_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp fail, state[%s] is not in RETRY_SERVER_STATE_TRANSFER_LABEL",
        GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState expectNextState = serverTransferIt->second;

    // 获取预期的对端agent状态
    auto agentStateIt = RETRY_SERVER_WAIT_AGENT_STATE_LABEL.find(curState);
    CHK_PRT_RET(agentStateIt == RETRY_SERVER_WAIT_AGENT_STATE_LABEL.end(),
        HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp fail, state[%s] is not in RETRY_SERVER_WAIT_AGENT_STATE_LABEL",
        GetReadableState(curState)), HCCL_E_INTERNAL);
    RetryState expectagentState = agentStateIt->second;
    HCCL_DEBUG("[OpRetry][Server]OpRetryServerWaitResp state[%s], expect next state[%s], expect peer state[%s]",
        GetReadableState(curState), GetReadableState(expectNextState), GetReadableState(expectagentState));

    std::set<u32> recvVaild;
    while (recvVaild.size() < retryCtx->needRetryServerRanks_.size()) {
        CHK_PRT_RET(retryCtx->isServerStateWaitResume_, HCCL_RUN_INFO("[OpRetry][Server]switched state form wait resp to wait resume"), HCCL_SUCCESS);
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp timeout"), HCCL_E_TIMEOUT);

        for (auto rank : retryCtx->needRetryServerRanks_) {
            if (recvVaild.find(rank) != recvVaild.end()) {
                continue;
            }
            auto &agentRetryInfo = retryCtx->serverSockets_[rank];
            // 接收agent信息
            HcclResult ret = WaitResponse(agentRetryInfo.socket, agentRetryInfo.retryInfo);
            CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp WaitResponse fail, ret[%u]", ret), ret);

            RetryState dstState = agentRetryInfo.retryInfo.retryState;
            if (ret == HCCL_SUCCESS && dstState == expectagentState) { // 接收到对端信息且状态有效
                recvVaild.insert(rank);
                HCCL_INFO("[OpRetry][Server]OpRetryServerWaitResp recv success from dst[%u], state[%s]",
                    rank, GetReadableState(dstState));
            } else if (ret == HCCL_SUCCESS && dstState == RETRY_STATE_RESP_RUNNING_ERR) { // 对端重执行失败
                recvVaild.insert(rank);
                PrintAgentInfoAfterFail(retryCtx->serverSockets_, recvVaild);
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp dst rank[%u] with IpInfo[%s] retry fail, " \
                    "command all rank retry fail", rank, agentRetryInfo.retryInfo.dfxIpInfo);
                retryCtx->isNeedReportOpRetryErr = agentRetryInfo.retryInfo.isNeedReportOpRetryErr;
                HCCL_RUN_INFO("[OpRetry][Server]OpRetryServerWaitResp retry fail, isNeedReportOpRetryErr[%d]", retryCtx->isNeedReportOpRetryErr);
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
                return HCCL_SUCCESS;
            }
        }
    }

    CHK_RET(CreateOpRetryServerByState(expectNextState, retryCtx));
    return HCCL_SUCCESS;
}

void OpRetryServerWaitResp::PrintAgentInfoAfterFail(std::map<u32, HcclAgentRetryInfo> &serverSockets,
    std::set<u32> &recvVaild)
{
    for (auto it = serverSockets.begin(); it != serverSockets.end(); ++it) {
        if (recvVaild.find(it->first) == recvVaild.end()) { // 未接收到有效数据
            continue;
        }
        auto &opInfo = it->second.retryInfo.opInfo;
        const char* tag = reinterpret_cast<const char*>(opInfo.opId.tag);
        u32 index = opInfo.opId.index;
        const KfcStatus &aicpuState = opInfo.execStatus.kfcStatus;
        if (aicpuState == KfcStatus::kEnd) { // 该rank未下发算子，或算子已执行结束
            HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitResp dst[%u] with IpInfo[%s], hccl op not launch or "\
                "is complete, hccl aicpu can not retry", it->first, it->second.retryInfo.dfxIpInfo);
        }
        HCCL_RUN_INFO("[OpRetry][Server]Print rank[%u], tag[%s], index[%u], aicpuStatus[%d]",
            it->first, tag, index, aicpuState);
    }
}

HcclResult OpRetryServerCheckOp::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = CheckRetryInfo(*retryCtx);
    RetryState nextState = (ret == HCCL_SUCCESS) ? RETRY_STATE_CMD_CAN_RETRY : RETRY_STATE_SERVER_RETRY_FAIL;

    if (ret == HCCL_E_OPRETRY_FAIL) {
        HCCL_RUN_INFO("[OpRetry][Server][CheckRetryInfo] can not retry, RETRY_CONSTRAINT, ret[%u]", ret);
        retryCtx->isNeedReportOpRetryErr = true;
    }

    HCCL_RUN_INFO("[OpRetry][Server]check op ret[%d], nextState[%s]", ret, GetReadableState(nextState));
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerWaitLinkInfo::ProcessEvent(RetryContext* retryCtx)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_SEND_RECV_TIMEOUT);
    // 下一个server状态
    RetryState nextState = RETRY_STATE_CHECK_ALL_LINK;

    std::set<u32> recvVaild;
    while (recvVaild.size() < retryCtx->needRetryServerRanks_.size()) {
        CHK_PRT_RET(retryCtx->isServerStateWaitResume_, HCCL_RUN_INFO("[OpRetry][Server]switched state form wait link to wait resume"), HCCL_SUCCESS);
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout, HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitLinkInfo timeout"), HCCL_E_TIMEOUT);

        for (auto rank : retryCtx->needRetryServerRanks_) {
            if (recvVaild.find(rank) != recvVaild.end()) {
                continue;
            }
            auto &agentRetryInfo = retryCtx->serverSockets_[rank];
            // 接收agent信息
            HcclResult ret = WaitLinkPortCheckResult(agentRetryInfo.socket, agentRetryInfo.linkPortStatus);
            CHK_PRT_RET(ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN,
                HCCL_ERROR("[OpRetry][Server]OpRetryServerWaitLinkCheckResult fail, ret[%u]", ret), ret);
            if (ret == HCCL_SUCCESS) {
                recvVaild.insert(rank);
                HCCL_INFO("[OpRetry][Server]OpRetryServerWaitLinkCheckResult recv success from dst[%u], ", rank);
            }
        }
    }
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    HCCL_INFO("[OpRetry][Server]OpRetryServerWaitLinkInfo success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerCheckAllLink::ProcessEvent(RetryContext* retryCtx)
{
    // 收集所有rank的主备网口信息
    std::map<u32, std::pair<bool, bool>> allLinkInfo;
    for (auto rank: retryCtx->needRetryServerRanks_) {
        auto &linkPortStatus = retryCtx->serverSockets_[rank].linkPortStatus;
        allLinkInfo.insert({rank, std::make_pair(linkPortStatus.defaultPort, linkPortStatus.backupPort)});
    }

    // 对所有rank依次遍历
    for (auto rank: retryCtx->needRetryServerRanks_) {
        u32 remoteRankIndex = 0;
        auto &linkPortStatus = retryCtx->serverSockets_[rank].linkPortStatus;
        // 对rank的所有对端进行遍历
        for (u32 i = 0; i < linkPortStatus.rankSize; i++) {
            u32 remoteRank = linkPortStatus.rankList[i];
            retryCtx->serverSockets_[rank].changeLinkInfo.remoteRankList[remoteRankIndex] = remoteRank;
            if (allLinkInfo[rank].first && allLinkInfo[remoteRank].first) {
                // 本端和对端的主网口均up，则使用主网口
                retryCtx->serverSockets_[rank].changeLinkInfo.isUseDefaultPort[remoteRankIndex] = true;
            } else if (allLinkInfo[rank].second && allLinkInfo[remoteRank].second) {
                // 本端和对端的备网口均up，则使用备网口
                retryCtx->serverSockets_[rank].changeLinkInfo.isUseDefaultPort[remoteRankIndex] = false;
            } else {
                // 本端和对端无可用的网口，重执行失败
                HCCL_ERROR("[OpRetry][Server]rank[%u]:default[%d], backup[%d], IpInfo[%s]; rank[%u]:default[%d], "
                    "backup[%d], can not find same port, can not retry", rank, allLinkInfo[rank].first,
                    allLinkInfo[rank].second, retryCtx->serverSockets_[rank].retryInfo.dfxIpInfo, remoteRank,
                    allLinkInfo[remoteRank].first, allLinkInfo[remoteRank].second);
                CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RETRY_FAIL, retryCtx));
                return HCCL_SUCCESS;
            }
            remoteRankIndex += 1;
        }
        retryCtx->serverSockets_[rank].changeLinkInfo.remoteRankNum = remoteRankIndex;
    }

    // 打印所有rank的借轨信息
    for (auto rank: retryCtx->needRetryServerRanks_) {
        auto &changeLinkInfo = retryCtx->serverSockets_[rank].changeLinkInfo;
        std::string changeLinkInfoStr = "rank[" + std::to_string(rank) + "]";
        for (u32 i = 0; i < changeLinkInfo.remoteRankNum; i++) {
            changeLinkInfoStr += (std::to_string(changeLinkInfo.remoteRankList[i]) + ":" + 
                std::to_string(changeLinkInfo.isUseDefaultPort[i]) + "; ");
        }
        HCCL_INFO("[OpRetry][Server]changeLinkInfoStr:%s", changeLinkInfoStr.c_str());
    }

    // 所有rank网口确认成功，切换到给agent发借轨命令状态
    CHK_RET(CreateOpRetryServerByState(RETRY_STATE_CMD_RESUME_TRANSPORT, retryCtx));
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerIssueChangeLinkAndResume::ProcessEvent(RetryContext* retryCtx)
{
    HcclResult ret = HCCL_SUCCESS;
    RetryState curState = retryCtx->GetRetryState();
    // 先将每个rank的changeLinkInfo发送至对应agent
    for (auto rank : retryCtx->needRetryServerRanks_) {
        ret = IssueChangeLink(retryCtx->serverSockets_[rank].socket, retryCtx->serverSockets_[rank].changeLinkInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerIssueChangeLink fail, curState[%s]",
            GetReadableState(curState)), ret);
        HCCL_INFO("[OpRetry][Server]OpRetryServerIssueChangeLink send to rank[%u] success", rank);
    }
    // 再发送resume transport命令至每个agent
    std::shared_ptr<OpRetryBase> retryPtr = nullptr;
    EXECEPTION_CATCH((retryPtr = std::make_shared<OpRetryServerIssueCmd>()), return HCCL_E_PTR);
    RetryState nextState = RETRY_STATE_CMD_RESUME_TRANSPORT;
    retryCtx->SetRetryState(nextState, retryPtr);
    HCCL_INFO("[OpRetry][Server]OpRetryServerIssueChangeLinkAndResume success");
    return HCCL_SUCCESS;
}

HcclResult OpRetryServerRetryFail::ProcessEvent(RetryContext* retryCtx)
{
    RetryCommandInfo commandInfo;
    commandInfo.command = RETRY_CMD_RETRY_FAIL;
    if (retryCtx->isNeedReportOpRetryErr) {
        commandInfo.command = RETRY_CMD_RETRY_CONSTRAINT_FAIL;
        HCCL_RUN_INFO("[OpRetry][Server]OpRetryServerRetryFail isNeedReportOpRetryErr[%d], command[%s]",retryCtx->isNeedReportOpRetryErr, GetReadableCmd(commandInfo.command));
    }
    HCCL_INFO("[OpRetry][Server]retry fail, command all rank %s", GetReadableCmd(commandInfo.command));
    for (auto rank : retryCtx->needRetryServerRanks_) {
        HcclResult ret = IssueCommandWithOpId(retryCtx->serverSockets_[rank].socket, commandInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[OpRetry][Server]OpRetryServerRetryFail IssueCommandWithOpId AgentId[%u] fail", rank),
            ret);
    }

    // 重执行异常，通知心跳存在异常 -> 广播异常给整个集群
    Heartbeat::GetInstance(retryCtx->deviceLogicId_).SetOpretryErr();

    RetryState nextState = RETRY_STATE_SERVER_RUNNING;
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    return HCCL_SUCCESS;
}

bool SwitchNicServerCheckAllSwitchRanks::CompareSwitchRankList(const u32* firstSwitchRankList,
    const u32* switchRankList, const u32 switchRankNum)
{
    if (switchRankNum == 0 || switchRankNum > AICPU_MAX_RANK_NUM) {
        return false;
    }

    std::set<u32> switchRankSet;
    for (u32 i = 0;  i < switchRankNum; i++) {
        switchRankSet.insert(firstSwitchRankList[i]);
    }

    u32 ranksNum[AICPU_MAX_RANK_NUM] = {0};
    for (u32 i = 0;  i < switchRankNum; i++) {
        if (switchRankSet.find(switchRankList[i]) == switchRankSet.end()) {
            HCCL_ERROR("[SwitchNic][Server] rankList has error, id[%u]", switchRankList[i]);
            return false;
        } else {
            ranksNum[switchRankList[i]]++;
        }
        if (ranksNum[switchRankList[i]] > 1) {
            HCCL_ERROR("[SwitchNic][Server] rankList has error, id[%u], num[%u]",
                switchRankList[i], ranksNum[switchRankList[i]]);
            return false;
        }
    }
    return true;
}

bool SwitchNicServerCheckAllSwitchRanks::CompareUseBackupLists(const bool* firstArray,
    const bool* secondArray, const u32 switchRankNum)
{
    if (switchRankNum == 0 || switchRankNum > AICPU_MAX_RANK_NUM) {
        return false;
    }
    for (u32 i = 0;  i < switchRankNum; i++) {
        if (firstArray[i] != secondArray[i]) {
            HCCL_ERROR("[SwitchNic][Server] backupLists has error first[%u], second[%u], index[%u]",
                firstArray[i], secondArray[i], i);
            return false;
        }
    }
    return true;
}

bool SwitchNicServerCheckAllSwitchRanks::CheckRemotePorts(const u32 rankId, const ActiveSwitchInfo &switchRankInfo)
{
    for (u32 i = 0;  i < switchRankInfo.remoteRankNum; i++) {
        if (switchRankInfo.remoteRankNicStatus[i] == CONNECT_REMOTE_DEFAULT && !switchRankInfo.defaultPortStatus) {
            HCCL_ERROR("[SwitchNic][Server] defaultPortStatus has error, localRank[%u], remoteRank[%u], nicStatus[%u]",
                rankId, i, switchRankInfo.remoteRankNicStatus[i]);
            return false;
        }
        if (switchRankInfo.remoteRankNicStatus[i] == CONNECT_REMOTE_BACKUP && !switchRankInfo.backupPortStatus) {
            HCCL_ERROR("[SwitchNic][Server] backupPortStatus has error, localRank[%u], remoteRank[%u], nicStatus[%u]",
                rankId, i, switchRankInfo.remoteRankNicStatus[i]);
            return false;
        }
    }
    return true;
}

HcclResult SwitchNicServerCheckAllSwitchRanks::CollectSingleAgentActiveSwitchInfo(RetryContext *retryCtx, u32 rankId,
    HcclAgentRetryInfo &agentInfo)
{
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(GetExternalInputHcclLinkTimeOut() * ACTIVE_SWITCH_TIMES);
    HcclResult ret = HCCL_SUCCESS;
    while (true) {
        CHK_PRT_RET(retryCtx->isServerStateWaitResume_, HCCL_RUN_INFO("[OpRetry][Server]switched state form check switch Nic to wait resume"), HCCL_SUCCESS);
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - startTime);
        CHK_PRT_RET(elapsed > timeout,
            HCCL_ERROR("[SwitchNic][Server] timeout in recv agent RetryInfo, waitime[%u>%u]",
            elapsed, timeout), HCCL_E_TIMEOUT);
        ret = WaitResponse(agentInfo.socket, agentInfo.retryInfo);
        if (ret == HCCL_SUCCESS) { // 成功接收到数据
            // 成功接收到retryInfo含RETRY_STATE_SEND_SWITCH_INFO，接收 ActiveSwitchInfo，否则为保活数据，忽略
            if (agentInfo.retryInfo.retryState == RETRY_STATE_SEND_SWITCH_INFO) {
                ret = RecvActiveSwitchInfo(agentInfo.socket, rankId, agentInfo.switchInfo);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                HCCL_INFO("[SwitchNic][server] recv ActiveSwitchInfo form rank[%u] while collecting", rankId);
                retryCtx->switchInfoMap_[rankId] = agentInfo.switchInfo;
                return HCCL_SUCCESS;
            }
        } else if (ret == HCCL_E_AGAIN) {
            RetryCommand command = RETRY_CMD_RUNNING;
            CHK_RET(IssueCommand(agentInfo.socket, command));
            HCCL_DEBUG("[SwitchNic][Server] send keeping active info to dst[%u]", rankId);
        } else {
            HCCL_ERROR("[SwitchNic][Server] get active switch info failed, ret[%u], dst[%u]", ret, rankId);
            break;
        }
        SaluSleep(OP_RETRY_POLL_AICPU_STATE_INTERVAL);
    }
    return ret;
}

HcclResult SwitchNicServerCheckAllSwitchRanks::CollectAgentActiveSwitchInfo(RetryContext *retryCtx)
{
    HCCL_RUN_INFO("[SwitchNic][Server] began to CollectAgentActiveSwitchInfo");
    HcclResult ret = HCCL_SUCCESS;
    // 轮询接收agent信息
    for (auto &it : retryCtx->serverSockets_) {
        const u32 &rank = it.first;
        if (retryCtx->switchInfoMap_.find(rank) != retryCtx->switchInfoMap_.end()) {
            HCCL_DEBUG("[SwitchNic][Server] rank[%u] has been received", rank);
            continue;
        }
        // 轮询接收agent状态机信息
        ret = CollectSingleAgentActiveSwitchInfo(retryCtx, it.first, it.second);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
    }
    // 可不检查，理论上不会不等于
    if (retryCtx->switchInfoMap_.size() != retryCtx->serverSockets_.size()) {
        return HCCL_E_UNAVAIL;
    }
    return ret;
}

HcclResult SwitchNicServerCheckAllSwitchRanks::CheckAgentActiveSwitchInfo(RetryContext *retryCtx)
{
    HCCL_RUN_INFO("[SwitchNic][Server] began to CheckAgentActiveSwitchInfo");
    if (retryCtx->switchInfoMap_.empty()) {
        HCCL_ERROR("[SwitchNic][Server] switchInfoMap is empty");
        return HCCL_E_PARA;
    }

    RetryCommand command = RETRY_CMD_NOTIFY_SWITCH_SUC;

    auto firstInfo = retryCtx->switchInfoMap_.begin();
    auto firstRankId = firstInfo->first;
    ActiveSwitchInfo &firstSwitchInfo = firstInfo->second;

    for (const auto &it : retryCtx->switchInfoMap_) {
        const u32 &rankId = it.first;
        const ActiveSwitchInfo &switchInfo = it.second;
        if (!switchInfo.refreshTransportFin) {
            HCCL_ERROR("[SwitchNic][Server] refreshTransportFin is false, first[%u], rank[%u]", firstRankId, rankId);
            command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
            break;
        }
        if (switchInfo.switchRankNum != firstSwitchInfo.switchRankNum) {
            HCCL_ERROR("[SwitchNic][Server] switchRankNum is not same as the first[%u:%u], rank[%u:%u]",
                firstRankId, firstSwitchInfo.switchRankNum, rankId, switchInfo.switchRankNum);
            command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
            break;
        } else {
            if (!CompareSwitchRankList(firstSwitchInfo.switchRankList,
                switchInfo.switchRankList, switchInfo.switchRankNum)) {
                HCCL_ERROR("[SwitchNic][Server] SwitchRankList is not same as the first[%u], rank[%u], rankNum[%u]",
                    firstRankId, rankId, switchInfo.switchRankNum);
                command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
                break;
            }
            if (!CompareUseBackupLists(firstSwitchInfo.switchUseBackup,
                switchInfo.switchUseBackup, switchInfo.switchRankNum)) {
                HCCL_ERROR("[SwitchNic][Server] UseBackupLists is not same as the first[%u], rank[%u], rankNum[%u]",
                    firstRankId, rankId, switchInfo.switchRankNum);
                command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
                break;
            }
        }

        if (!switchInfo.localPortsCheckRet) {
            HCCL_ERROR("[SwitchNic][Server] localPortsCheckRet is false, first[%u], rank[%u]", firstRankId, rankId);
            command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
            break;
        }
        if (!CheckRemotePorts(rankId, switchInfo)) {
            command = RETRY_CMD_NOTIFY_SWITCH_FAIL;
            break;
        }
    }

    // 全部卡确认无误后或者发现错误后，通知OpRetryAgent。
    for (auto it : retryCtx->serverSockets_) {
        HcclResult ret = IssueCommand(it.second.socket, command);
        CHK_PRT_RET(ret != HCCL_SUCCESS,
            HCCL_ERROR("[SwitchNic][Server] CheckAllSwitchRanks IssueCommand AgentId[%u] fail", it.first),
            ret);
    }
    retryCtx->switchInfoMap_.clear();
    return HCCL_SUCCESS;
}

// OpRetryServer遍历通信域内的所有卡，如何接收？接收完成借轨的flag，portStatusMap
// 并且校验每张卡网卡状态、switchRanks信息一致（新增struct用于send recv信息）。
// 全部卡确认无误后或者发现错误后，通知OpRetryAgent。

HcclResult SwitchNicServerCheckAllSwitchRanks::ProcessEvent(RetryContext *retryCtx)
{
    HCCL_RUN_INFO("[SwitchNic][Server] CheckAllSwitchRanks begin");
    RetryState nextState = RETRY_STATE_SERVER_RUNNING;

    CHK_RET(CollectAgentActiveSwitchInfo(retryCtx));
    CHK_RET(CheckAgentActiveSwitchInfo(retryCtx));
    CHK_RET(CreateOpRetryServerByState(nextState, retryCtx));
    HCCL_RUN_INFO("[SwitchNic][Server] CheckAllSwitchRanks end");
    return HCCL_SUCCESS;
}


HcclResult OpRetryServerWaitResume::ProcessEvent(RetryContext *retryCtx)
{
    if (!retryCtx->isServerStateWaitResume_) {
        retryCtx->ResetServerState();
        CHK_RET(CreateOpRetryServerByState(RETRY_STATE_SERVER_RUNNING, retryCtx));
        HCCL_RUN_INFO("[OpRetry][Server]OpRetryServerWaitResume, set state to running");
    }

    const std::chrono::seconds timeout = std::chrono::seconds(OP_RETRY_KEEP_INTERVAL);
    // 轮询接收agent信息
    for (auto &it : retryCtx->serverSockets_) {
        const u32 &agentId = it.first;
        // 若对端已经关闭, 则不再轮询
        if (disableAgent_.find(agentId) != disableAgent_.end()) {
            continue;
        }
        // 记录时间, 检测和对端上一次通信时间是否超过保活时间
        std::chrono::steady_clock::time_point curTime = std::chrono::steady_clock::now();
        if (lastRecvTimes_.find(agentId) == lastRecvTimes_.end()) {
            lastRecvTimes_.insert(std::make_pair(agentId, curTime));
        }

        // 轮询接收agent状态机信息
        HcclResult ret = WaitResponse(it.second.socket, it.second.retryInfo);
        if (ret == HCCL_SUCCESS) {  // 成功接收到数据
            lastRecvTimes_[agentId] = curTime;
        } else if (ret == HCCL_E_AGAIN) {  // 未接收到数据
            // 校验是否超时
            const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(curTime - lastRecvTimes_[agentId]);
            if (elapsed > timeout) {
                RetryCommandInfo commandInfo;
                commandInfo.command = RETRY_CMD_RUNNING;
                CHK_RET(IssueCommandWithOpId(it.second.socket, commandInfo));
                HCCL_WARNING("[OpRetry][Server]OpRetryServerRunning recv Retry Frame from agentId[%u] timeout", agentId);
                lastRecvTimes_[agentId] = curTime;
            }
        } else {  // 接收数据失败
            disableAgent_.insert(agentId);
            HCCL_RUN_INFO("[OpRetry][Server]WaitResponse from agentId[%u] fail, ret[%u]", agentId, ret);
        }
    }
    return HCCL_SUCCESS;
}
}