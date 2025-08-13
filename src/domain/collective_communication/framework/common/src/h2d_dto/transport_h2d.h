/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef H2D_DTO_TRANSPORT_H2D_H
#define H2D_DTO_TRANSPORT_H2D_H

// Transport 内存类型
enum class HcclAiRMAMemType : u32 {
    LOCAL_INPUT = 0,
    REMOTE_INPUT,

    LOCAL_OUTPUT,
    REMOTE_OUTPUT,

    // 可透传更多的内存，可在MAX_NUM之前追加，例如：
    // LOCAL_EXP,
    // REMOTE_EXP,
    MAX_NUM
};

constexpr u32 GetAiMemTypeVal(HcclAiRMAMemType value) {
    return static_cast<u32>(value);
}

constexpr u32 AiMemMaxNum = GetAiMemTypeVal(HcclAiRMAMemType::MAX_NUM);

// Transport 内存信息
struct HcclAiRMAMemInfo {
    u32 memMaxNum;  // 最大内存数量，等于 HcclAiRMAMemType::MAX_NUM
    u32 sizeOfMemDetails;  // sizeof(MemDetails)，用于内存校验和偏移计算
    u64 memDetailPtr;  // MemDetails数组首地址, 个数: HcclAiRMAMemType::MAX_NUM
    // 可追加字段

    HcclAiRMAMemInfo() :
        memMaxNum(0), sizeOfMemDetails(0), memDetailPtr(0)
    {}
};

// 全部 Transport QP/Mem 信息
struct HcclAiRMAInfo {
    u32 curRankId;  // 当前rankId
    u32 rankNum;  // rank数量
    u32 qpNum;  // 单个Transport的QP数量

    u32 sizeOfAiRMAWQ;  // sizeof(HcclAiRMAWQ)
    u32 sizeOfAiRMACQ;  // sizeof(HcclAiRMACQ)
    u32 sizeOfAiRMAMem;  // sizeof(HcclAiRMAMemInfo)

    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SQ指针：sqPtr + dstRankId * qpNum * sizeOfAiRMAWQ + qpIndex
    void* sqPtr;
    
    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取SCQ指针：scqPtr + dstRankId * qpNum * sizeOfAiRMACQ + qpIndex
    void* scqPtr;
    
    // HcclAiRMAWQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RQ指针：rqPtr + dstRankId * qpNum * sizeOfAiRMAWQ + qpIndex
    void* rqPtr;

    // HcclAiRMACQ二维数组首地址
    // QP个数: rankNum * qpNum
    // 计算偏移获取RCQ指针: rcqPtr + dstRankId * qpNum * sizeOfAiRMACQ + qpIndex
    void* rcqPtr;

    // HcclAivMemInfo一维数组
    // 内存信息个数: rankNum
    // 计算偏移获取内存信息指针: memPtr + rankId * sizeOfAiRMAMem
    // srcRankId 获取自身内存信息，dstRankId 获取 Transport 内存信息
    void* memPtr;

    // 可往后追加字段

    HcclAiRMAInfo() :
        curRankId(0), rankNum(0), qpNum(0),
        sizeOfAiRMAWQ(0), sizeOfAiRMACQ(0), sizeOfAiRMAMem(0),
        sqPtr(nullptr), scqPtr(nullptr), rqPtr(nullptr), rcqPtr(nullptr), memPtr(nullptr)
    {}
};

#endif