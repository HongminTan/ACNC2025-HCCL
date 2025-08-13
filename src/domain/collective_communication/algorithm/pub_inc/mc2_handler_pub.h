/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_HANDLER_PUB_H
#define MC2_HANDLER_PUB_H

#include "alg_template_base_pub.h"
#include "log.h"

namespace hccl {

constexpr u32 MC2_MAX_TURN = 32;
constexpr u32 MC2_MAX_RANK_NUM = 256;

class Mc2HandlerPub {
public:
    Mc2HandlerPub();

    HcclResult Mc2WaitValue(HcclDispatcher dispatcherPtr, hccl::Stream &stream, Mc2Handler *mc2Handler, u32 step);
    HcclResult Mc2WriteValue(HcclDispatcher dispatcherPtr, hccl::Stream &stream, Mc2Handler *mc2Handler);
private:
    u32 mc2TurnNum_[MC2_MAX_TURN * MC2_MAX_RANK_NUM];
    u32 turnNumForWrite_ = 0;
};

}

#endif