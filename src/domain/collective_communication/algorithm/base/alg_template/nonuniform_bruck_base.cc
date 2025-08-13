/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "nonuniform_bruck_base.h"

namespace hccl {


NBBase::NBBase(const HcclDispatcher dispatcher)
    : AlgTemplateBase(dispatcher)
{
}

NBBase::~NBBase()
{
}

u32 NBBase::CalcCeilLog2(const u32 num)
{
    u32 ans = 0;
    for (u32 tmp = num - 1; tmp != 0; tmp >>= 1, ++ans) {}
    return ans;
}


}   // ~~ namespace hccl
