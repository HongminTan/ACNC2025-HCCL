/*
 * Copyright (c) 2024 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include <functional>
#include <memory>

#include "alg_template_base_pub.h"

namespace hccl {

using AlgTemplateCreator = std::function<AlgTemplateBase *(const HcclDispatcher)>;

template <typename P>
static AlgTemplateBase *DefaultTemplateCreator(const HcclDispatcher dispatcher)
{
    static_assert(std::is_base_of<AlgTemplateBase, P>::value,
        "Template type must derived from Hccl::AlgTemplateBase");
    return new (std::nothrow) P(dispatcher);
}

class AlgTemplateRegistry {
public:
    static AlgTemplateRegistry &Instance();
    AlgTemplateRegistry();
    HcclResult Register(const TemplateType type, const AlgTemplateCreator &algTemplateCreator);
    std::unique_ptr<AlgTemplateBase> GetAlgTemplate(const TemplateType type,
        const HcclDispatcher dispatcher);

private:
    std::vector<AlgTemplateCreator> tempCreators_;
    mutable std::mutex mu_;
};

#define REGISTER_TEMPLATE_HELPER(ctr, type, algTempBase)       \
    static HcclResult g_func_##algTempBase##_##ctr             \
        = AlgTemplateRegistry::Instance().Register(type, DefaultTemplateCreator<algTempBase>)

#define REGISTER_TEMPLATE(type, algTempBase) REGISTER_TEMPLATE_HELPER(__COUNTER__, type, algTempBase)

}   // namespace hccl
