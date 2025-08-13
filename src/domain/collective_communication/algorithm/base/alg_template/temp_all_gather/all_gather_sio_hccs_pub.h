/*
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
 
#ifndef ALL_GATHER_SIO_HCCS_PUB_H
#define ALL_GATHER_SIO_HCCS_PUB_H
 
#include "alg_template_base_pub.h"
 
namespace hccl {
class AllGatherSioHccs : public AlgTemplateBase {
public:
    explicit AllGatherSioHccs(const HcclDispatcher dispatcher);
    ~AllGatherSioHccs() override;

    HcclResult Prepare(SubCommInfo &outerCommInfoHccs, SubCommInfo &outerCommInfoSio, DeviceMem &usrInMem,
        DeviceMem &usrOutMem, u32 totalCount, const HcclDataType dataType, const Stream &mainStream,
        std::vector<Stream> &meshStreams, std::vector<std::shared_ptr<LocalNotify>> &meshSignal,
        std::vector<std::shared_ptr<LocalNotify>> &meshSignalAux, u32 userRank) override;
    HcclResult RunAsync(const u32 rank, const u32 rankSize, const std::vector<LINK> &links) override;

protected:
private:
    HcclResult NotifySubStreamStart();
    HcclResult WaitSubStreamFinish();
    HcclResult RunInterDie(const u32 dieRankId, const std::vector<LINK> &links, const u32 srcDMAMemSliceId);
 
    DeviceMem inputMem_;
    DeviceMem outputMem_;
    SubCommInfo outerCommInfoHccs_;
    SubCommInfo outerCommInfoSio_;
    std::vector<DeviceMem> dmaMem_;
    Stream stream_;
    std::vector<Stream> meshStreams_; /** 多steam**/
    std::vector<std::shared_ptr<LocalNotify>> meshSignal_;  /* 每个ring创建一个signal */
    std::vector<std::shared_ptr<LocalNotify>> meshSignalAux_; /* 从stream wait，主steam record */
    u32 intraRankSize_;
    u32 userRank_;
    HcclDataType dataType_;
    u64 dataBytes_;
    u32 notifyIdx_ = 0; // 新增notify资源索引
};
} // namespace hccl
 
#endif /* ALL_GATHER_SIO_HCCS_PUB_H */