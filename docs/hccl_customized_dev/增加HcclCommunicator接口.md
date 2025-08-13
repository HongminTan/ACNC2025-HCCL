# 增加HcclCommunicator接口<a name="ZH-CN_TOPIC_0000001941345833"></a>

HcclCommunicator是通信域功能的执行层，在HCCL架构中隶属于通信框架层。HcclCommunicator与算子类通过三个接口进行交互（算法选择接口、资源计算接口与编排接口，详细可参考[增加通信算子Operator](增加通信算子Operator.md)），并进行资源创建（stream、notify、memory、建链等）。

涉及代码文件：
-   src/domain/collective\_communication/framework/communicator/impl/hccl\_communicator.cc
-   src/domain/collective\_communication/framework/communicator/impl/hccl\_communicator.h
-   src/domain/collective\_communication/algorithm/impl/alg\_configurator.h
-   CANN软件安装目录/include/experiment/hccl/hccl\_common.h

1.  添加新的算子枚举值。
    1.  在“hccl\_common.h”文件中的HcclCMDType枚举类中为新算子添加一个枚举值。

        枚举值的格式为“HCCL\_CMD\_XXX”，每个算子都唯一对应HcclCMDType中的一个值。

        其中HCCL\_CMD\_INVALID，HCCL\_CMD\_MAX 和 HCCL\_CMD\_ALL为特殊值，具有特定作用。

        -   HCCL\_CMD\_INVALID 表示无效算子，必须放在第一个，且值等于0。
        -   HCCL\_CMD\_MAX 记录了 HcclCMDType 中枚举值的数量，必须放在最后。
        -   HCCL\_CMD\_ALL 在某些场景下表示所有算子，建议放在 HCCL\_CMD\_MAX 的前一个位置。

    2.  在 “alg\_configurator.h” 中的以下map成员的默认值中添加新枚举值。

        ```
        algType_
        isAlgoLevel1Default_
        ```

2. 定义新算子的API。

    在 hccl\_communicator.h 中声明新算子的接口。

    ```
    HcclResult MyOperatorOutPlace(args...)
    ```

    其中OutPlace后缀代表单算子模式。

    在 hccl\_communicator.cc 中添加新算子的定义。

    以 ReduceScatter 算子为例：

    ```
    HcclResult ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr, u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    ```

3.  异常流程处理（可选）

    处理异常流程可以有效避免预期之外的行为，减少错误或提升效率。

    例如检查当前device类型是否支持该算子，检查通信域是否已经初始化等。

    **说明：**源码中的硬件类型体现的是Soc Version，您可以在安装昇腾AI处理器的服务器中执行“**npu-smi info**”命令查询，查询到的“Chip Name”即为对应的Soc Version。

    以 ReduceScatter 算子为例：

    ```
    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        ...
    
        // 硬件类型为Atlas 推理系列产品（Ascend 310P处理器）中的加速卡时，不支持ReduceScatter算子
        CHK_PRT_RET(Is310P3Common(), HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]"
            "ReduceScatterOutPlace is not supported"), HCCL_E_NOT_SUPPORT);
    
        // 通信域未初始化，返回报错
        if (!IsAtomicInit()) {
            HCCL_ERROR("[HcclCommunicator][ReduceScatterOutPlace]errNo[0x%016llx] hccl init must be called before"
                " call this function", HCCL_ERROR_CODE(HCCL_E_UNAVAIL));
            return HCCL_E_UNAVAIL;
        }
    
        ...
    }
    ```

4.  添加Debug信息（按需）。

    HCCL提供了若干维测功能，可记录算子运行时的一些信息，用于分析算子行为，有助于问题定位。

    例如算子统计功能：在算子执行前后分别调用 StarsCounter 接口，进行头计数和尾计数

    ```
    HcclResult StarsCounter(const HcclDispatcher &dispatcher, Stream &stream, int flag)
    ```

    <a name="table10972811151010"></a>
    <table><thead align="left"><tr id="row597216118108"><th class="cellrowborder" valign="top" width="16.31%" id="mcps1.1.5.1.1"><p id="p13972141181016"><a name="p13972141181016"></a><a name="p13972141181016"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="28.82%" id="mcps1.1.5.1.2"><p id="p79724118103"><a name="p79724118103"></a><a name="p79724118103"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="13.889999999999999%" id="mcps1.1.5.1.3"><p id="p697281113106"><a name="p697281113106"></a><a name="p697281113106"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="40.98%" id="mcps1.1.5.1.4"><p id="p1097216118107"><a name="p1097216118107"></a><a name="p1097216118107"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row16972201131014"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p49722011111016"><a name="p49722011111016"></a><a name="p49722011111016"></a>dispatcher</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p497281114108"><a name="p497281114108"></a><a name="p497281114108"></a>const HcclDispatcher &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p20972111111109"><a name="p20972111111109"></a><a name="p20972111111109"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p119727115101"><a name="p119727115101"></a><a name="p119727115101"></a>调度器，一般传入成员dispatcher_即可</p>
    </td>
    </tr>
    <tr id="row1497241191012"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p1097231113109"><a name="p1097231113109"></a><a name="p1097231113109"></a>stream</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p7943917191114"><a name="p7943917191114"></a><a name="p7943917191114"></a>Stream &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p139722011141017"><a name="p139722011141017"></a><a name="p139722011141017"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p09731011141014"><a name="p09731011141014"></a><a name="p09731011141014"></a>算子的主流</p>
    </td>
    </tr>
    <tr id="row0866183715103"><td class="cellrowborder" valign="top" width="16.31%" headers="mcps1.1.5.1.1 "><p id="p386643721018"><a name="p386643721018"></a><a name="p386643721018"></a>flag</p>
    </td>
    <td class="cellrowborder" valign="top" width="28.82%" headers="mcps1.1.5.1.2 "><p id="p168668372108"><a name="p168668372108"></a><a name="p168668372108"></a>int</p>
    </td>
    <td class="cellrowborder" valign="top" width="13.889999999999999%" headers="mcps1.1.5.1.3 "><p id="p3866173761013"><a name="p3866173761013"></a><a name="p3866173761013"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="40.98%" headers="mcps1.1.5.1.4 "><p id="p1086614376105"><a name="p1086614376105"></a><a name="p1086614376105"></a>0代表头，1代表尾</p>
    </td>
    </tr>
    </tbody>
    </table>

    其中，HcclDispacher 为调度器类，用于封装内存拷贝操作；Stream 为流类。

    返回值：HCCL执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    以 ReduceScatter 算子为例：

    ```
    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        ...
        // 头计数任务
        CHK_RET(StarsCounter(dispatcher_, streamObj, HEAD));
        // 调用算子执行接口
        ...
        // 尾计数任务
        CHK_RET(StarsCounter(dispatcher_, streamObj, TAIL));
        return HCCL_SUCCESS;
    }
    ```

5. 调用算子执行接口。

    通过调用ExecOp接口执行算子流程，包含通过opType获取算子实例，算法选择，根据资源计算结果进行资源创建，执行算法编排。

    ```
    HcclResult ExecOp(HcclCMDType opType, const OpParam &opParam)
    ```

    参数含义如下表所示。

    **表 1**  ExecOp接口参数说明

    <a name="table827101275518"></a>
    <table><thead align="left"><tr id="row429121265517"><th class="cellrowborder" valign="top" width="13.68%" id="mcps1.2.5.1.1"><p id="p1329121214558"><a name="p1329121214558"></a><a name="p1329121214558"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="23.080000000000002%" id="mcps1.2.5.1.2"><p id="p146768713238"><a name="p146768713238"></a><a name="p146768713238"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="12.5%" id="mcps1.2.5.1.3"><p id="p10230141454318"><a name="p10230141454318"></a><a name="p10230141454318"></a>输入/输出</p>
    </th>
    <th class="cellrowborder" valign="top" width="50.739999999999995%" id="mcps1.2.5.1.4"><p id="p83121275519"><a name="p83121275519"></a><a name="p83121275519"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row1131131265511"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.5.1.1 "><p id="p191061137121320"><a name="p191061137121320"></a><a name="p191061137121320"></a>opType</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.080000000000002%" headers="mcps1.2.5.1.2 "><p id="p46778720237"><a name="p46778720237"></a><a name="p46778720237"></a>HcclCMDType</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.5%" headers="mcps1.2.5.1.3 "><p id="p16105133721316"><a name="p16105133721316"></a><a name="p16105133721316"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p10105143741311"><a name="p10105143741311"></a><a name="p10105143741311"></a>算子类型</p>
    </td>
    </tr>
    <tr id="row18118485118"><td class="cellrowborder" valign="top" width="13.68%" headers="mcps1.2.5.1.1 "><p id="p11104837101311"><a name="p11104837101311"></a><a name="p11104837101311"></a>opParam</p>
    </td>
    <td class="cellrowborder" valign="top" width="23.080000000000002%" headers="mcps1.2.5.1.2 "><p id="p20677197162311"><a name="p20677197162311"></a><a name="p20677197162311"></a>const OpParam &amp;</p>
    </td>
    <td class="cellrowborder" valign="top" width="12.5%" headers="mcps1.2.5.1.3 "><p id="p8103173701314"><a name="p8103173701314"></a><a name="p8103173701314"></a>输入</p>
    </td>
    <td class="cellrowborder" valign="top" width="50.739999999999995%" headers="mcps1.2.5.1.4 "><p id="p151038375137"><a name="p151038375137"></a><a name="p151038375137"></a>算子的入参，包括输入输出指针、数据量等信息。</p>
        <p id="p1764074615421"><a name="p1764074615421"></a><a name="p1764074615421"></a>OpParam数据结构的介绍可参见<a href="OpParam.md">OpParam</a>，构造OpParam时只需为当前算子实际用到的成员赋值即可。</p>
    </td>
    </tr>
    </tbody>
    </table>


    返回值：HCCL执行结果，成功时返回HCCL\_SUCCESS，异常时返回相应的错误类型。

    **注意：** 

    -   若自定义算子使用了OpParam未包含的入参，需在OpParam的定义中对应增加新的成员。
    -   调用ExecOp时，opType需要传入步骤[1](#li1544184665913)新增的枚举值，opParam需要用算子入参构造。

    以 ReduceScatter 算子为例：

    ```
    HcclResult HcclCommunicator::ReduceScatterOutPlace(const std::string &tag, void *inputPtr, void *outputPtr,
        u64 count, HcclDataType dataType, HcclReduceOp op, HcclRtStream stream)
    {
        ...
    
        u32 perDataSize = SIZE_TABLE[dataType];
        // 用算子入参构造OpParam
        OpParam opParam;
        opParam.tag = tag;
        opParam.inputPtr = inputPtr;
        opParam.inputSize = userRankSize_ * count * perDataSize;
        opParam.outputPtr = outputPtr;
        opParam.outputSize = count * perDataSize;
        opParam.DataDes.count = count;
        opParam.DataDes.dataType = dataType;
        opParam.reduceType = op;
        opParam.stream = streamObj;
        // 调用算子执行接口
        CHK_RET(ExecOp(HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam));
    
        ...
    }
    ```

