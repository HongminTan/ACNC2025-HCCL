# OpParam<a name="ZH-CN_TOPIC_0000001995601776"></a>

## 功能说明

OpParam结构体用于承载算子所有可能用到的入参信息。

## 原型定义

```
struct OpParam {
    std::string tag = "";
    Stream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    bool aicpuUnfoldMode = false;
    HcclOpBaseAtraceInfo* opBaseAtraceInfo = nullptr;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
        } DataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            u64 sendCount;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
            void* sendCountMatrix;
        } All2AllDataDes;
        struct { 
            HcclSendRecvItem* sendRecvItemsPtr;
            u32 itemNum;
        } BatchSendRecvDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
};
```

## 成员介绍

**表 1**  OpParam成员说明

<table><thead align="left"><tr id="row877218322430"><th class="cellrowborder" colspan="2" valign="top" id="mcps1.2.4.1.1"><p id="p147722032184320"><a name="p147722032184320"></a><a name="p147722032184320"></a>成员</p>
</th>
<th class="cellrowborder" valign="top" id="mcps1.2.4.1.2"><p id="p377233214436"><a name="p377233214436"></a><a name="p377233214436"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row137726320436"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p116754327116"><a name="p116754327116"></a><a name="p116754327116"></a>tag</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1295918143115"><a name="p1295918143115"></a><a name="p1295918143115"></a>算子在通信域中的标记，用于维测功能。</p>
</td>
</tr>
<tr id="row13772163215439"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p1227315321312"><a name="p1227315321312"></a><a name="p1227315321312"></a>stream</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p1277293220433"><a name="p1277293220433"></a><a name="p1277293220433"></a>算子执行的主流。</p>
</td>
</tr>
<tr id="row77721232134315"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p8613162913142"><a name="p8613162913142"></a><a name="p8613162913142"></a>inputPtr</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p5772203234319"><a name="p5772203234319"></a><a name="p5772203234319"></a>输入内存的指针，默认为nullptr。</p>
</td>
</tr>
<tr id="row27721732104316"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p6990165012619"><a name="p6990165012619"></a><a name="p6990165012619"></a>inputSize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p4408162132312"><a name="p4408162132312"></a><a name="p4408162132312"></a>输入内存大小。</p>
</td>
</tr>
<tr id="row7532371880"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p16667501186"><a name="p16667501186"></a><a name="p16667501186"></a>outputPtr</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p15531937283"><a name="p15531937283"></a><a name="p15531937283"></a>输出内存的指针，默认为nullptr。</p>
</td>
</tr>
<tr id="row85318373819"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p866618501687"><a name="p866618501687"></a><a name="p866618501687"></a>outputSize</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p175418378818"><a name="p175418378818"></a><a name="p175418378818"></a>输出内存大小。</p>
</td>
</tr>
<tr id="row1540379810"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p16534113913104"><a name="p16534113913104"></a><a name="p16534113913104"></a>reduceType</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p157502576384"><a name="p157502576384"></a><a name="p157502576384"></a>消减运算类型，枚举值。</p>
<p id="p3546374814"><a name="p3546374814"></a><a name="p3546374814"></a>定义如下，默认值为HCCL_REDUCE_RESERVED。</p>
<pre class="screen" id="screen1652043103512"><a name="screen1652043103512"></a><a name="screen1652043103512"></a>typedef enum {
    HCCL_REDUCE_SUM = 0,    /* sum */
    HCCL_REDUCE_PROD = 1,   /* prod */
    HCCL_REDUCE_MAX = 2,    /* max */
    HCCL_REDUCE_MIN = 3,    /* min */
    HCCL_REDUCE_RESERVED    /* reserved */
} HcclReduceOp;</pre>
</td>
</tr>
<tr id="row155414373819"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p1534133981020"><a name="p1534133981020"></a><a name="p1534133981020"></a>syncMode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p19542378818"><a name="p19542378818"></a><a name="p19542378818"></a>notifywait超时类型，默认为DEFAULT_TIMEWAITSYNCMODE。</p>
<p id="p386622911326"><a name="p386622911326"></a><a name="p386622911326"></a>SyncMode类型定义如下：</p>
<pre class="screen" id="screen7769240175311"><a name="screen7769240175311"></a><a name="screen7769240175311"></a>enum class SyncMode {
    DEFAULT_TIMEWAITSYNCMODE = 0,  <span id="ph630403963117"><a name="ph630403963117"></a><a name="ph630403963117"></a>     // 默认模式</span>
    CONFIGURABLE_TIMEWAITSYNCMODE = 1,  <span id="ph143041228325"><a name="ph143041228325"></a><a name="ph143041228325"></a>// 从环境变量配置</span>
    UNLIMITED_TIMEWAITSYNCMODE<span id="ph176385541325"><a name="ph176385541325"></a><a name="ph176385541325"></a>          // 无限等待模式</span>
};</pre>
</td>
</tr>
<tr id="row951423531018"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p1553473919104"><a name="p1553473919104"></a><a name="p1553473919104"></a>root</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p9253710154118"><a name="p9253710154118"></a><a name="p9253710154118"></a>root节点的rank id，默认值为INVALID_VALUE_RANKID，用于Reduce、Scatter和BroadCast算子。</p>
</td>
</tr>
<tr id="row351413541016"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p85341739201017"><a name="p85341739201017"></a><a name="p85341739201017"></a>dstRank</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p185141935171010"><a name="p185141935171010"></a><a name="p185141935171010"></a>目的rank id，用于Send/Recv算子。</p>
</td>
</tr>
<tr id="row175143354101"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p13534133913104"><a name="p13534133913104"></a><a name="p13534133913104"></a>srcRank</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p7515133571017"><a name="p7515133571017"></a><a name="p7515133571017"></a>源rank id，用于Send/Recv算子。</p>
</td>
</tr>
<tr id="row1661183763512"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p96114378350"><a name="p96114378350"></a><a name="p96114378350"></a>aicpuUnfoldMode</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p061137133516"><a name="p061137133516"></a><a name="p061137133516"></a><span id="ph1312310149334"><a name="ph1312310149334"></a><a name="ph1312310149334"></a>是否为aicpu</span><span id="ph596302453417"><a name="ph596302453417"></a><a name="ph596302453417"></a>展开模式。</span></p>
</td>
</tr>
<tr id="row1651563551013"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p9255952201011"><a name="p9255952201011"></a><a name="p9255952201011"></a>opBaseAtraceInfo</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p18731922173311"><a name="p18731922173311"></a><a name="p18731922173311"></a>Atrace管理类对象的指针，用于保存trace日志。</p>
</td>
</tr>
<tr id="row666513424417"><td class="cellrowborder" rowspan="14" valign="top" headers="mcps1.2.4.1.1 "><p id="p2312114418413"><a name="p2312114418413"></a><a name="p2312114418413"></a>union</p>
<p id="p127885534719"><a name="p127885534719"></a><a name="p127885534719"></a><strong id="b6788556476"><a name="b6788556476"></a><a name="b6788556476"></a>说明：</strong></p>
<p id="p18521110144713"><a name="p18521110144713"></a><a name="p18521110144713"></a>对于一个算子，DataDes、All2AllDataDes、BatchSendRecvDataDes只会生效其中的一个。</p>
</td>
<td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 mcps1.2.4.1.2 "><p id="p1522019072812"><a name="p1522019072812"></a><a name="p1522019072812"></a><strong id="b18783105111454"><a name="b18783105111454"></a><a name="b18783105111454"></a>DataDes</strong>（通用定义）</p>
</td>
</tr>
<tr id="row115152351106"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p21711227161513"><a name="p21711227161513"></a><a name="p21711227161513"></a>count</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p7515335171015"><a name="p7515335171015"></a><a name="p7515335171015"></a>输入数据个数</p>
</td>
</tr>
<tr id="row3651174412147"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1865264481417"><a name="p1865264481417"></a><a name="p1865264481417"></a>dataType</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2652344131417"><a name="p2652344131417"></a><a name="p2652344131417"></a>输入数据类型，如int8, in16, in32, float16, fload32等</p>
</td>
</tr>
<tr id="row99141535134518"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 mcps1.2.4.1.2 "><p id="p1722010132810"><a name="p1722010132810"></a><a name="p1722010132810"></a><strong id="b1040224934518"><a name="b1040224934518"></a><a name="b1040224934518"></a>All2AllDataDes</strong>（ AlltoAll操作专用）</p>
</td>
</tr>
<tr id="row16515123521017"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p121151620151617"><a name="p121151620151617"></a><a name="p121151620151617"></a>sendType</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p196324110192"><a name="p196324110192"></a><a name="p196324110192"></a>发送数据类型</p>
</td>
</tr>
<tr id="row13737122515192"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p773792521914"><a name="p773792521914"></a><a name="p773792521914"></a>recvType</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p87378258192"><a name="p87378258192"></a><a name="p87378258192"></a>接收数据类型</p>
</td>
</tr>
<tr id="row520163521919"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p142103591917"><a name="p142103591917"></a><a name="p142103591917"></a>sendCounts</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p82173541915"><a name="p82173541915"></a><a name="p82173541915"></a>发送数据个数</p>
</td>
</tr>
<tr id="row142163531919"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1214355198"><a name="p1214355198"></a><a name="p1214355198"></a>recvCounts</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p112113353195"><a name="p112113353195"></a><a name="p112113353195"></a>接收数据个数</p>
</td>
</tr>
<tr id="row17830174161917"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p48316416199"><a name="p48316416199"></a><a name="p48316416199"></a>sdispls</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1683104116194"><a name="p1683104116194"></a><a name="p1683104116194"></a>表示发送偏移量的uint64数组</p>
</td>
</tr>
<tr id="row1983194113192"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p78313411196"><a name="p78313411196"></a><a name="p78313411196"></a>rdispls</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p48310411196"><a name="p48310411196"></a><a name="p48310411196"></a>表示接收偏移量的uint64数组</p>
</td>
</tr>
<tr id="row1783114111193"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p17831124120194"><a name="p17831124120194"></a><a name="p17831124120194"></a>sendCountMatrix</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p183114415194"><a name="p183114415194"></a><a name="p183114415194"></a>代表每张卡要发给别人的count的信息</p>
</td>
</tr>
<tr id="row668785519452"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 mcps1.2.4.1.2 "><p id="p76879551452"><a name="p76879551452"></a><a name="p76879551452"></a><strong id="b1458517918462"><a name="b1458517918462"></a><a name="b1458517918462"></a>BatchSendRecvDataDes</strong>（BatchSendRecv操作专用）</p>
</td>
</tr>
<tr id="row165151235131018"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p2220130122818"><a name="p2220130122818"></a><a name="p2220130122818"></a>orderedList</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1747095991811"><a name="p1747095991811"></a><a name="p1747095991811"></a>发送和接收的item列表</p>
</td>
</tr>
<tr id="row28241920181813"><td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p1382592017184"><a name="p1382592017184"></a><a name="p1382592017184"></a>itemNum</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.1 "><p id="p17825182012189"><a name="p17825182012189"></a><a name="p17825182012189"></a>item数量</p>
</td>
</tr>
<tr id="row19515163591012"><td class="cellrowborder" colspan="2" valign="top" headers="mcps1.2.4.1.1 "><p id="p185151735111015"><a name="p185151735111015"></a><a name="p185151735111015"></a>opType</p>
</td>
<td class="cellrowborder" valign="top" headers="mcps1.2.4.1.2 "><p id="p12515193541010"><a name="p12515193541010"></a><a name="p12515193541010"></a>算子类型</p>
</td>
</tr>
</tbody>
</table>

