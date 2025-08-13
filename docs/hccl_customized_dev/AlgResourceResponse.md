# AlgResourceResponse<a name="ZH-CN_TOPIC_0000002063167329"></a>

## 功能说明

AlgResourceResponse结构体用于存储资源创建的结果，由通信框架层创建并赋值。

## 原型定义

```
struct AlgResourceResponse {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    std::vector<Stream> slaveStreams;
    std::vector<Stream> slaveDevStreams;
    std::vector<std::shared_ptr<LocalNotify> > notifiesM2S;  // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesS2M;  // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevM2S;  // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify> > notifiesDevS2M;  // 大小等同于slaveStreams
    OpCommTransport opTransportResponse;
};
```

## 成员介绍

<table><thead align="left"><tr id="row7577203473615"><th class="cellrowborder" valign="top" width="21.362136213621362%" id="mcps1.1.4.1.1"><p id="p15771434113613"><a name="p15771434113613"></a><a name="p15771434113613"></a>成员</p>
</th>
<th class="cellrowborder" valign="top" width="22.562256225622562%" id="mcps1.1.4.1.2"><p id="p1157713411361"><a name="p1157713411361"></a><a name="p1157713411361"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="56.07560756075607%" id="mcps1.1.4.1.3"><p id="p1057713343368"><a name="p1057713343368"></a><a name="p1057713343368"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row12577153413361"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p144671421145416"><a name="p144671421145416"></a><a name="p144671421145416"></a>cclInputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p57291740115511"><a name="p57291740115511"></a><a name="p57291740115511"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1057715340365"><a name="p1057715340365"></a><a name="p1057715340365"></a>和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输入。</p>
</td>
</tr>
<tr id="row13577113423610"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1748873013542"><a name="p1748873013542"></a><a name="p1748873013542"></a>cclOutputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p164519438550"><a name="p164519438550"></a><a name="p164519438550"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p38972055205817"><a name="p38972055205817"></a><a name="p38972055205817"></a>和通信域绑定的一块Device内存，单算子模式下可用于建链，通常用于缓存输出。</p>
</td>
</tr>
<tr id="row35772347366"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p2095212359548"><a name="p2095212359548"></a><a name="p2095212359548"></a>paramInputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p84594454559"><a name="p84594454559"></a><a name="p84594454559"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p7577113413611"><a name="p7577113413611"></a><a name="p7577113413611"></a>算子的输入Device内存，图模式下可用于建链。</p>
</td>
</tr>
<tr id="row195771234173612"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1120215414546"><a name="p1120215414546"></a><a name="p1120215414546"></a>paramOutputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p619120482556"><a name="p619120482556"></a><a name="p619120482556"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1823821165914"><a name="p1823821165914"></a><a name="p1823821165914"></a>算子的输出Device内存，图模式下可用于建链。</p>
</td>
</tr>
<tr id="row1157873414365"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p18715104620546"><a name="p18715104620546"></a><a name="p18715104620546"></a>scratchMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p7518115014557"><a name="p7518115014557"></a><a name="p7518115014557"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p185411729135915"><a name="p185411729135915"></a><a name="p185411729135915"></a>算子的workspace内存，单算子或图模式下均可能使用，可用于建链。</p>
</td>
</tr>
<tr id="row1545611515549"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p119473135516"><a name="p119473135516"></a><a name="p119473135516"></a>aivInputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p169181052145515"><a name="p169181052145515"></a><a name="p169181052145515"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p104561351125419"><a name="p104561351125419"></a><a name="p104561351125419"></a>算子的workspace内存，仅aiv场景使用。</p>
</td>
</tr>
<tr id="row9503115595415"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p2092613625513"><a name="p2092613625513"></a><a name="p2092613625513"></a>aivOutputMem</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p646155411550"><a name="p646155411550"></a><a name="p646155411550"></a>内存对象</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p134291251614"><a name="p134291251614"></a><a name="p134291251614"></a>算子的workspace内存，仅aiv场景使用。</p>
</td>
</tr>
<tr id="row13876757131116"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p12876457111112"><a name="p12876457111112"></a><a name="p12876457111112"></a>slaveStreams</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p197747176143"><a name="p197747176143"></a><a name="p197747176143"></a>流对象列表</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p787605714118"><a name="p787605714118"></a><a name="p787605714118"></a>算子需要的从流stream对象。</p>
</td>
</tr>
<tr id="row1076310817125"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p12763178101212"><a name="p12763178101212"></a><a name="p12763178101212"></a>slaveDevStreams</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p976317851212"><a name="p976317851212"></a><a name="p976317851212"></a><span id="ph14859111618494"><a name="ph14859111618494"></a><a name="ph14859111618494"></a>流对象列表</span></p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1976315814129"><a name="p1976315814129"></a><a name="p1976315814129"></a><span id="ph113906342492"><a name="ph113906342492"></a><a name="ph113906342492"></a>aicpu展开模式下，算子需要的从流stream对象。</span></p>
</td>
</tr>
<tr id="row1594217513126"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p169421951101219"><a name="p169421951101219"></a><a name="p169421951101219"></a>notifiesM2S</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p1573373112148"><a name="p1573373112148"></a><a name="p1573373112148"></a>notify对象列表</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p5734183115142"><a name="p5734183115142"></a><a name="p5734183115142"></a>算子主<span id="ph11241182915113"><a name="ph11241182915113"></a><a name="ph11241182915113"></a>流</span><span id="ph1895112522117"><a name="ph1895112522117"></a><a name="ph1895112522117"></a>通知</span>从流需要的notify资源。</p>
</td>
</tr>
<tr id="row8496349141214"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1749664920123"><a name="p1749664920123"></a><a name="p1749664920123"></a>notifiesS2M</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p449614961212"><a name="p449614961212"></a><a name="p449614961212"></a><span id="ph666081112111"><a name="ph666081112111"></a><a name="ph666081112111"></a>notify对象列表</span></p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p449619490120"><a name="p449619490120"></a><a name="p449619490120"></a><span id="ph1233625914110"><a name="ph1233625914110"></a><a name="ph1233625914110"></a>算子<span id="ph536675423"><a name="ph536675423"></a><a name="ph536675423"></a>从流</span></span><span id="ph1233511591817"><a name="ph1233511591817"></a><a name="ph1233511591817"></a>通知</span><span id="ph1165512712214"><a name="ph1165512712214"></a><a name="ph1165512712214"></a>主</span><span id="ph13335259218"><a name="ph13335259218"></a><a name="ph13335259218"></a>流</span><span id="ph4336205914112"><a name="ph4336205914112"></a><a name="ph4336205914112"></a>需要的notify资源。</span></p>
</td>
</tr>
<tr id="row5712114614128"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p137123468124"><a name="p137123468124"></a><a name="p137123468124"></a>notifiesDevM2S</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p1671284661211"><a name="p1671284661211"></a><a name="p1671284661211"></a><span id="ph9461912216"><a name="ph9461912216"></a><a name="ph9461912216"></a>notify对象列表</span></p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1371216469128"><a name="p1371216469128"></a><a name="p1371216469128"></a><span id="ph0385619316"><a name="ph0385619316"></a><a name="ph0385619316"></a>aicpu展开模式下，</span><span id="ph1813819541721"><a name="ph1813819541721"></a><a name="ph1813819541721"></a>算子主</span><span id="ph61386541928"><a name="ph61386541928"></a><a name="ph61386541928"></a>流</span><span id="ph13138125412212"><a name="ph13138125412212"></a><a name="ph13138125412212"></a>通知</span><span id="ph21391054828"><a name="ph21391054828"></a><a name="ph21391054828"></a>从流需要的notify资源。</span></p>
</td>
</tr>
<tr id="row195971096130"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p15971694132"><a name="p15971694132"></a><a name="p15971694132"></a>notifiesDevS2M</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p13597109111320"><a name="p13597109111320"></a><a name="p13597109111320"></a><span id="ph6463133120"><a name="ph6463133120"></a><a name="ph6463133120"></a>notify对象列表</span></p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1959716981319"><a name="p1959716981319"></a><a name="p1959716981319"></a><span id="ph131414714318"><a name="ph131414714318"></a><a name="ph131414714318"></a>aicpu展开模式下，</span><span id="ph4662165617219"><a name="ph4662165617219"></a><a name="ph4662165617219"></a>算子<span id="ph3662135614213"><a name="ph3662135614213"></a><a name="ph3662135614213"></a>从流</span></span><span id="ph8662155612212"><a name="ph8662155612212"></a><a name="ph8662155612212"></a>通知</span><span id="ph1366216566216"><a name="ph1366216566216"></a><a name="ph1366216566216"></a>主</span><span id="ph19662115618216"><a name="ph19662115618216"></a><a name="ph19662115618216"></a>流</span><span id="ph9662656829"><a name="ph9662656829"></a><a name="ph9662656829"></a>需要的notify资源。</span></p>
</td>
</tr>
<tr id="row11790142617554"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1981532135514"><a name="p1981532135514"></a><a name="p1981532135514"></a>opTransportResponse</p>
</td>
<td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p149019391569"><a name="p149019391569"></a><a name="p149019391569"></a>建链表示结构体</p>
</td>
<td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p16791192625516"><a name="p16791192625516"></a><a name="p16791192625516"></a>和建链诉求是同一个结构体，可通过里面的links字段获取建好的链路。</p>
</td>
</tr>
</tbody>
</table>

