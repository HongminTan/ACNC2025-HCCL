# AlgResourceRequest<a name="ZH-CN_TOPIC_0000002027049104"></a>

## 功能说明

AlgResourceRequest结构体用于承载executor执行需要的资源诉求，包含从流数量、主从流同步需要的notify数量、Scratch Buffer、建链诉求等信息，由通信算法层计算并赋值。

## 原型定义

```
struct AlgResourceRequest {
    u64 scratchMemSize = 0;
    u32 streamNum = 0;
    u32 notifyNum = 0;
    bool needAivBuffer = false;
    DeviceMode mode = DeviceMode::HOST;
    OpCommTransport opTransport;
    void Describe()
    {
        HCCL_DEBUG("[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], needAivBuffer[%u], "
            "DeviceMode[%d].", scratchMemSize, streamNum, notifyNum, needAivBuffer, mode);
    };
};
```

## 成员介绍

<table><thead align="left"><tr id="row18958414101111"><th class="cellrowborder" valign="top" width="22.662266226622663%" id="mcps1.1.4.1.1"><p id="p9958614111110"><a name="p9958614111110"></a><a name="p9958614111110"></a>成员</p>
</th>
<th class="cellrowborder" valign="top" width="18.421842184218423%" id="mcps1.1.4.1.2"><p id="p89581614131117"><a name="p89581614131117"></a><a name="p89581614131117"></a>类型</p>
</th>
<th class="cellrowborder" valign="top" width="58.91589158915891%" id="mcps1.1.4.1.3"><p id="p17958114121116"><a name="p17958114121116"></a><a name="p17958114121116"></a>说明</p>
</th>
</tr>
</thead>
<tbody><tr id="row159588144119"><td class="cellrowborder" valign="top" width="22.662266226622663%" headers="mcps1.1.4.1.1 "><p id="p116754327116"><a name="p116754327116"></a><a name="p116754327116"></a>scratchMemSize</p>
</td>
<td class="cellrowborder" valign="top" width="18.421842184218423%" headers="mcps1.1.4.1.2 "><p id="p1995961417119"><a name="p1995961417119"></a><a name="p1995961417119"></a>u64</p>
</td>
<td class="cellrowborder" valign="top" width="58.91589158915891%" headers="mcps1.1.4.1.3 "><p id="p1295918143115"><a name="p1295918143115"></a><a name="p1295918143115"></a>Executor执行需要的Scratch Buffer大小，用于暂存算法运行的中间结果。</p>
</td>
</tr>
<tr id="row11959101412112"><td class="cellrowborder" valign="top" width="22.662266226622663%" headers="mcps1.1.4.1.1 "><p id="p1227315321312"><a name="p1227315321312"></a><a name="p1227315321312"></a>streamNum</p>
</td>
<td class="cellrowborder" valign="top" width="18.421842184218423%" headers="mcps1.1.4.1.2 "><p id="p9959121413116"><a name="p9959121413116"></a><a name="p9959121413116"></a>u32</p>
</td>
<td class="cellrowborder" valign="top" width="58.91589158915891%" headers="mcps1.1.4.1.3 "><p id="p19959191421113"><a name="p19959191421113"></a><a name="p19959191421113"></a>Executor执行需要的从流数量。</p>
</td>
</tr>
<tr id="row17381182171416"><td class="cellrowborder" valign="top" width="22.662266226622663%" headers="mcps1.1.4.1.1 "><p id="p8613162913142"><a name="p8613162913142"></a><a name="p8613162913142"></a>notifyNum</p>
</td>
<td class="cellrowborder" valign="top" width="18.421842184218423%" headers="mcps1.1.4.1.2 "><p id="p1651112335146"><a name="p1651112335146"></a><a name="p1651112335146"></a>u32</p>
</td>
<td class="cellrowborder" valign="top" width="58.91589158915891%" headers="mcps1.1.4.1.3 "><p id="p1538152171410"><a name="p1538152171410"></a><a name="p1538152171410"></a>主从流同步需要的notify数量。</p>
</td>
</tr>
<tr id="row1695789114915"><td class="cellrowborder" valign="top" width="22.662266226622663%" headers="mcps1.1.4.1.1 "><p id="p1295718912498"><a name="p1295718912498"></a><a name="p1295718912498"></a>mode</p>
</td>
<td class="cellrowborder" valign="top" width="18.421842184218423%" headers="mcps1.1.4.1.2 "><p id="p15957139114915"><a name="p15957139114915"></a><a name="p15957139114915"></a>DeviceMode</p>
</td>
<td class="cellrowborder" valign="top" width="58.91589158915891%" headers="mcps1.1.4.1.3 "><p id="p17957197490"><a name="p17957197490"></a><a name="p17957197490"></a>用于区分是Host模式，还是AI CPU模式。</p>
<p id="p34191349904"><a name="p34191349904"></a><a name="p34191349904"></a>DeviceMode枚举类型定义如下：</p>
<pre class="screen" id="screen1871016338017"><a name="screen1871016338017"></a><a name="screen1871016338017"></a>enum DeviceMode {
    HOST = 0,
    AICPU = 1
};</pre>
</td>
</tr>
<tr id="row12744122511144"><td class="cellrowborder" valign="top" width="22.662266226622663%" headers="mcps1.1.4.1.1 "><p id="p8232154091720"><a name="p8232154091720"></a><a name="p8232154091720"></a>opTransport</p>
</td>
<td class="cellrowborder" valign="top" width="18.421842184218423%" headers="mcps1.1.4.1.2 "><p id="p485144551714"><a name="p485144551714"></a><a name="p485144551714"></a>OpCommTransport</p>
</td>
<td class="cellrowborder" valign="top" width="58.91589158915891%" headers="mcps1.1.4.1.3 "><p id="p1274582517146"><a name="p1274582517146"></a><a name="p1274582517146"></a>表示Executor执行需要的建链关系。</p>
<p id="p2106104819207"><a name="p2106104819207"></a><a name="p2106104819207"></a>关于OpCommTransport的说明可参见<a href="OpCommTransport.md">OpCommTransport</a>。</p>
</td>
</tr>
</tbody>
</table>

