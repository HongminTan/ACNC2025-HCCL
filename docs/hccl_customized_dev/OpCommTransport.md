# OpCommTransport<a name="ZH-CN_TOPIC_0000002027207528"></a>

Executor执行过程中，不同的Rank间需要进行通信，Rank间通信依赖框架提前创建好对应的transport链路。Executor需要的建链诉求用结构体OpCommTransport表示。结构体OpCommTransport是一个vector，表示所有层的建链诉求；OpCommTransport中vector元素为LevelNSubCommTransport，表示每一层的建链诉求；一层的建链诉求可能会有多个子通信域组成，因此LevelNSubCommTransport也是一个vector，其成员为SingleSubCommTransport，表示某一层某个子通信域的建链诉求。

-   OpCommTransport：表示所有层的建链诉求，定义如下所示：

    ```
    using OpCommTransport = std::vector<LevelNSubCommTransport>;
    ```

-   LevelNSubCommTransport：表示每一层的建链诉求，定义如下所示：

    ```
    using LevelNSubCommTransport = std::vector<SingleSubCommTransport>;
    ```

- SingleSubCommTransport：表示某一层某个子通信域的建链诉求，定义如下所示：

  ```
  struct SingleSubCommTransport {
      std::vector<TransportRequest> transportRequests;
      std::vector<LINK> links;
      bool isUsedRdma = false;
      u64 taskNum = 0;
      std::map<u32, u32> userRank2subCommRank;
      std::map<u32, u32> subCommRank2UserRank;
      bool supportDataReceivedAck = false;
      LinkMode linkMode = LinkMode::LINK_DUPLEX_MODE;
      bool enableUseOneDoorbell = false;
      bool needVirtualLink =false; // for alltoall 多线程性能提升使用
      std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
  };
  ```

  

  <table><thead align="left"><tr id="row850716311303"><th class="cellrowborder" valign="top" width="27.382738273827385%" id="mcps1.1.4.1.1"><p id="p18507133116306"><a name="p18507133116306"></a><a name="p18507133116306"></a>成员</p>
  </th>
  <th class="cellrowborder" valign="top" width="24.222422242224223%" id="mcps1.1.4.1.2"><p id="p12507163118306"><a name="p12507163118306"></a><a name="p12507163118306"></a>类型</p>
  </th>
  <th class="cellrowborder" valign="top" width="48.394839483948395%" id="mcps1.1.4.1.3"><p id="p17507153119301"><a name="p17507153119301"></a><a name="p17507153119301"></a>说明</p>
  </th>
  </tr>
  </thead>
  <tbody><tr id="row1850711311304"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p63371243237"><a name="p63371243237"></a><a name="p63371243237"></a>transportRequests</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p1873716122233"><a name="p1873716122233"></a><a name="p1873716122233"></a>std::vector&lt;TransportRequest&gt;</p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p35071631103019"><a name="p35071631103019"></a><a name="p35071631103019"></a>当前rank到子平面内其他rank的建链诉求，size为子平面内rank的个数。</p>
  <p id="p2827924172611"><a name="p2827924172611"></a><a name="p2827924172611"></a>关于TransportRequest类型的说明，可参见<a href="#li19491152018412">TransportRequest</a>。</p>
  </td>
  </tr>
  <tr id="row13507133110303"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p584665022313"><a name="p584665022313"></a><a name="p584665022313"></a>links</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p350723103014"><a name="p350723103014"></a><a name="p350723103014"></a>std::vector&lt;LINK&gt;</p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p68878552254"><a name="p68878552254"></a><a name="p68878552254"></a>返回建链诉求时，该字段不用填。框架创建好链路之后，会填入这个字段。（注：建链诉求和建链响应使用了相同的结构体）</p>
  </td>
  </tr>
  <tr id="row7507831103016"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p20666117172412"><a name="p20666117172412"></a><a name="p20666117172412"></a>isUsedRdma</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p1150743193010"><a name="p1150743193010"></a><a name="p1150743193010"></a>bool</p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><div class="p" id="p10881322152317"><a name="p10881322152317"></a><a name="p10881322152317"></a>该子平面内使用RDMA建链还是SDMA建链。<a name="ul10634183364213"></a><a name="ul10634183364213"></a><ul id="ul10634183364213"><li>true：表示使用RDMA建链。</li><li>flase：表示使用SDMA建链。</li></ul>
  </div>
  </td>
  </tr>
  <tr id="row165075319306"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p135338192415"><a name="p135338192415"></a><a name="p135338192415"></a>taskNum</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p150733116304"><a name="p150733116304"></a><a name="p150733116304"></a>u64</p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p050743116304"><a name="p050743116304"></a><a name="p050743116304"></a>图模式，ring环建链使用，其他场景可不关注。</p>
  </td>
  </tr>
  <tr id="row102761629174311"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p1727617297433"><a name="p1727617297433"></a><a name="p1727617297433"></a>userRank2subCommRank</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p42768296436"><a name="p42768296436"></a><a name="p42768296436"></a><span id="ph52821014121616"><a name="ph52821014121616"></a><a name="ph52821014121616"></a>std::map&lt;u32, u32&gt;</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p132761229134311"><a name="p132761229134311"></a><a name="p132761229134311"></a><span id="ph7156432191614"><a name="ph7156432191614"></a><a name="ph7156432191614"></a>记录user rank和子通信域内局部rank的对应关系</span><span id="ph15465420135818"><a name="ph15465420135818"></a><a name="ph15465420135818"></a>。</span></p>
  </td>
  </tr>
  <tr id="row14224183104313"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p5224831174316"><a name="p5224831174316"></a><a name="p5224831174316"></a>subCommRank2UserRank</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p18224231164318"><a name="p18224231164318"></a><a name="p18224231164318"></a><span id="ph18237142611610"><a name="ph18237142611610"></a><a name="ph18237142611610"></a>std::map&lt;u32, u32&gt;</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p92241631204313"><a name="p92241631204313"></a><a name="p92241631204313"></a><span id="ph1776232261712"><a name="ph1776232261712"></a><a name="ph1776232261712"></a>记录子通信域内局部rank<span id="ph84982029121711"><a name="ph84982029121711"></a><a name="ph84982029121711"></a>和user rank</span>的对应关系</span><span id="ph1924522195810"><a name="ph1924522195810"></a><a name="ph1924522195810"></a>。</span></p>
  </td>
  </tr>
  <tr id="row1653161192515"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p1447163815256"><a name="p1447163815256"></a><a name="p1447163815256"></a>supportDataReceivedAck</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p16531418251"><a name="p16531418251"></a><a name="p16531418251"></a>bool</p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><div class="p" id="p8653810255"><a name="p8653810255"></a><a name="p8653810255"></a>RDMA场景下数据传输是否进行额外的后同步，默认值为false。<a name="ul1633795115819"></a><a name="ul1633795115819"></a><ul id="ul1633795115819"><li>true：需要。</li><li>flase：不需要。</li></ul>
  </div>
  </td>
  </tr>
  <tr id="row1926124816433"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p1892613482436"><a name="p1892613482436"></a><a name="p1892613482436"></a>linkMode</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p3927194864316"><a name="p3927194864316"></a><a name="p3927194864316"></a><span id="ph14685114317598"><a name="ph14685114317598"></a><a name="ph14685114317598"></a>LinkMode</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p119275486435"><a name="p119275486435"></a><a name="p119275486435"></a><span id="ph660614511535"><a name="ph660614511535"></a><a name="ph660614511535"></a>链路模式</span><span id="ph1338512995819"><a name="ph1338512995819"></a><a name="ph1338512995819"></a>。</span></p>
  <p id="p386622911326"><a name="p386622911326"></a><a name="p386622911326"></a><span id="ph74688151420"><a name="ph74688151420"></a><a name="ph74688151420"></a>LinkMode</span>类型定义如下：</p>
  <pre class="screen" id="screen7769240175311"><a name="screen7769240175311"></a><a name="screen7769240175311"></a>enum class SyncMode {
      LINK_SIMPLEX_MODE= 0,   单工模式
      LINK_DUPLEX_MODE = 1,   双工模式(默认)
      LINK_RESERVED_MODE      保留项
  };</pre>
  </td>
  </tr>
  <tr id="row1839105624310"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p784010562433"><a name="p784010562433"></a><a name="p784010562433"></a>enableUseOneDoorbell</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p14840175619438"><a name="p14840175619438"></a><a name="p14840175619438"></a><span id="ph4287143915914"><a name="ph4287143915914"></a><a name="ph4287143915914"></a>bool</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p138400565436"><a name="p138400565436"></a><a name="p138400565436"></a><span id="ph067071735115"><a name="ph067071735115"></a><a name="ph067071735115"></a>是否使用单doorbell模式</span><span id="ph1372153145817"><a name="ph1372153145817"></a><a name="ph1372153145817"></a>。</span></p>
  <p id="p56937316576"><a name="p56937316576"></a><a name="p56937316576"></a>在某些特定算法下，会需要发送内存不连续的数据，开启此模式后，只在最后一次下发doorbell任务，而不是每片内存都下发doorbell任务。</p>
  <a name="ul15285733145917"></a><a name="ul15285733145917"></a><ul id="ul15285733145917"><li>true：开启单doorbell模式。</li><li>false：不开启单doorbell模式。</li></ul>
  </td>
  </tr>
  <tr id="row182855464419"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p528520413442"><a name="p528520413442"></a><a name="p528520413442"></a>needVirtualLink</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p42851544449"><a name="p42851544449"></a><a name="p42851544449"></a><span id="ph5147736165616"><a name="ph5147736165616"></a><a name="ph5147736165616"></a>bool</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p142851245445"><a name="p142851245445"></a><a name="p142851245445"></a><span id="ph5121195510517"><a name="ph5121195510517"></a><a name="ph5121195510517"></a>是否需要</span><span id="ph1656804045411"><a name="ph1656804045411"></a><a name="ph1656804045411"></a>创建虚拟链路（用于alltoall多线程性能提升</span><span id="ph6676012145910"><a name="ph6676012145910"></a><a name="ph6676012145910"></a>）</span></p>
  </td>
  </tr>
  <tr id="row662113617440"><td class="cellrowborder" valign="top" width="27.382738273827385%" headers="mcps1.1.4.1.1 "><p id="p46217664416"><a name="p46217664416"></a><a name="p46217664416"></a>virtualLinks</p>
  </td>
  <td class="cellrowborder" valign="top" width="24.222422242224223%" headers="mcps1.1.4.1.2 "><p id="p262217619441"><a name="p262217619441"></a><a name="p262217619441"></a><span id="ph242351115616"><a name="ph242351115616"></a><a name="ph242351115616"></a>std::vector&lt;LINK&gt;</span></p>
  </td>
  <td class="cellrowborder" valign="top" width="48.394839483948395%" headers="mcps1.1.4.1.3 "><p id="p062216614416"><a name="p062216614416"></a><a name="p062216614416"></a><span id="ph15689182316575"><a name="ph15689182316575"></a><a name="ph15689182316575"></a>needVirtualLink</span><span id="ph10230102875719"><a name="ph10230102875719"></a><a name="ph10230102875719"></a>为true时，</span><span id="ph137171321105515"><a name="ph137171321105515"></a><a name="ph137171321105515"></a>框架会创建虚拟链路，然后填入这个字段。</span></p>
  </td>
  </tr>
  </tbody>
  </table>

-   TransportRequest：表示当前rank到子平面内其他rank的建链诉求，定义如下所示：

    ```
    struct TransportRequest {
        bool isValid = false;
        RankId localUserRank = 0;
        RankId remoteUserRank = 0;
        TransportMemType inputMemType = TransportMemType::RESERVED;
        TransportMemType outputMemType = TransportMemType::RESERVED;
    };
    ```

    
    <table><thead align="left"><tr id="row7577203473615"><th class="cellrowborder" valign="top" width="21.362136213621362%" id="mcps1.1.4.1.1"><p id="p15771434113613"><a name="p15771434113613"></a><a name="p15771434113613"></a>成员</p>
    </th>
    <th class="cellrowborder" valign="top" width="22.562256225622562%" id="mcps1.1.4.1.2"><p id="p1157713411361"><a name="p1157713411361"></a><a name="p1157713411361"></a>类型</p>
    </th>
    <th class="cellrowborder" valign="top" width="56.07560756075607%" id="mcps1.1.4.1.3"><p id="p1057713343368"><a name="p1057713343368"></a><a name="p1057713343368"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row12577153413361"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1966213456369"><a name="p1966213456369"></a><a name="p1966213456369"></a>isValid</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p20594496364"><a name="p20594496364"></a><a name="p20594496364"></a>bool</p>
    </td>
    <td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><div class="p" id="p07037513518"><a name="p07037513518"></a><a name="p07037513518"></a>该链路是否需要生效。<a name="ul799616557510"></a><a name="ul799616557510"></a><ul id="ul799616557510"><li>true：表示生效。</li><li>false：表示不生效。</li></ul>
    </div>
    <p id="p1057715340365"><a name="p1057715340365"></a><a name="p1057715340365"></a>算法返回的建链诉求是按需建链，不需要建链的场景下，该字段填false。</p>
    </td>
    </tr>
    <tr id="row13577113423610"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p14551530133815"><a name="p14551530133815"></a><a name="p14551530133815"></a>localUserRank</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p183874415384"><a name="p183874415384"></a><a name="p183874415384"></a>u32</p>
    </td>
    <td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p596525413385"><a name="p596525413385"></a><a name="p596525413385"></a>本rank对应的userRank。</p>
    </td>
    </tr>
    <tr id="row35772347366"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p1248433911390"><a name="p1248433911390"></a><a name="p1248433911390"></a>remoteUserRank</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p185771234163617"><a name="p185771234163617"></a><a name="p185771234163617"></a>u32</p>
    </td>
    <td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p7577113413611"><a name="p7577113413611"></a><a name="p7577113413611"></a>远端rank对应的userRank。</p>
    </td>
    </tr>
    <tr id="row195771234173612"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p957313520406"><a name="p957313520406"></a><a name="p957313520406"></a>inputMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p144904139403"><a name="p144904139403"></a><a name="p144904139403"></a>TransportMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1957811347366"><a name="p1957811347366"></a><a name="p1957811347366"></a>建链使用的输入内存类型。</p>
    <p id="p1747811197295"><a name="p1747811197295"></a><a name="p1747811197295"></a>枚举类型TransportMemType的定义如下：</p>
    <pre class="screen" id="screen1145514972910"><a name="screen1145514972910"></a><a name="screen1145514972910"></a>enum TransportMemType {
        CCL_INPUT = 0,
        CCL_OUTPUT,
        SCRATCH,
        PARAM_INPUT,
        PARAM_OUTPUT,
        AIV_INPUT,
        AIV_OUTPUT,
        RESERVED
    };</pre>
    </td>
    </tr>
    <tr id="row1157873414365"><td class="cellrowborder" valign="top" width="21.362136213621362%" headers="mcps1.1.4.1.1 "><p id="p332261284111"><a name="p332261284111"></a><a name="p332261284111"></a>outputMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="22.562256225622562%" headers="mcps1.1.4.1.2 "><p id="p11339125714015"><a name="p11339125714015"></a><a name="p11339125714015"></a>TransportMemType</p>
    </td>
    <td class="cellrowborder" valign="top" width="56.07560756075607%" headers="mcps1.1.4.1.3 "><p id="p1257893413363"><a name="p1257893413363"></a><a name="p1257893413363"></a>建链使用的输出内存类型。</p>
    </td>
    </tr>
    </tbody>
    </table>

