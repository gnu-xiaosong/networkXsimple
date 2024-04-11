import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties


# 设置matplotlib正常显示中文和负号
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号


class netGraph:
    """
    desc: 简化版的networkX绘制简单且容易上手的神经网络绘图库
    guide:
        from package_xskj_NetworkXsimple import netGraph
        import numpy as np
        import matplotlib.pyplot as plt


        # 设置正常显示符号
        plt.rcParams["axes.unicode_minus"] = False
        sns.set_theme()
        plt.rcParams['font.sans-serif']=['SimHei']   # 用黑体显示中文


        # 实例化netGraph对象
        networkGraph =netGraph(type=1)

        # 增加网络节点
        networkGraph.addNode(
            name="节点名称标识",
            pos=(所在网络层layer, 该网络层中1开始从下往上的节点索引),
            label= "该节点node的描述label",
            label_color="label的颜色，默认black",
            # 出度edge边信息
            nexts=[
                {
                    "node": "连接node的name",
                    "label": "edge边标签",
                    "color": "edge边标签颜色",
                    "weight": edge边的权重
                },
            ],
            previous=[
                 {
                    "node": "连接node的name",
                    "label": "edge边标签",
                    "color": "edge边标签颜色",
                    "weight": edge边的权重
                },
            ]
        )

        # 增加edge边
        edge = {
            "node":  name,        # 入度连接节点的node     str     对应于节点在网络中唯一标识符 name
            "label":  label,      # 入度edge的标签        str     作为edge信息展示   默认为None
            "color":  color,      # 标签label和edge的颜色  str     默认为 black
            "weight": weight      # 入度edge的权重        float    默认为1
          }

        # 绘制网络图
        networkGraph.draw()

        # 获取nx.Graph()实例，用于直接访问networkX定义的属性和方法，便于用户更多的操作网络图的自主性
        G = networkGraph.getNetworkXInstance()
        。。。。。
    """




    def __init__(self, type=0, step= 1, AlgorithmType="line", node_size=1000, **keyarg):
        """
        desc: 初始化netGraph
        parameters:
            type:  图类型  int
                   0    无向图
                   1    有向图
            step: 同层节点间的步长间隔 float
                  默认为1
            AlgorithmType:  节点坐标不惧算法类型 str
                            默认 line型
            node_size:      节点大小  int
                            默认1000
            keyarg
        """
        if type == 0:
            self.G = nx.Graph()                # 无向图
        else:
            self.G = nx.DiGraph()              # 有向图

        self.nodes_arr = []                    # 节点数组
        self.pos = {}                          # 节点位置信息字典
        self.layers = []                       # 层数
        self.step = step                       # 同层节点间的步长间隔
        self.AlgorithmType = AlgorithmType     # 调整算法类型
        self.node_size = node_size

    def addNode(self, name="", pos=(1, 1), previous=[], nexts=[] ,label="", label_color="black" , **kwargs):
        """
        desc：        增加网络节点node
        paremeters:
            name:   节点名称              str    作为节点的在网络中的唯一标识，所以要保证唯一性，同时也作为节点的显示名称
            label:  节点描述标签           str    位于节点附近的文字展示
            label_color: 节点描述标签的颜色 str    默认为black黑色
            pos：   节点在网络中的位置,网络位置采用层数+节点序号命名  tuple
                    pos=(n,  j)
                    n:  node所在层数layer,从左到右, 从1索引开始
                    j:  该层所在的序号, 从下往上,   从1索引开始
            previous: 入度的edge   list  数组中每一个item元素都作为一条入度edge信息
                      >>previous=[
                          {
                            "node":  name,        # 入度连接节点的node     str     对应于节点在网络中唯一标识符 name
                            "label":  label,      # 入度edge的标签        str     作为edge信息展示   默认为None
                            "color":  color,      # 标签label和edge的颜色  str     默认为 black
                            "weight": weight      # 入度edge的权重        float    默认为1
                          }
                      ]
            nexts: 出度edge信息，参数意义同上
        """
        degrees = []
        for item in previous:
            degrees.append({
                "type": 0,  # 入度
                "node": item["node"],
                "label": item["label"]   if "label" in item else "",
                "color": item["color"]   if "color" in item else "black",
                "weight": item["weight"] if "weight" in item  else 1
            })

        for item in nexts:
            degrees.append({
                "type": 1,  # 出度
                "node": item["node"],
                "label": item["label"] if "label" in item else "",
                "color": item["color"] if "color" in item else "black",
                "weight": item["weight"] if "weight" in item else 1
            })

        node = {
            "id": len(self.nodes_arr) + 1,
            "name": name,
            "label": label,
            "labelColor": label_color,
            "position": self.setPostion(pos),
            "degree": degrees
        }

        # print(node)
        # 判断是否重复设置了相同的node
        if name  in self.pos:
            # print(f"node={name} alread exist!")
            return None

        self.pos[name] = self.setPostion(pos)
        self.nodes_arr.append(node)
        # 增加层数： 判断是否存在layer数组中
        if pos[0] not in self.layers:
            self.layers.append(pos[0])

        # 数据类型转化
        self.netGraphToNetworkX()

    def addEdge(self, e, edge):
        """
        desc: 增加边
        paremeters:
              e (a,b)  tuple
              edge dict 度的信息
        """
        self.G.add_edge(*e, weight = float(edge["weight"]), color=edge["color"])


    def getNetworkXInstance(self):
        """
        desc: 获取networkX的实例化对象，也方便用户直接使用networkX的属性和方法操作网络图 指向 nx.Graph() or nx.DiGraph()
        return G对象
        """

        return  self.G


    def adjust_layer_nodes_position(self):
        """
        desc:  调整除第一层外所有层的坐标值
        paremeters:
            type   str  调整算法类型
        """

        def adjust_line():
            """
            desc: 直线型节点坐标调整算法,只调整除layer=1层的节点
            """
            # 计算第一层的节点数
            layer_1_node = self.compute_layer_nodes(1)

            self.layers.remove(1)
            for layer in self.layers:
                # print(f"-------layer = {layer}")
                # 计算layer层节点数
                layer_node = self.compute_layer_nodes(layer)
                # 计算layer差额 偏量
                bias = np.abs(layer_1_node - layer_node) / 2 * self.step
                # print(f"bias={bias}")
                # 遍历nodes 调整layer层各节点的坐标
                for key in self.pos:
                    # 排除layer=1层
                    if self.pos[key][0] != 1 and self.pos[key][0] == layer:
                        # 获取各节点的坐标
                        x = self.pos[key][0]
                        adj_y = self.pos[key][1] + bias
                        # 更新坐标值
                        self.pos[key] = (x, adj_y)




        if self.AlgorithmType == 'line':
            adjust_line()
        else:
            print("未知调整算法类型")

        # 增加node节点的处的标签
        self.addNodeLabel()


    def setPostion(self, pos):

        def line(pos):
            """
            desc: 直线型坐标布置算法
            """
            # 该层的节点数
            n = self.compute_layer_nodes(pos[0]) + 1

            # 坐标设置
            x = pos[0]
            y = n * self.step

            return (x, y)

        # 默认直线型算法
        return line(pos)

    def compute_layer_nodes(self, l):
        """
        desc: 计算l层的节点数
        return:
            n  int l层的节点数
        """
        n = 0
        for node in self.nodes_arr:
            if node["position"][0] == l:
                n += 1

        return n

    def addEdgeFromNodesArr(self):
        """
        desc: 遍历nodes中所有的degress
        """
        for node in self.nodes_arr:
            if len(node["degree"]) != 0:
                # 遍历所有节点
                for edge in node["degree"]:
                    if edge["type"] == 0:
                        self.addEdge((edge["node"], node["name"]),edge)
                    else:
                        self.addEdge((node["name"], edge["node"]),edge)

                    # 添加edge标签：两点坐标中点公式
                    self.addEdgeLabel(self.pos[edge["node"]], self.pos[node["name"]], edge["label"], edge["color"])

    def netGraphToNetworkX(self):
        last_node = self.nodes_arr[-1]
        self.G.add_node(last_node["name"])
        for node in last_node["degree"]:
            if node["type"] == 0:
                self.G.add_edge(node["node"], last_node["name"], weight=node["weight"])
            else:
                self.G.add_edge(last_node["name"], node["node"], weight=node["weight"])

    def draw(self):
        """
        desc:   绘制网络图
                self.ax  matplotlib對象
        """
        # 绘制图
        fig, self.ax = plt.subplots(1, 1, figsize=(10, 10))

        # 算法调整
        self.adjust_layer_nodes_position()
        # 增加边
        self.addEdgeFromNodesArr()

        # 绘制
        nx.draw(self.G,
                pos=self.pos,
                ax=self.ax,
                with_labels=True,
                node_color='white',
                edgecolors='black',
                linewidths=0.5,
                width=0.5,
                node_size=self.node_size)


        plt.xlim((0, 5))
        plt.ylim((0,15))
        plt.tight_layout()
        plt.title("神经网络图")
        plt.show()

    def addEdgeLabel(self, node1=(), node2=(), label="", color="red"):
        """
        desc: edge标签设置
        paremeters:
            node1   tuple  度其中节点1的坐标
            node2   tuple  度其中节点2的坐标
            label   str    度标签描述
        """
        # print(f"node1 = {node1}  node2={node2}  label={label}")

        # 偏量
        bias =  0
        # 中点坐标
        mid_x, mid_y = (node1[0] + node2[0]) / 2 , (node1[1] + node2[1]) / 2

        font = FontProperties()
        font.set_weight("bold")
        font.set_size(7)


        # 求解度的正切值-》反解求旋转角度
        (x1, y1) = (node1[0], node1[1])
        (x2, y2) = (node2[0], node2[1])

        # in degrees
        bios_degree = 31

        angle = np.arctan2(y1 - y2, x1 - x2) / (2.0 * np.pi) * 360
        if y1 - y2 < 0:
            angle += bios_degree
        elif (y1 - y2)>0:
            angle -= bios_degree

        # 绘制
        self.ax.text(mid_x + bias, mid_y - bias,
                     r'$%s$' % str(label) if label else "",
                     fontproperties=font,
                     rotation=angle,
                     # transform=self.ax.transData,
                     verticalalignment="center",
                     # zorder=1,
                     bbox=   dict(boxstyle="round",
                               # alpha=0.5,
                               ec=(1.0, 1.0, 1.0),
                               fc=(1.0, 1.0, 1.0),
                               ),
                     color=color)


    def findNodeByPosKeyInNodesArr(self, node_name):
        """
        desc: 根据node名称查找节点对象 在nodes_arr中
        paremeters:
            node_name   str / int  node的名称name
        """
        # 获取节点信息
        # print(f"--------node_arr------")
        # print(self.nodes_arr)
        node = {}
        for item in self.nodes_arr:
            if item["name"] == node_name:
                # print(f"{node_name}:")
                # print(item)

                # 获取目标node
                node = item
        return  node


    def addNodeLabel(self):
        """
        desc: node节点标签设置
        paremeters:
            pos_key  str self.pos dict对应的键key
        """

        for pos_key in self.pos:
            # 获取node的坐标 tuple元组
            pos_x_y = self.pos[pos_key]

            # 计算标签的文字大小
            label_len = len(pos_key)
            # 获取目标节点对象
            node = self.findNodeByPosKeyInNodesArr(pos_key)
            # font
            font = FontProperties()
            font.set_weight("bold")
            font.set_size(10)

            if node:
                if pos_x_y[0]==1:
                    # layer==1层
                    x_bias = -label_len * 0.2
                    y_bias = 0

                    # 坐标偏移量
                    pos_x_y_bias = (pos_x_y[0] + x_bias, pos_x_y[1] + y_bias)

                    self.ax.text(*pos_x_y_bias,
                             r'$%s$' % str(node["label"]) if node["label"] else "",
                             verticalalignment='center',
                             horizontalalignment="right",
                             fontproperties=font,
                             bbox=dict(boxstyle="rarrow" ,
                                       ec=(1., 0.5, 0.5),
                                       fc=(1., 0.8, 0.8),
                                       ),
                             color=node["labelColor"])
                else:
                    # other layer
                    x_bias = 0
                    y_bias = -self.node_size/1000
                    # 坐标偏移量
                    pos_x_y_bias = (pos_x_y[0] + x_bias, pos_x_y[1] + y_bias)

                    self.ax.text(*pos_x_y_bias,
                                 r'$%s$' % str(node["label"]) if node["label"] else "",
                                 verticalalignment='center',
                                 horizontalalignment="center",
                                 fontproperties=font,
                                 bbox= None,
                    color = node["labelColor"])
            else:
                # 绘制
                self.ax.text(*pos_x_y_bias, "this node is removed", color=node["labelColor"], fontsize=10)


