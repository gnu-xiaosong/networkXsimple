# networkXsimple

This is a simple encapsulation for networkX, which further simplifies the process of networkX for drawing neural network diagrams, so that the author can only focus on the logical writing of neural networks, without spending time on the visual presentation of neural networks. At the same time, this module also inherits all the methods of network and does not affect the normal function calls of networkX.

# install
`
pip install 
`
# guid

### api
```python
graph.addNode(name="node name", desc="node A desc", pos=(int layer, int node No. in this layer) , nexts=[
    {
        "node": "B",
        "label":"B"
    },
    previous=[]
])
```
* nexts : output degress   dict
* previous : input degress  dict
``
 {
        "node": "next node name",
        "label":"edge desc"
    }
``

# exmaple

```python

import numpy as np
import matplotlib.pyplot as plt
import math
import netGraph


# 示例用法
graph = netGraph(type=1)

# 添加节点
graph.addNode(name="A", desc="node A", pos=(1, 1) , nexts=[
    {
        "node": "B",
        "label":"B"
    }
])
graph.addNode(name="B", desc="node B", pos=(1, 2))
graph.addNode(name="C", desc="node C", pos=(1, 3))
graph.addNode(name="D", desc="node A", pos=(1, 4))
graph.addNode(name="E", desc="node A", pos=(1, 5))
graph.addNode(name="F", desc="node A", pos=(1, 6))
graph.addNode(name="I", desc="node A", pos=(1, 7))
graph.addNode(name="G", desc="node B", pos=(2, 1))
graph.addNode(name="H", desc="node B", pos=(2, 2))
graph.addNode(name="Z", desc="node B", pos=(2, 3))

graph.addNode(name="1", desc="node B", pos=(3, 1))
graph.addNode(name="2", desc="node B", pos=(3, 2))
graph.addNode(name="3", desc="node B", pos=(3, 3))

# add edge
graph.addEdge(("B", "G"))
# draw network
graph.draw()
```
### show
![img.png](img.png)
