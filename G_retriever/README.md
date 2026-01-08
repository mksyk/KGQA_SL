# G-Retriever模块

G-Retriever是一个基于PCST（Prize-Collecting Steiner Tree）算法的知识图谱子图检索模块，用于从知识图谱中检索相关节点和边，构建最优子图来回答问题。

## 功能特性

1. **边Embedding管理**：为边类型生成embedding并高效存储（约10种边类型）
2. **节点和边检索**：根据查询从知识图谱中检索相关节点和边
3. **PCST子图构建**：使用PCST算法构建最优子图
4. **前端可视化**：支持子图的交互式可视化

## 模块结构

```
G-retriever/
├── __init__.py           # 模块初始化
├── edge_embedding.py     # 边embedding管理
├── pcst_algorithm.py     # PCST算法实现
├── g_retriever.py        # G-Retriever主模块
├── utils.py              # 工具函数
└── README.md             # 本文档
```

## 安装依赖

```bash
pip install pcst_fast
```

或者从requirements.txt安装：
```bash
pip install -r requirements.txt
```

## 配置

在`config.py`中配置G-Retriever参数：

```python
# G-Retriever配置
G_RETRIEVER_ENABLED = True  # 是否启用G-Retriever
G_RETRIEVER_TOPK_NODES = 3  # PCST算法选择的top-k节点数量
G_RETRIEVER_TOPK_EDGES = 5  # PCST算法选择的top-k边数量
G_RETRIEVER_COST_EDGE = 0.5  # PCST算法中边的成本参数
G_RETRIEVER_INITIAL_TOP_K = 10  # 初始检索的节点数量
G_RETRIEVER_MAX_HOP = 2  # 最大跳数
```

## 使用方法

### 在代码中使用

```python
from G_retriever.utils import get_g_retriever

# 获取G-Retriever实例
g_retriever = get_g_retriever()

# 构建子图
subgraph_result = g_retriever.build_subgraph(
    query="你的查询文本",
    top_k=10,
    max_hop=2
)

# 访问结果
nodes = subgraph_result["nodes"]
edges = subgraph_result["edges"]
subgraph_info = subgraph_result["subgraph_info"]
```

### 在主应用中使用

1. 设置环境变量或修改`config.py`：
   ```python
   G_RETRIEVER_ENABLED = True
   ```

2. 在主应用（app.py）中，G-Retriever会自动启用并替代传统的KG检索方式。

3. 在问答界面中，子图结果会自动显示在"G-Retriever子图结果"部分。

## PCST算法说明

PCST（Prize-Collecting Steiner Tree）算法是一种用于在图中找到最优子图的算法。在我们的实现中：

1. **节点Prize**：根据节点与查询的余弦相似度，选择top-k节点作为候选节点
2. **边Prize**：根据边类型与查询的余弦相似度，选择top-k边类型
3. **成本函数**：边的成本由参数`cost_edge`控制
4. **算法输出**：返回包含最相关节点和边的连通子图

## 边Embedding存储

由于边类型数量较少（约10种），我们采用以下高效存储方式：

1. 每个边类型生成一个embedding
2. 使用Neo4j中的虚拟节点（标签为`EdgeType`）存储边类型embedding
3. 使用内存缓存提高访问效率

## 故障排除

1. **导入错误**：确保`pcst_fast`已安装
2. **Neo4j连接错误**：检查Neo4j配置是否正确
3. **Ollama连接错误**：确保Ollama服务正在运行
4. **边embedding未初始化**：首次使用时会自动初始化，可能需要一些时间

## 注意事项

1. 首次使用时会初始化所有边类型的embedding，可能需要一些时间
2. PCST算法的参数（topk_nodes, topk_edges, cost_edge）会影响子图的大小和质量
3. 如果查询没有返回结果，会自动回退到传统的KG检索方式

