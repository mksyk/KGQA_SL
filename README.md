# 智能问答机器人 - Streamlit 版本

这是基于 Flask 版本转换而来的 Streamlit 应用，保持了所有后端逻辑不变。

## 功能特性

- ✅ 用户登录/注册系统
- ✅ 多窗口对话管理
- ✅ 工作流选择（支持多个 Dify 工作流）
- ✅ NER 实体识别（医疗实体）
- ✅ 知识图谱检索（Neo4j）
- ✅ **三元组Rerank模块**：基于embedding相似度对知识图谱三元组进行筛选和排序（新增）
- ✅ 集成 Dify API 进行问答
- ✅ 管理员功能（查看原始响应）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址为 `http://localhost:8501`

## 配置说明

### 环境变量

可以通过环境变量配置以下参数：

- `NER_BASE_MODEL_PATH`: NER 基础模型路径
- `NER_CHECKPOINT_PATH`: NER checkpoint 路径
- `NER_DEVICE`: 设备类型（cuda/cpu）
- `NEO4J_URI`: Neo4j 数据库 URI
- `NEO4J_USER`: Neo4j 用户名
- `NEO4J_PASSWORD`: Neo4j 密码
- `OLLAMA_URL`: Ollama 服务地址
- `KG_TOP_K`: 知识图谱检索返回的最相似节点数量
- `KG_SIMILARITY_THRESHOLD`: 相似度阈值
- `KG_ENABLE_RERANK`: 是否启用三元组rerank功能（true/false，默认true）
- `KG_RERANK_TOP_N`: rerank后每个实体返回的最相关三元组数量（默认5）
- `KG_RERANK_THRESHOLD`: rerank相似度阈值（默认0.0）
- `KG_RERANK_MODEL`: rerank使用的embedding模型名称（默认bge-m3）
- `DIFY_API_URL`: Dify API 地址
- `DIFY_API_KEY`: Dify API Key
- `DIFY_USER_ID`: Dify 用户 ID

### 工作流配置

在 `app.py` 中的 `WORKFLOWS` 字典中配置工作流：

```python
WORKFLOWS = {
    "testflow": {
        "url": "http://172.25.219.127/v1/chat-messages",
        "key": "app-LUFceHJWRbUuhGBDD96VIsJX",
    },
    "agentflow1": {
        "url": "http://172.25.219.127/v1/chat-messages",
        "key": "app-is4ADPVdQhq5ArvSlm5CJty3",
    }
}
```

## 默认账户

- 管理员账户：`admin` / `admin123`

## 项目结构

```
KGQA_SL/
├── app.py                  # Streamlit 主应用文件
├── config.py               # 配置文件
├── ner_module.py           # NER 模块
├── kg_retriever.py         # 知识图谱检索模块（已集成rerank功能）
├── triple_reranker.py      # 三元组Rerank模块（新增）
├── user_storage.py         # 用户存储模块
├── requirements.txt        # 依赖列表
└── tmp_data/              # 用户数据存储目录（自动创建）
```

## 与 Flask 版本的主要区别

1. **前端框架**：使用 Streamlit 替代 Flask + HTML/CSS/JS
2. **状态管理**：使用 `st.session_state` 替代 Flask session
3. **UI 组件**：使用 Streamlit 内置组件（`st.chat_message`, `st.sidebar` 等）
4. **后端逻辑**：完全保持不变（NER、KG 检索、API 调用等）

## 三元组Rerank模块说明

### 开发背景

知识图谱信息提取模块从图谱中提取三元组时，由于数量多且比较杂乱，需要过滤手段来提高检索质量。参考KG-Rank项目中的rerank机制，开发了基于embedding的三元组rerank模块。

### 功能特点

1. **基于Embedding的相似度计算**：使用embedding模型（默认bge-m3）计算用户查询与三元组的语义相似度
2. **三元组筛选和排序**：根据相似度分数对三元组进行排序，只保留最相关的top-n个三元组
3. **可配置的阈值过滤**：支持设置相似度阈值，过滤掉相关性较低的三元组
4. **无缝集成**：已集成到知识图谱检索流程中，可通过配置开关启用/禁用

### 技术实现

- **参考项目**：KG-Rank项目中的CohereReranker机制
- **Embedding模型**：使用embedEntity.py中的模型（bge-m3），可通过配置修改
- **相似度计算**：使用余弦相似度计算查询与三元组的embedding相似度
- **三元组表示**：将三元组（头实体-关系-尾实体）格式化为文本进行embedding计算

### 使用方法

rerank功能默认启用，可通过环境变量或配置文件进行控制：

```python
# 在config.py中配置
KG_ENABLE_RERANK = True  # 启用rerank
KG_RERANK_TOP_N = 5      # 每个实体返回5个最相关的三元组
KG_RERANK_THRESHOLD = 0.0  # 相似度阈值（0.0表示不过滤）
KG_RERANK_MODEL = "bge-m3"  # 使用的embedding模型
```

### 代码修改说明

1. **triple_reranker.py**：新增的三元组rerank模块
   - `TripleReranker`类：实现rerank核心功能
   - `rerank_kg_results`方法：处理完整的知识图谱检索结果
   - 支持自定义embedding模型（参考embedEntity.py中的配置方式）

2. **kg_retriever.py**：修改了`retrieve_knowledge`方法
   - 添加了`enable_rerank`、`query`、`rerank_top_n`、`rerank_threshold`参数
   - 保持向后兼容，默认不启用rerank

3. **app.py**：更新了知识图谱检索调用
   - 传入用户查询用于rerank
   - 使用配置文件中的rerank参数

### 修改Embedding模型

如果需要修改embedding模型，可以在`triple_reranker.py`中的`generate_embedding`方法中修改：

```python
# 在TripleReranker.__init__中指定模型名称
reranker = TripleReranker(ollama_url, model_name="your-model-name")

# 或者在generate_embedding方法中添加GPU选择等选项
payload = {
    "model": self.model_name,
    "prompt": text,
    "options": {
        "main_gpu": 3  # 参考embedEntity.py中的配置
    }
}
```

## 注意事项

- 确保 Neo4j 数据库正在运行
- 确保 Ollama 服务正在运行（用于生成 embedding）
- 确保 Dify API 可访问
- NER 模型首次加载可能需要一些时间
- **Rerank功能需要额外的embedding计算**，可能会增加响应时间，但能显著提高检索质量

