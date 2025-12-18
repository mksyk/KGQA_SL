# 智能问答机器人 - Streamlit 版本

这是基于 Flask 版本转换而来的 Streamlit 应用，保持了所有后端逻辑不变。

## 功能特性

- ✅ 用户登录/注册系统
- ✅ 多窗口对话管理
- ✅ 工作流选择（支持多个 Dify 工作流）
- ✅ NER 实体识别（医疗实体）
- ✅ 知识图谱检索（Neo4j）
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
├── app.py              # Streamlit 主应用文件
├── config.py           # 配置文件
├── ner_module.py       # NER 模块
├── kg_retriever.py     # 知识图谱检索模块
├── user_storage.py     # 用户存储模块
├── requirements.txt    # 依赖列表
└── tmp_data/          # 用户数据存储目录（自动创建）
```

## 与 Flask 版本的主要区别

1. **前端框架**：使用 Streamlit 替代 Flask + HTML/CSS/JS
2. **状态管理**：使用 `st.session_state` 替代 Flask session
3. **UI 组件**：使用 Streamlit 内置组件（`st.chat_message`, `st.sidebar` 等）
4. **后端逻辑**：完全保持不变（NER、KG 检索、API 调用等）

## 注意事项

- 确保 Neo4j 数据库正在运行
- 确保 Ollama 服务正在运行（用于生成 embedding）
- 确保 Dify API 可访问
- NER 模型首次加载可能需要一些时间

