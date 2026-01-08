"""
配置文件：NER 和知识图谱相关配置
"""
import os

# NER 模型配置
NER_BASE_MODEL_PATH = os.environ.get(
    "NER_BASE_MODEL_PATH", 
    "/home/lx/crj/KG4medLLM/models/qwen2.5-1.5B-instruct"
)
NER_CHECKPOINT_PATH = os.environ.get(
    "NER_CHECKPOINT_PATH",
    "/home/lx/crj/KG4medLLM/output/Qwen2.5-MedNER/checkpoint-1688"
)
NER_DEVICE = os.environ.get("NER_DEVICE", "cuda")

# Neo4j 配置
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "Crj123456")

# Ollama 配置（用于生成 embedding）
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")

# 知识图谱检索配置
KG_TOP_K = int(os.environ.get("KG_TOP_K", "5"))  # 每个实体返回的最相似节点数量
KG_SIMILARITY_THRESHOLD = float(os.environ.get("KG_SIMILARITY_THRESHOLD", "0.5"))  # 相似度阈值

# 三元组Rerank配置
KG_ENABLE_RERANK = os.environ.get("KG_ENABLE_RERANK", "true").lower() == "true"  # 是否启用三元组rerank
KG_RERANK_TOP_N = int(os.environ.get("KG_RERANK_TOP_N", "5"))  # rerank后每个实体返回的最相关三元组数量
KG_RERANK_THRESHOLD = float(os.environ.get("KG_RERANK_THRESHOLD", "0.0"))  # rerank相似度阈值
KG_RERANK_MODEL = os.environ.get("KG_RERANK_MODEL", "bge-m3")  # rerank使用的embedding模型名称

# G-Retriever配置
G_RETRIEVER_ENABLED = True  # 是否启用G-Retriever
G_RETRIEVER_TOPK_NODES = int(os.environ.get("G_RETRIEVER_TOPK_NODES", "3"))  # PCST算法选择的top-k节点数量
G_RETRIEVER_TOPK_EDGES = int(os.environ.get("G_RETRIEVER_TOPK_EDGES", "5"))  # PCST算法选择的top-k边数量
G_RETRIEVER_COST_EDGE = float(os.environ.get("G_RETRIEVER_COST_EDGE", "0.5"))  # PCST算法中边的成本参数
G_RETRIEVER_INITIAL_TOP_K = int(os.environ.get("G_RETRIEVER_INITIAL_TOP_K", "10"))  # 初始检索的节点数量
G_RETRIEVER_MAX_HOP = int(os.environ.get("G_RETRIEVER_MAX_HOP", "2"))  # 最大跳数
G_RETRIEVER_SIMILARITY_THRESHOLD = float(os.environ.get("G_RETRIEVER_SIMILARITY_THRESHOLD", "0.3"))  # G-Retriever相似度阈值（默认0.3，比传统KG检索的0.5更宽松）

# 应用配置
ENABLE_LOGIN = False  # 是否启用登录界面（设为false可直接进入聊天界面，方便测试）

