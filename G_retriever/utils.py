"""
G-Retriever工具函数
"""
import logging
import sys
import os
from typing import Dict, Any, Optional
from neo4j import GraphDatabase

# 添加父目录到路径，以便导入kg_retriever和config
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from kg_retriever import get_kg_retriever
import config as app_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# 全局G-Retriever实例（延迟加载）
_g_retriever = None


def get_g_retriever():
    """获取G-Retriever实例（单例模式）"""
    global _g_retriever
    if _g_retriever is None:
        from .g_retriever import GRetriever
        
        # 获取Neo4j驱动（从kg_retriever中复用）
        kg_retriever = get_kg_retriever()
        neo4j_driver = kg_retriever.neo4j_driver
        
        # 优先使用G-Retriever专用的相似度阈值，如果没有配置则使用传统KG检索的阈值
        similarity_threshold = getattr(app_config, 'G_RETRIEVER_SIMILARITY_THRESHOLD', app_config.KG_SIMILARITY_THRESHOLD)
        
        _g_retriever = GRetriever(
            neo4j_driver=neo4j_driver,
            ollama_url=app_config.OLLAMA_URL,
            topk_nodes=app_config.G_RETRIEVER_TOPK_NODES,
            topk_edges=app_config.G_RETRIEVER_TOPK_EDGES,
            cost_edge=app_config.G_RETRIEVER_COST_EDGE,
            similarity_threshold=similarity_threshold
        )
        logger.info("G-Retriever实例初始化完成")
    
    return _g_retriever

