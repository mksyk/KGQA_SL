"""
知识图谱检索模块：使用向量相似度匹配检索相关实体
"""
import requests
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KnowledgeGraphRetriever:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str, ollama_url: str):
        """
        初始化知识图谱检索器
        
        Args:
            neo4j_uri: Neo4j 数据库连接 URI
            neo4j_user: Neo4j 用户名
            neo4j_password: Neo4j 密码
            ollama_url: Ollama 服务地址（用于生成 embedding）
        """
        self.neo4j_driver = GraphDatabase.driver(neo4j_uri, auth=(neo4j_user, neo4j_password))
        self.ollama_url = ollama_url
        logger.info("知识图谱检索器初始化完成")
    
    def close(self):
        """关闭数据库连接"""
        self.neo4j_driver.close()
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        使用 Ollama 的 bge-m3 模型生成 embedding
        
        Args:
            text: 要生成 embedding 的文本
            
        Returns:
            embedding 向量列表，失败返回 None
        """
        payload = {
            "model": "bge-m3",
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.ollama_url}/api/embeddings",
                json=payload,
                timeout=60
            )
            response.raise_for_status()
            
            result = response.json()
            embedding = result.get("embedding")
            
            if embedding:
                logger.debug(f"成功为文本 '{text}' 生成 embedding，维度: {len(embedding) if isinstance(embedding, list) else 'unknown'}")
                return embedding
            else:
                logger.warning(f"文本 '{text}' 未返回 embedding")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"生成文本 '{text}' 的 embedding 失败: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度值
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def find_similar_nodes(self, entity_text: str, top_k: int = 5, 
                          similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        在 Neo4j 中查找与给定实体文本最相似的节点
        
        Args:
            entity_text: 实体文本
            top_k: 返回最相似的 k 个节点
            similarity_threshold: 相似度阈值，低于此值的节点将被过滤
            
        Returns:
            最相似节点的列表，每个节点包含 node_id, name, similarity, labels
        """
        # 生成实体的 embedding
        entity_embedding = self.generate_embedding(entity_text)
        if entity_embedding is None:
            logger.warning(f"无法为实体 '{entity_text}' 生成 embedding")
            return []
        
        if not isinstance(entity_embedding, list) or len(entity_embedding) == 0:
            logger.warning(f"实体 '{entity_text}' 的 embedding 格式不正确")
            return []
        
        logger.info(f"为实体 '{entity_text}' 生成了维度为 {len(entity_embedding)} 的 embedding")
        
        try:
            with self.neo4j_driver.session() as session:
                # 使用 elementId() 替代已弃用的 id() 函数（Neo4j 5.x）
                # 查询所有具有 embedding 属性的节点
                query = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                RETURN elementId(n) as node_id, n.name as name, n.embedding as embedding, labels(n) as labels
                """
                result = session.run(query)
                
                similarities = []
                processed_count = 0
                
                for record in result:
                    node_id = record["node_id"]
                    node_name = record["name"]
                    node_embedding = record["embedding"]
                    node_labels = record["labels"]
                    
                    # 检查节点 embedding 是否有效
                    if node_embedding is None:
                        continue
                    
                    # 确保 embedding 是列表格式
                    if not isinstance(node_embedding, list):
                        logger.warning(f"节点 '{node_name}' 的 embedding 不是列表格式: {type(node_embedding)}")
                        continue
                    
                    # 检查 embedding 维度是否匹配
                    if len(node_embedding) != len(entity_embedding):
                        logger.debug(f"节点 '{node_name}' 的 embedding 维度不匹配: {len(node_embedding)} vs {len(entity_embedding)}")
                        continue
                    
                    processed_count += 1
                    
                    # 计算余弦相似度
                    similarity = self.cosine_similarity(entity_embedding, node_embedding)
                    
                    if similarity >= similarity_threshold:
                        similarities.append({
                            "node_id": node_id,
                            "name": node_name,
                            "similarity": similarity,
                            "labels": node_labels
                        })
                
                logger.info(f"处理了 {processed_count} 个节点，找到 {len(similarities)} 个相似节点（阈值 >= {similarity_threshold}）")
                
                # 按相似度降序排序，返回 top_k 个
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                return similarities[:top_k]
                
        except Exception as e:
            logger.error(f"查找相似节点失败: {e}", exc_info=True)
            return []
    
    def get_node_relations(self, node_id: str, relation_types: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取节点的关系和邻居节点信息
        
        Args:
            node_id: 节点 elementId（Neo4j 5.x 使用 elementId 替代 id）
            relation_types: 要查询的关系类型列表，None 表示查询所有关系
            
        Returns:
            包含关系和邻居节点信息的字典
        """
        try:
            with self.neo4j_driver.session() as session:
                if relation_types:
                    # 查询指定类型的关系，使用 elementId() 替代已弃用的 id()
                    query = """
                    MATCH (n)-[r]->(m)
                    WHERE elementId(n) = $node_id AND type(r) IN $relation_types
                    RETURN type(r) as relation_type, m.name as neighbor_name, labels(m) as neighbor_labels, elementId(m) as neighbor_id
                    ORDER BY type(r)
                    """
                    result = session.run(query, node_id=node_id, relation_types=relation_types)
                else:
                    # 查询所有关系，使用 elementId() 替代已弃用的 id()
                    query = """
                    MATCH (n)-[r]->(m)
                    WHERE elementId(n) = $node_id
                    RETURN type(r) as relation_type, m.name as neighbor_name, labels(m) as neighbor_labels, elementId(m) as neighbor_id
                    ORDER BY type(r)
                    """
                    result = session.run(query, node_id=node_id)
                
                relations = {}
                for record in result:
                    rel_type = record["relation_type"]
                    if rel_type not in relations:
                        relations[rel_type] = []
                    relations[rel_type].append({
                        "name": record["neighbor_name"],
                        "labels": record["neighbor_labels"],
                        "node_id": record["neighbor_id"]
                    })
                
                return relations
                
        except Exception as e:
            logger.error(f"获取节点关系失败 (node_id: {node_id}): {e}", exc_info=True)
            return {}
    
    def retrieve_knowledge(self, entities: List[Dict[str, str]], top_k: int = 5, 
                          similarity_threshold: float = 0.5, 
                          enable_rerank: bool = False,
                          query: str = "",
                          rerank_top_n: int = 5,
                          rerank_threshold: float = 0.0) -> Dict[str, Any]:
        """
        为多个实体检索知识图谱信息
        
        Args:
            entities: 实体列表，每个实体包含 text 和 label
            top_k: 每个实体返回的最相似节点数量
            similarity_threshold: 相似度阈值
            enable_rerank: 是否启用三元组rerank功能（默认False，保持向后兼容）
            query: 用户查询文本（用于rerank，仅在enable_rerank=True时使用）
            rerank_top_n: rerank后每个实体返回的最相关三元组数量（仅在enable_rerank=True时使用）
            rerank_threshold: rerank相似度阈值（仅在enable_rerank=True时使用）
            
        Returns:
            包含检索结果的字典
        """
        results = {
            "entities": [],
            "total_matched": 0
        }
        
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_label = entity.get("label", "")
            
            if not entity_text:
                continue
            
            # 查找相似节点
            similar_nodes = self.find_similar_nodes(
                entity_text, 
                top_k=top_k, 
                similarity_threshold=similarity_threshold
            )
            
            # 为每个匹配的节点获取关系信息
            enriched_nodes = []
            for node in similar_nodes:
                node_id = node["node_id"]
                relations = self.get_node_relations(node_id)
                node["relations"] = relations
                enriched_nodes.append(node)
            
            entity_result = {
                "entity_text": entity_text,
                "entity_label": entity_label,
                "matched_nodes": enriched_nodes
            }
            
            results["entities"].append(entity_result)
            if enriched_nodes:
                results["total_matched"] += 1
        
        # 如果启用rerank，对结果进行重排序
        if enable_rerank and query:
            try:
                from triple_reranker import get_reranker
                reranker = get_reranker()
                results = reranker.rerank_kg_results(
                    query=query,
                    kg_results=results,
                    top_n_per_entity=rerank_top_n,
                    similarity_threshold=rerank_threshold
                )
                logger.info("已对知识图谱检索结果进行rerank")
            except Exception as e:
                logger.error(f"rerank失败，返回原始结果: {e}", exc_info=True)
                # rerank失败时返回原始结果，不影响主流程
        
        return results


# 全局检索器实例（延迟加载）
_kg_retriever = None


def get_kg_retriever(neo4j_uri: str = None,
                     neo4j_user: str = None,
                     neo4j_password: str = None,
                     ollama_url: str = None) -> KnowledgeGraphRetriever:
    """获取知识图谱检索器实例（单例模式）"""
    global _kg_retriever
    if _kg_retriever is None:
        # 从配置文件读取参数
        try:
            import config as app_config
            neo4j_uri = neo4j_uri or app_config.NEO4J_URI
            neo4j_user = neo4j_user or app_config.NEO4J_USER
            neo4j_password = neo4j_password or app_config.NEO4J_PASSWORD
            ollama_url = ollama_url or app_config.OLLAMA_URL
        except ImportError:
            # 如果配置文件不存在，使用默认值
            neo4j_uri = neo4j_uri or "bolt://localhost:7687"
            neo4j_user = neo4j_user or "neo4j"
            neo4j_password = neo4j_password or "Crj123456"
            ollama_url = ollama_url or "http://localhost:11434"
        
        _kg_retriever = KnowledgeGraphRetriever(neo4j_uri, neo4j_user, neo4j_password, ollama_url)
    return _kg_retriever

