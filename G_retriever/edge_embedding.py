"""
边Embedding管理模块
由于边类型数量较少（约10种），采用高效存储方式：
- 每个边类型生成一个embedding并存储在Neo4j中
- 使用字典缓存以提高访问效率
"""
import logging
from typing import Dict, List, Optional, Any
import requests
from neo4j import GraphDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EdgeEmbeddingManager:
    """边类型Embedding管理器"""
    
    def __init__(self, neo4j_driver, ollama_url: str):
        """
        初始化边Embedding管理器
        
        Args:
            neo4j_driver: Neo4j驱动实例
            ollama_url: Ollama服务地址（用于生成embedding）
        """
        self.neo4j_driver = neo4j_driver
        self.ollama_url = ollama_url
        self._edge_type_cache: Dict[str, List[float]] = {}  # 边类型到embedding的缓存
        logger.info("边Embedding管理器初始化完成")
    
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
    
    def get_all_edge_types(self) -> List[str]:
        """
        从Neo4j中获取所有边类型
        
        Returns:
            边类型列表
        """
        try:
            with self.neo4j_driver.session() as session:
                query = """
                CALL db.relationshipTypes() YIELD relationshipType
                RETURN relationshipType
                ORDER BY relationshipType
                """
                result = session.run(query)
                edge_types = [record["relationshipType"] for record in result]
                logger.info(f"从Neo4j中获取到 {len(edge_types)} 种边类型: {edge_types}")
                return edge_types
        except Exception as e:
            logger.error(f"获取边类型失败: {e}", exc_info=True)
            return []
    
    def initialize_edge_embeddings(self) -> bool:
        """
        初始化所有边类型的embedding并存储到Neo4j
        
        Returns:
            是否成功初始化
        """
        edge_types = self.get_all_edge_types()
        if not edge_types:
            logger.warning("未找到任何边类型，无法初始化edge embedding")
            return False
        
        success_count = 0
        for edge_type in edge_types:
            # 检查是否已存在embedding
            existing_emb = self.get_edge_embedding(edge_type, from_db=True)
            if existing_emb:
                logger.debug(f"边类型 '{edge_type}' 的embedding已存在，跳过")
                self._edge_type_cache[edge_type] = existing_emb
                success_count += 1
                continue
            
            # 生成新的embedding
            embedding = self.generate_embedding(edge_type)
            if embedding:
                # 存储到Neo4j（使用虚拟节点存储）
                if self._store_edge_embedding(edge_type, embedding):
                    self._edge_type_cache[edge_type] = embedding
                    success_count += 1
                    logger.info(f"成功初始化边类型 '{edge_type}' 的embedding")
                else:
                    logger.warning(f"存储边类型 '{edge_type}' 的embedding失败")
            else:
                logger.warning(f"生成边类型 '{edge_type}' 的embedding失败")
        
        logger.info(f"边embedding初始化完成: {success_count}/{len(edge_types)}")
        return success_count > 0
    
    def _store_edge_embedding(self, edge_type: str, embedding: List[float]) -> bool:
        """
        将边类型的embedding存储到Neo4j
        使用虚拟节点存储，节点标签为 'EdgeType'，属性包含边类型名称和embedding
        
        Args:
            edge_type: 边类型名称
            embedding: embedding向量
            
        Returns:
            是否成功存储
        """
        try:
            with self.neo4j_driver.session() as session:
                # 使用MERGE确保唯一性，如果已存在则更新
                query = """
                MERGE (et:EdgeType {name: $edge_type})
                ON CREATE SET et.embedding = $embedding, et.created_at = datetime()
                ON MATCH SET et.embedding = $embedding, et.updated_at = datetime()
                RETURN elementId(et) as id
                """
                result = session.run(query, edge_type=edge_type, embedding=embedding)
                record = result.single()
                if record:
                    logger.debug(f"成功存储边类型 '{edge_type}' 的embedding")
                    return True
                else:
                    logger.warning(f"存储边类型 '{edge_type}' 的embedding时未返回结果")
                    return False
        except Exception as e:
            logger.error(f"存储边类型 '{edge_type}' 的embedding失败: {e}", exc_info=True)
            return False
    
    def get_edge_embedding(self, edge_type: str, from_db: bool = False) -> Optional[List[float]]:
        """
        获取边类型的embedding
        
        Args:
            edge_type: 边类型名称
            from_db: 是否从数据库读取（True时跳过缓存）
            
        Returns:
            embedding向量，失败返回None
        """
        # 先检查缓存
        if not from_db and edge_type in self._edge_type_cache:
            return self._edge_type_cache[edge_type]
        
        # 从数据库读取
        try:
            with self.neo4j_driver.session() as session:
                query = """
                MATCH (et:EdgeType {name: $edge_type})
                RETURN et.embedding as embedding
                """
                result = session.run(query, edge_type=edge_type)
                record = result.single()
                if record and record["embedding"]:
                    embedding = record["embedding"]
                    # 更新缓存
                    self._edge_type_cache[edge_type] = embedding
                    return embedding
                else:
                    logger.debug(f"边类型 '{edge_type}' 的embedding不存在于数据库中")
                    return None
        except Exception as e:
            logger.error(f"获取边类型 '{edge_type}' 的embedding失败: {e}", exc_info=True)
            return None
    
    def update_edge_embedding(self, edge_type: str) -> bool:
        """
        更新指定边类型的embedding
        
        Args:
            edge_type: 边类型名称
            
        Returns:
            是否成功更新
        """
        embedding = self.generate_embedding(edge_type)
        if embedding:
            if self._store_edge_embedding(edge_type, embedding):
                self._edge_type_cache[edge_type] = embedding
                logger.info(f"成功更新边类型 '{edge_type}' 的embedding")
                return True
            else:
                logger.warning(f"更新边类型 '{edge_type}' 的embedding失败")
                return False
        else:
            logger.warning(f"生成边类型 '{edge_type}' 的embedding失败")
            return False
    
    def get_edge_embeddings_batch(self, edge_types: List[str]) -> Dict[str, List[float]]:
        """
        批量获取边类型的embedding
        
        Args:
            edge_types: 边类型列表
            
        Returns:
            边类型到embedding的字典
        """
        result = {}
        for edge_type in edge_types:
            embedding = self.get_edge_embedding(edge_type)
            if embedding:
                result[edge_type] = embedding
        return result

