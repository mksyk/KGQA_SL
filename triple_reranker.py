"""
三元组重排序模块：基于embedding相似度对知识图谱中提取的三元组进行筛选和排序
参考KG-Rank项目的rerank机制，使用embedding模型计算查询与三元组的相似度
"""
import requests
import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripleReranker:
    """
    三元组重排序器：使用embedding模型对三元组进行相似度计算和排序
    
    参考KG-Rank项目中的rerank机制，但使用embedding模型（如bge-m3）替代Cohere API
    embedding模型部分可以使用embedEntity.py中的模型，便于修改
    """
    
    def __init__(self, ollama_url: str, model_name: str = "bge-m3"):
        """
        初始化三元组重排序器
        
        Args:
            ollama_url: Ollama服务地址（用于生成embedding）
            model_name: embedding模型名称，默认使用bge-m3
                       可以根据需要修改为其他模型，参考embedEntity.py中的配置
        """
        self.ollama_url = ollama_url
        self.model_name = model_name
        logger.info(f"三元组重排序器初始化完成，使用模型: {model_name}")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        使用Ollama生成embedding向量
        
        参考embedEntity.py中的generate_embeddings方法
        可以根据需要修改模型配置（如GPU选择等）
        
        Args:
            text: 要生成embedding的文本
            
        Returns:
            embedding向量列表，失败返回None
        """
        payload = {
            "model": self.model_name,
            "prompt": text
            # 如果需要指定GPU，可以添加options参数：
            # "options": {
            #     "main_gpu": 3  # 在这里选择GPU，参考embedEntity.py中的配置
            # }
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
                logger.debug(f"成功为文本生成embedding，维度: {len(embedding) if isinstance(embedding, list) else 'unknown'}")
                return embedding
            else:
                logger.warning(f"文本未返回embedding")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"生成embedding失败: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算两个向量的余弦相似度
        
        Args:
            vec1: 向量1
            vec2: 向量2
            
        Returns:
            余弦相似度值（范围：-1到1）
        """
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def format_triple_as_text(self, head_entity: str, relation: str, tail_entity: str) -> str:
        """
        将三元组格式化为文本表示
        
        Args:
            head_entity: 头实体名称
            relation: 关系类型
            tail_entity: 尾实体名称
            
        Returns:
            格式化后的三元组文本
        """
        # 格式：头实体 关系 尾实体
        return f"{head_entity} {relation} {tail_entity}"
    
    def extract_triples_from_relations(self, node_name: str, relations: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        从节点关系信息中提取三元组列表
        
        Args:
            node_name: 节点名称（作为头实体）
            relations: 关系字典，格式为 {relation_type: [neighbor_info, ...]}
            
        Returns:
            三元组列表，每个三元组包含 head_entity, relation, tail_entity, text
        """
        triples = []
        
        for relation_type, neighbors in relations.items():
            for neighbor in neighbors:
                tail_entity = neighbor.get("name", "")
                if tail_entity:
                    triple_text = self.format_triple_as_text(node_name, relation_type, tail_entity)
                    triples.append({
                        "head_entity": node_name,
                        "relation": relation_type,
                        "tail_entity": tail_entity,
                        "text": triple_text,
                        "neighbor_info": neighbor  # 保留原始邻居信息
                    })
        
        return triples
    
    def rerank_triples(self, query: str, triples: List[Dict[str, Any]], 
                      top_n: int = 10, similarity_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        对三元组进行重排序，基于查询与三元组的embedding相似度
        
        参考KG-Rank项目中CohereReranker的rerank方法
        但使用embedding模型计算相似度，而不是调用Cohere API
        
        Args:
            query: 用户查询文本
            triples: 三元组列表
            top_n: 返回top-n个最相关的三元组
            similarity_threshold: 相似度阈值，低于此值的三元组将被过滤
            
        Returns:
            重排序后的三元组列表，每个三元组包含原始信息加上relevance_score
        """
        if not triples:
            return []
        
        # 生成查询的embedding
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            logger.warning("无法为查询生成embedding，返回原始三元组列表")
            return triples[:top_n]
        
        # 为每个三元组生成embedding并计算相似度
        scored_triples = []
        for triple in triples:
            triple_text = triple.get("text", "")
            if not triple_text:
                continue
            
            # 生成三元组的embedding
            triple_embedding = self.generate_embedding(triple_text)
            if triple_embedding is None:
                continue
            
            # 检查embedding维度是否匹配
            if len(triple_embedding) != len(query_embedding):
                logger.debug(f"三元组embedding维度不匹配: {len(triple_embedding)} vs {len(query_embedding)}")
                continue
            
            # 计算余弦相似度
            similarity = self.cosine_similarity(query_embedding, triple_embedding)
            
            if similarity >= similarity_threshold:
                scored_triples.append({
                    **triple,  # 保留原始三元组信息
                    "relevance_score": similarity
                })
        
        # 按相似度降序排序
        scored_triples.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # 返回top-n个
        reranked_results = scored_triples[:top_n]
        
        logger.info(f"对 {len(triples)} 个三元组进行rerank，返回 {len(reranked_results)} 个结果（阈值 >= {similarity_threshold}）")
        
        return reranked_results
    
    def rerank_kg_results(self, query: str, kg_results: Dict[str, Any], 
                         top_n_per_entity: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        对知识图谱检索结果中的三元组进行重排序
        
        这是主要的接口方法，用于处理kg_retriever返回的完整结果
        
        Args:
            query: 用户查询文本
            kg_results: 知识图谱检索结果，格式与kg_retriever.retrieve_knowledge返回的格式一致
            top_n_per_entity: 每个实体返回的最相关三元组数量
            similarity_threshold: 相似度阈值
            
        Returns:
            重排序后的知识图谱结果，保持原有结构但三元组已按相关性排序
        """
        if not kg_results or not kg_results.get("entities"):
            return kg_results
        
        reranked_results = {
            "entities": [],
            "total_matched": kg_results.get("total_matched", 0)
        }
        
        for entity_result in kg_results["entities"]:
            entity_text = entity_result.get("entity_text", "")
            entity_label = entity_result.get("entity_label", "")
            matched_nodes = entity_result.get("matched_nodes", [])
            
            reranked_entity_result = {
                "entity_text": entity_text,
                "entity_label": entity_label,
                "matched_nodes": []
            }
            
            # 对每个匹配节点的关系进行rerank
            for node in matched_nodes:
                node_name = node.get("name", "")
                relations = node.get("relations", {})
                
                if not relations:
                    # 如果没有关系，保留节点但不添加reranked_relations
                    reranked_node = {**node}
                    reranked_entity_result["matched_nodes"].append(reranked_node)
                    continue
                
                # 提取三元组
                triples = self.extract_triples_from_relations(node_name, relations)
                
                if not triples:
                    reranked_node = {**node}
                    reranked_entity_result["matched_nodes"].append(reranked_node)
                    continue
                
                # 对三元组进行rerank
                reranked_triples = self.rerank_triples(
                    query, 
                    triples, 
                    top_n=top_n_per_entity,
                    similarity_threshold=similarity_threshold
                )
                
                # 重构关系字典（只保留rerank后的三元组）
                reranked_relations = defaultdict(list)
                for triple in reranked_triples:
                    relation_type = triple["relation"]
                    neighbor_info = triple["neighbor_info"]
                    # 添加相似度分数到邻居信息中
                    neighbor_info["relevance_score"] = triple["relevance_score"]
                    reranked_relations[relation_type].append(neighbor_info)
                
                # 创建rerank后的节点信息
                reranked_node = {
                    **node,
                    "relations": dict(reranked_relations),
                    "reranked_triples": reranked_triples  # 可选：保留详细的三元组信息
                }
                
                reranked_entity_result["matched_nodes"].append(reranked_node)
            
            reranked_results["entities"].append(reranked_entity_result)
        
        return reranked_results


# 全局reranker实例（延迟加载）
_reranker = None


def get_reranker(ollama_url: str = None, model_name: str = None) -> TripleReranker:
    """
    获取三元组重排序器实例（单例模式）
    
    Args:
        ollama_url: Ollama服务地址，如果为None则从配置文件读取
        model_name: embedding模型名称，如果为None则使用默认值
        
    Returns:
        TripleReranker实例
    """
    global _reranker
    if _reranker is None:
        # 从配置文件读取参数
        try:
            import config as app_config
            ollama_url = ollama_url or app_config.OLLAMA_URL
        except ImportError:
            # 如果配置文件不存在，使用默认值
            ollama_url = ollama_url or "http://localhost:11434"
        
        model_name = model_name or "bge-m3"
        _reranker = TripleReranker(ollama_url, model_name)
    
    return _reranker

