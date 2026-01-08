"""
G-Retriever主模块：从知识图谱中检索相关节点和边，使用PCST算法构建子图
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np
from neo4j import GraphDatabase

from .edge_embedding import EdgeEmbeddingManager
from .pcst_algorithm import PCSTSubgraphBuilder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GRetriever:
    """G-Retriever主类：负责从知识图谱中检索并构建子图"""
    
    def __init__(
        self,
        neo4j_driver,
        ollama_url: str,
        topk_nodes: int = 3,
        topk_edges: int = 5,
        cost_edge: float = 0.5,
        similarity_threshold: float = 0.5
    ):
        """
        初始化G-Retriever
        
        Args:
            neo4j_driver: Neo4j驱动实例
            ollama_url: Ollama服务地址（用于生成embedding）
            topk_nodes: PCST算法选择的top-k节点数量
            topk_edges: PCST算法选择的top-k边数量
            cost_edge: PCST算法中边的成本参数
            similarity_threshold: 节点相似度阈值
        """
        self.neo4j_driver = neo4j_driver
        self.ollama_url = ollama_url
        self.similarity_threshold = similarity_threshold
        
        # 初始化边embedding管理器
        self.edge_embedding_manager = EdgeEmbeddingManager(neo4j_driver, ollama_url)
        
        # 初始化PCST子图构建器
        self.pcst_builder = PCSTSubgraphBuilder(
            topk_nodes=topk_nodes,
            topk_edges=topk_edges,
            cost_edge=cost_edge
        )
        
        logger.info("G-Retriever初始化完成")
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """
        使用 Ollama 的 bge-m3 模型生成 embedding
        
        Args:
            text: 要生成 embedding 的文本
            
        Returns:
            embedding 向量列表，失败返回 None
        """
        import requests
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
                return embedding
            else:
                logger.warning(f"文本 '{text}' 未返回 embedding")
                return None
                
        except Exception as e:
            logger.error(f"生成文本 '{text}' 的 embedding 失败: {e}")
            return None
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """计算两个向量的余弦相似度"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))
    
    def retrieve_initial_nodes_and_edges(
        self,
        query: str,
        top_k: int = 10,
        max_hop: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        根据query检索初始的节点和边
        
        Args:
            query: 查询文本
            top_k: 每个实体返回的最相似节点数量
            max_hop: 最大跳数（从检索到的节点开始扩展）
            
        Returns:
            (nodes, edges): 节点列表和边列表
        """
        # 生成查询的embedding
        logger.info(f"开始为查询生成embedding: {query[:50]}...")
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            logger.error(f"无法为查询 '{query}' 生成embedding，请检查Ollama服务是否正常运行")
            return [], []
        logger.info(f"查询embedding生成成功，维度: {len(query_embedding)}")
        
        nodes = []
        node_id_to_index = {}  # Neo4j node elementId到索引的映射
        edges = []
        edge_set = set()  # 用于去重 (src_id, dst_id, rel_type)
        
        try:
            with self.neo4j_driver.session() as session:
                # 步骤1: 检索与query相似的节点
                logger.info(f"开始检索节点，相似度阈值: {self.similarity_threshold}, top_k: {top_k}")
                query_similarity = """
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH n, elementId(n) as node_id, n.embedding as embedding
                RETURN node_id, n.name as name, embedding, labels(n) as labels
                LIMIT 1000
                """
                result = session.run(query_similarity)
                
                total_nodes_checked = 0
                nodes_with_valid_embedding = 0
                similarities = []
                max_similarity = 0.0
                
                for record in result:
                    total_nodes_checked += 1
                    node_id = record["node_id"]
                    node_name = record["name"]
                    node_embedding = record["embedding"]
                    node_labels = record["labels"]
                    
                    if not isinstance(node_embedding, list):
                        continue
                    if len(node_embedding) != len(query_embedding):
                        continue
                    
                    nodes_with_valid_embedding += 1
                    similarity = self.cosine_similarity(query_embedding, node_embedding)
                    max_similarity = max(max_similarity, similarity)
                    
                    if similarity >= self.similarity_threshold:
                        similarities.append({
                            "node_id": node_id,
                            "name": node_name,
                            "similarity": similarity,
                            "labels": node_labels,
                            "embedding": node_embedding
                        })
                
                logger.info(f"检查了 {total_nodes_checked} 个节点，其中 {nodes_with_valid_embedding} 个有有效embedding，最高相似度: {max_similarity:.4f}")
                
                # 按相似度排序并取top-k
                similarities.sort(key=lambda x: x["similarity"], reverse=True)
                top_nodes = similarities[:top_k]
                
                if len(top_nodes) == 0:
                    logger.warning(f"没有节点通过相似度阈值 {self.similarity_threshold}，最高相似度为 {max_similarity:.4f}。建议降低相似度阈值或检查查询与知识图谱的匹配度。")
                    # 如果没有任何节点通过阈值，但有一些相似节点，至少返回最高相似度的节点
                    if similarities:
                        logger.info(f"返回最高相似度的 {min(top_k, len(similarities))} 个节点（即使低于阈值）")
                        top_nodes = similarities[:top_k]
                
                logger.info(f"检索到 {len(top_nodes)} 个相关节点")
                
                # 步骤2: 从这些节点开始，扩展到邻居节点（max_hop跳）
                explored_node_ids = set()
                nodes_to_explore = [(node["node_id"], 0) for node in top_nodes]  # (node_id, hop)
                
                while nodes_to_explore:
                    current_node_id, current_hop = nodes_to_explore.pop(0)
                    
                    if current_node_id in explored_node_ids:
                        continue
                    if current_hop > max_hop:
                        continue
                    
                    explored_node_ids.add(current_node_id)
                    
                    # 获取节点信息
                    node_query = """
                    MATCH (n)
                    WHERE elementId(n) = $node_id
                    RETURN elementId(n) as node_id, n.name as name, labels(n) as labels, n.embedding as embedding
                    """
                    node_result = session.run(node_query, node_id=current_node_id)
                    node_record = node_result.single()
                    
                    if node_record:
                        node_id = node_record["node_id"]
                        if node_id not in node_id_to_index:
                            node_index = len(nodes)
                            node_id_to_index[node_id] = node_index
                            nodes.append({
                                "node_id": node_id,
                                "name": node_record["name"],
                                "labels": node_record["labels"],
                                "embedding": node_record["embedding"],
                                "index": node_index
                            })
                        
                        # 获取节点的出边和入边
                        if current_hop < max_hop:
                            edge_query = """
                            MATCH (n)-[r]->(m)
                            WHERE elementId(n) = $node_id
                            RETURN type(r) as rel_type, elementId(m) as dst_id, m.name as dst_name, labels(m) as dst_labels, m.embedding as dst_embedding
                            """
                            edge_result = session.run(edge_query, node_id=current_node_id)
                            
                            for edge_record in edge_result:
                                rel_type = edge_record["rel_type"]
                                dst_id = edge_record["dst_id"]
                                
                                # 添加目标节点
                                if dst_id not in node_id_to_index:
                                    dst_index = len(nodes)
                                    node_id_to_index[dst_id] = dst_index
                                    dst_embedding = edge_record["dst_embedding"]
                                    nodes.append({
                                        "node_id": dst_id,
                                        "name": edge_record["dst_name"],
                                        "labels": edge_record["dst_labels"],
                                        "embedding": dst_embedding if dst_embedding else None,
                                        "index": dst_index
                                    })
                                
                                # 添加边（去重）
                                edge_key = (current_node_id, dst_id, rel_type)
                                if edge_key not in edge_set:
                                    edge_set.add(edge_key)
                                    src_index = node_id_to_index[current_node_id]
                                    dst_index = node_id_to_index[dst_id]
                                    edges.append({
                                        "src_id": current_node_id,
                                        "dst_id": dst_id,
                                        "src_index": src_index,
                                        "dst_index": dst_index,
                                        "rel_type": rel_type,
                                        "edge_index": len(edges)
                                    })
                                
                                # 将目标节点加入探索队列
                                if dst_id not in explored_node_ids:
                                    nodes_to_explore.append((dst_id, current_hop + 1))
                
                logger.info(f"扩展后得到 {len(nodes)} 个节点，{len(edges)} 条边")
                
        except Exception as e:
            logger.error(f"检索节点和边失败: {e}", exc_info=True)
            return [], []
        
        return nodes, edges
    
    def build_subgraph(self, query: str, top_k: int = 10, max_hop: int = 2) -> Dict[str, Any]:
        """
        根据query构建子图
        
        Args:
            query: 查询文本
            top_k: 初始检索的节点数量
            max_hop: 最大跳数
            
        Returns:
            包含子图信息的字典
        """
        # 步骤1: 确保边embedding已初始化
        if not self.edge_embedding_manager._edge_type_cache:
            logger.info("初始化边embedding...")
            self.edge_embedding_manager.initialize_edge_embeddings()
        
        # 步骤2: 检索初始节点和边
        nodes, edges = self.retrieve_initial_nodes_and_edges(query, top_k=top_k, max_hop=max_hop)
        
        if not nodes or not edges:
            logger.warning("未检索到节点或边，返回空子图")
            return {
                "nodes": [],
                "edges": [],
                "query_embedding": None,
                "subgraph_info": {
                    "num_nodes": 0,
                    "num_edges": 0
                }
            }
        
        # 步骤3: 生成查询embedding
        query_embedding = self.generate_embedding(query)
        if query_embedding is None:
            logger.warning("无法生成查询embedding，返回原始图")
            return {
                "nodes": nodes,
                "edges": edges,
                "query_embedding": None,
                "subgraph_info": {
                    "num_nodes": len(nodes),
                    "num_edges": len(edges)
                }
            }
        
        # 步骤4: 构建节点和边的embedding矩阵
        # 过滤掉没有embedding的节点
        valid_nodes = []
        valid_node_indices = []
        for i, node in enumerate(nodes):
            if node.get("embedding") is not None:
                valid_nodes.append(node)
                valid_node_indices.append(i)
        
        if not valid_nodes:
            logger.warning("没有有效embedding的节点")
            return {
                "nodes": nodes,
                "edges": edges,
                "query_embedding": query_embedding,
                "subgraph_info": {
                    "num_nodes": len(nodes),
                    "num_edges": len(edges)
                }
            }
        
        # 构建节点embedding矩阵
        node_embeddings = torch.tensor([node["embedding"] for node in valid_nodes], dtype=torch.float32)
        
        # 构建边embedding矩阵
        edge_embeddings_list = []
        valid_edges = []
        edge_type_to_embedding = self.edge_embedding_manager.get_edge_embeddings_batch(
            list(set(edge["rel_type"] for edge in edges))
        )
        
        # 创建索引映射：原节点索引到有效节点索引
        old_to_new_node_index = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_node_indices)}
        valid_node_set = set(valid_node_indices)
        
        for edge in edges:
            src_idx = edge["src_index"]
            dst_idx = edge["dst_index"]
            
            # 只保留两个端点都在有效节点中的边
            if src_idx in valid_node_set and dst_idx in valid_node_set:
                rel_type = edge["rel_type"]
                edge_emb = edge_type_to_embedding.get(rel_type)
                
                if edge_emb:
                    # 更新边的索引为有效节点索引
                    edge["src_index"] = old_to_new_node_index[src_idx]
                    edge["dst_index"] = old_to_new_node_index[dst_idx]
                    valid_edges.append(edge)
                    edge_embeddings_list.append(edge_emb)
        
        if not valid_edges:
            logger.warning("没有有效的边")
            return {
                "nodes": valid_nodes,
                "edges": [],
                "query_embedding": query_embedding,
                "subgraph_info": {
                    "num_nodes": len(valid_nodes),
                    "num_edges": 0
                }
            }
        
        edge_embeddings = torch.tensor(edge_embeddings_list, dtype=torch.float32)
        query_emb_tensor = torch.tensor(query_embedding, dtype=torch.float32)
        
        # 构建边索引矩阵
        edge_index = torch.tensor(
            [[edge["src_index"] for edge in valid_edges],
             [edge["dst_index"] for edge in valid_edges]],
            dtype=torch.long
        )
        
        # 步骤5: 使用PCST算法构建子图
        node_ids = [node["node_id"] for node in valid_nodes]
        selected_node_indices, selected_edge_indices, node_mapping = self.pcst_builder.build_subgraph(
            node_embeddings=node_embeddings,
            edge_embeddings=edge_embeddings,
            edge_index=edge_index,
            query_embedding=query_emb_tensor,
            node_ids=node_ids,
            edge_info=valid_edges
        )
        
        # 步骤6: 提取选中的节点和边
        selected_nodes = [valid_nodes[i] for i in selected_node_indices]
        selected_edges = [valid_edges[i] for i in selected_edge_indices]
        
        # 更新选中节点的索引（使用新的映射，确保与边的索引一致）
        for node in selected_nodes:
            old_index = node.get("index")
            if old_index is not None:
                # 使用node_mapping将节点的index更新为新的索引
                new_index = node_mapping.get(old_index)
                if new_index is not None:
                    node["index"] = new_index
        
        # 更新选中边的索引（使用新的映射）
        for edge in selected_edges:
            old_src = edge["src_index"]
            old_dst = edge["dst_index"]
            edge["src_index"] = node_mapping.get(old_src, old_src)
            edge["dst_index"] = node_mapping.get(old_dst, old_dst)
        
        logger.info(f"PCST子图构建完成: {len(selected_nodes)} 个节点, {len(selected_edges)} 条边")
        
        return {
            "nodes": selected_nodes,
            "edges": selected_edges,
            "query_embedding": query_embedding,
            "subgraph_info": {
                "num_nodes": len(selected_nodes),
                "num_edges": len(selected_edges)
            }
        }

