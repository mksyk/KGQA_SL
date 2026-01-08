"""
PCST (Prize-Collecting Steiner Tree) 算法实现
用于从知识图谱中构建最优子图
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
import torch

try:
    from pcst_fast import pcst_fast
    PCST_AVAILABLE = True
except ImportError:
    PCST_AVAILABLE = False
    logging.warning("pcst_fast未安装，PCST算法将不可用。请运行: pip install pcst_fast")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PCSTSubgraphBuilder:
    """基于PCST算法的子图构建器"""
    
    def __init__(self, topk_nodes: int = 3, topk_edges: int = 5, cost_edge: float = 0.5, decay_factor: float = 0.01):
        """
        初始化PCST子图构建器
        
        Args:
            topk_nodes: 选择的top-k节点数量
            topk_edges: 选择的top-k边数量
            cost_edge: 边的成本参数
            decay_factor: 边的prize衰减因子
        """
        if not PCST_AVAILABLE:
            raise ImportError("pcst_fast未安装，无法使用PCST算法。请运行: pip install pcst_fast")
        
        self.topk_nodes = topk_nodes
        self.topk_edges = topk_edges
        self.cost_edge = cost_edge
        self.decay_factor = decay_factor
        logger.info(f"PCST子图构建器初始化完成 (topk_nodes={topk_nodes}, topk_edges={topk_edges}, cost_edge={cost_edge})")
    
    def build_subgraph(
        self,
        node_embeddings: torch.Tensor,
        edge_embeddings: torch.Tensor,
        edge_index: torch.Tensor,
        query_embedding: torch.Tensor,
        node_ids: List[Any],
        edge_info: List[Dict[str, Any]]
    ) -> Tuple[List[int], List[int], Dict[int, int]]:
        """
        使用PCST算法构建子图
        
        Args:
            node_embeddings: 节点embedding矩阵 (num_nodes, dim)
            edge_embeddings: 边embedding矩阵 (num_edges, dim)
            edge_index: 边索引 (2, num_edges)，第一行为源节点，第二行为目标节点
            query_embedding: 查询的embedding向量 (dim,)
            node_ids: 节点ID列表，与node_embeddings的索引对应
            edge_info: 边信息列表，每个元素包含边的相关信息，与edge_embeddings的索引对应
        
        Returns:
            (selected_node_indices, selected_edge_indices, node_mapping)
            selected_node_indices: 被选中的节点在原node_ids中的索引列表
            selected_edge_indices: 被选中的边在原edge_info中的索引列表
            node_mapping: 从原节点索引到新节点索引的映射
        """
        if node_embeddings.size(0) == 0 or edge_embeddings.size(0) == 0:
            logger.warning("节点或边为空，返回空子图")
            return [], [], {}
        
        num_nodes = node_embeddings.size(0)
        num_edges = edge_embeddings.size(0)
        
        # 确保query_embedding是1D向量
        if query_embedding.dim() > 1:
            query_embedding = query_embedding.squeeze()
        
        # 计算节点prizes
        if self.topk_nodes > 0:
            # 计算节点与查询的余弦相似度
            node_similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                node_embeddings,
                dim=-1
            )
            
            topk = min(self.topk_nodes, num_nodes)
            _, topk_node_indices = torch.topk(node_similarities, topk, largest=True)
            
            # 设置prize：top-k节点按排名递减
            node_prizes = torch.zeros_like(node_similarities)
            node_prizes[topk_node_indices] = torch.arange(topk, 0, -1, dtype=torch.float)
        else:
            node_prizes = torch.zeros(num_nodes)
        
        # 计算边prizes
        if self.topk_edges > 0:
            # 计算边与查询的余弦相似度
            edge_similarities = torch.nn.functional.cosine_similarity(
                query_embedding.unsqueeze(0),
                edge_embeddings,
                dim=-1
            )
            
            unique_similarities = edge_similarities.unique()
            topk_e = min(self.topk_edges, unique_similarities.size(0))
            
            if topk_e > 0:
                topk_e_values, _ = torch.topk(unique_similarities, topk_e, largest=True)
                edge_prizes = torch.zeros_like(edge_similarities)
                
                # 只保留top-k相似度值对应的边
                edge_prizes[edge_similarities < topk_e_values[-1]] = 0.0
                
                # 为每个相似度等级分配prize值
                last_topk_e_value = self.topk_edges
                for k in range(topk_e):
                    indices = edge_similarities == topk_e_values[k]
                    count = indices.sum().item()
                    if count > 0:
                        value = min((topk_e - k) / count, last_topk_e_value)
                        edge_prizes[indices] = value
                        last_topk_e_value = value * (1 - self.decay_factor)
                
                # 调整边的成本，确保至少有一条边被选中
                max_edge_prize = edge_prizes.max().item()
                if max_edge_prize > 0:
                    self.cost_edge = min(self.cost_edge, max_edge_prize * (1 - self.decay_factor / 2))
            else:
                edge_prizes = torch.zeros(num_edges)
        else:
            edge_prizes = torch.zeros(num_edges)
        
        # 构建PCST输入
        costs = []
        edges = []
        virtual_node_prizes = []
        virtual_edges = []
        virtual_costs = []
        mapping_edge = {}  # PCST图中的边索引到原图边索引的映射
        mapping_virtual_node = {}  # 虚拟节点到原图边索引的映射
        
        edge_index_np = edge_index.cpu().numpy()
        edge_prizes_np = edge_prizes.cpu().numpy()
        
        for i in range(num_edges):
            src = int(edge_index_np[0, i])
            dst = int(edge_index_np[1, i])
            prize_e = edge_prizes_np[i]
            
            if prize_e <= self.cost_edge:
                # 边prize小于等于成本，直接作为普通边处理
                mapping_edge[len(edges)] = i
                edges.append((src, dst))
                costs.append(self.cost_edge - prize_e)
            else:
                # 边prize大于成本，创建虚拟节点
                virtual_node_id = num_nodes + len(virtual_node_prizes)
                mapping_virtual_node[virtual_node_id] = i
                virtual_edges.append((src, virtual_node_id))
                virtual_edges.append((virtual_node_id, dst))
                virtual_costs.append(0.0)
                virtual_costs.append(0.0)
                virtual_node_prizes.append(prize_e - self.cost_edge)
        
        # 合并所有节点prize
        all_node_prizes = torch.cat([node_prizes, torch.tensor(virtual_node_prizes, dtype=torch.float)]).numpy()
        
        # 合并所有边和成本
        num_regular_edges = len(edges)
        if len(virtual_edges) > 0:
            all_edges = edges + virtual_edges
            all_costs = costs + virtual_costs
        else:
            all_edges = edges
            all_costs = costs
        
        # 转换为numpy数组
        all_edges = np.array(all_edges, dtype=np.int32)
        all_costs = np.array(all_costs, dtype=np.float64)
        
        # 调用PCST算法
        root = -1  # unrooted
        num_clusters = 1
        pruning = 'gw'
        verbosity_level = 0
        
        try:
            selected_vertices, selected_pcst_edges = pcst_fast(
                all_edges,
                all_node_prizes,
                all_costs,
                root,
                num_clusters,
                pruning,
                verbosity_level
            )
        except Exception as e:
            logger.error(f"PCST算法执行失败: {e}", exc_info=True)
            # 失败时返回空结果
            return [], [], {}
        
        # 处理PCST结果
        # 分离真实节点和虚拟节点
        real_node_indices = [v for v in selected_vertices if v < num_nodes]
        
        # 处理选中的边
        selected_edge_indices = []
        for e_idx in selected_pcst_edges:
            if e_idx < num_regular_edges:
                # 普通边
                if e_idx in mapping_edge:
                    selected_edge_indices.append(mapping_edge[e_idx])
            else:
                # 虚拟边，通过虚拟节点找到原始边
                # 这里需要找到与虚拟节点相关的边
                pass
        
        # 处理虚拟节点对应的边
        virtual_vertex_indices = [v for v in selected_vertices if v >= num_nodes]
        for virtual_vertex_idx in virtual_vertex_indices:
            if virtual_vertex_idx in mapping_virtual_node:
                edge_idx = mapping_virtual_node[virtual_vertex_idx]
                if edge_idx not in selected_edge_indices:
                    selected_edge_indices.append(edge_idx)
        
        # 确保选中的节点包含所有边的端点
        selected_node_set = set(real_node_indices)
        edge_index_np = edge_index.cpu().numpy()
        for edge_idx in selected_edge_indices:
            src = int(edge_index_np[0, edge_idx])
            dst = int(edge_index_np[1, edge_idx])
            selected_node_set.add(src)
            selected_node_set.add(dst)
        
        selected_node_indices = sorted(list(selected_node_set))
        
        # 创建节点映射（从原索引到新索引）
        node_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_node_indices)}
        
        logger.info(f"PCST构建子图完成: {len(selected_node_indices)} 个节点, {len(selected_edge_indices)} 条边")
        
        return selected_node_indices, selected_edge_indices, node_mapping

