import torch
import torch.nn.functional as F

class Propagation:

    @staticmethod
    def aggregate_k_hop_features(adj: torch.Tensor, x: torch.Tensor, k: int):
        # adj 是图的邻接矩阵（N x N），x 是节点的特征矩阵（N x F），k 是跳数
        # N 是节点数，F 是特征数
        
        # 初始化节点的聚合特征为原始特征
        aggregated_features = x

        # 对邻接矩阵进行归一化，防止特征数目随着传播而变大
        degree = adj.sum(dim=1, keepdim=True)  # 计算度矩阵
        adj_normalized = adj / degree  # 归一化邻接矩阵
        
        # 通过消息传递机制聚合 k 跳特征
        for _ in range(k):
            
            # 消息传递：邻接矩阵乘以当前的聚合特征
            aggregated_features = torch.matmul(adj_normalized, aggregated_features)
            
            # 非线性激活（可以根据需要添加）
            aggregated_features = F.relu(aggregated_features)
        
        return aggregated_features