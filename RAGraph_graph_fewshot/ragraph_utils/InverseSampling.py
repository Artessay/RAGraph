import torch

class InverseSampling:
    
    @staticmethod
    def compute_sample_prob(adj):
        # 计算PR和DC
        page_rank = InverseSampling.pagerank_algorithm(adj)
        degree_centrality = InverseSampling.degree_centrality_algorithm(adj)

        # 根据PR和DC计算采样概率
        node_importance_alpha = 0.5
        node_importance_eps = 1e-6
        node_importance = node_importance_alpha * page_rank + (1 - node_importance_alpha) * degree_centrality
        inverse_node_importance = 1 / (node_importance + node_importance_eps)
        sum_inverse_node_importance = torch.sum(inverse_node_importance)
        sample_prob = inverse_node_importance / sum_inverse_node_importance
        
        return sample_prob
    
    @staticmethod
    def pagerank_algorithm(adj: torch.Tensor, d=0.85, eps=1e-6):
        N = adj.shape[0]
        # 计算出链总数
        out_degree = torch.sum(adj, dim=1)  # 计算每行的和，而不是每列的和
        # 处理零出度：创建一个布尔索引，指示哪些行（节点）的出度为零
        zero_out_degree = out_degree == 0
        # 防止零除，使用一个小数替换零出度
        out_degree[zero_out_degree] = 1
        # 归一化邻接矩阵得到转移概率矩阵
        adj_normalized = adj / out_degree[:, None]  # 归一化每一行
        # 将零出度节点的转移概率设置为均匀分布
        adj_normalized[zero_out_degree] = 1.0 / N  # 设置整行为1/N
        # 初始化PageRank值，每个节点初始值为1/N
        p = torch.ones(N, dtype=torch.float32, device=adj.device) / N
        # 转移概率矩阵转置，因为是按行归一化的
        adj_normalized_t = adj_normalized.t()
        
        while True:
            # PageRank的迭代公式
            new_p = (1 - d) / N + d * torch.mv(adj_normalized_t, p)
            # 检查收敛性，即新旧PageRank值的差异是否小于eps
            if torch.norm(new_p - p, p=1) < eps:
                break
            p = new_p

        return p

    @staticmethod
    def degree_centrality_algorithm(adj):
        # 对于无向图，计算每个节点的度数（即与其相连的节点数）
        degree = torch.sum(adj, dim=0)  # 按列求和或按行求和都可以，因为是无向图
        # 度中心性是度数除以（节点数-1）
        N = adj.shape[0]
        centrality = degree / (N - 1)
        return centrality
    