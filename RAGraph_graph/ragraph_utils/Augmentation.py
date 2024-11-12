import torch

from .InverseSampling import InverseSampling

class Augmentation:
    
    @staticmethod
    def augment_features(features, sample_prob):
        standard_deviation = 0.1
        dropout_rate = 0.01

        # 添加高斯噪声
        noise = torch.randn_like(features) * standard_deviation
        noisy_features = features + noise

        # 节点丢弃
        dropout_mask = torch.bernoulli(sample_prob * dropout_rate).unsqueeze(-1)
        dropped_features = noisy_features * dropout_mask

        return dropped_features

    @staticmethod
    def augment_adj(adj, sample_prob: torch.Tensor):
        # 边重写
        keep_prob = (sample_prob.unsqueeze(1) + sample_prob.unsqueeze(0)) / 2
        random_probs = torch.rand(adj.shape, device=adj.device)
        new_adj = torch.where(random_probs < keep_prob, torch.ones_like(adj), torch.zeros_like(adj))

        return new_adj

    @staticmethod
    def interpolation_node(feature: torch.Tensor, adj: torch.Tensor, interpolation_num: int = 5, alpha: float = 0.5):
        # 节点插入
        
        # prepare new feature
        new_feature = torch.zeros((feature.shape[0] + interpolation_num, feature.shape[1]))
        new_feature[:feature.shape[0]] = feature

        new_adj = torch.zeros((feature.shape[0] + interpolation_num, feature.shape[0] + interpolation_num))
        new_adj[:feature.shape[0], :feature.shape[0]] = adj

        for i in range(feature.shape[0] , feature.shape[0] + interpolation_num):
            src_node, dst_node = torch.randint(0, feature.shape[0], (2,))
            new_feature[i] = alpha * feature[src_node] + (1 - alpha) * feature[dst_node]
            new_adj[i, src_node] = new_adj[src_node, i] = alpha
            new_adj[i, dst_node] = new_adj[dst_node, i] = 1 - alpha

        return new_feature, new_adj
    
    @staticmethod
    def augment_graph(num_augment_scale, features, adj):
        num_loop = 1 + num_augment_scale
        sample_prob = InverseSampling.compute_sample_prob(adj)

        for i in range(num_loop):
            # 应用增强
            if i > 0:
                aug_features = Augmentation.augment_features(features, sample_prob)
                aug_adj = Augmentation.augment_adj(adj, sample_prob)
            else:
                aug_features = features
                aug_adj = adj

            yield aug_features, aug_adj
