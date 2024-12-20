import torch

class InverseSampling:

    @staticmethod
    def compute_sample_prob(adj: torch.sparse.FloatTensor):
        # Compute PageRank and Degree Centrality
        page_rank = InverseSampling.pagerank_algorithm(adj)
        degree_centrality = InverseSampling.degree_centrality_algorithm(adj)

        # Calculate sampling probability based on PageRank and Degree Centrality
        node_importance_alpha = 0.5
        node_importance_eps = 1e-6
        node_importance = node_importance_alpha * page_rank + (1 - node_importance_alpha) * degree_centrality
        inverse_node_importance = 1 / (node_importance + node_importance_eps)
        sum_inverse_node_importance = torch.sum(inverse_node_importance)
        sample_prob = inverse_node_importance / sum_inverse_node_importance

        return sample_prob

    @staticmethod
    def pagerank_algorithm(adj: torch.sparse.FloatTensor, d=0.85, eps=1e-6):
        N = adj.shape[0]

        # Calculate out-degree by summing each row
        out_degree = torch.sparse.sum(adj, dim=1).to_dense()

        # Identify nodes with zero out-degree
        zero_out_degree = out_degree == 0

        # Replace zero out-degree with 1 to avoid division by zero
        out_degree_adj = out_degree.clone()
        out_degree_adj[zero_out_degree] = 1

        # Normalize the adjacency matrix to obtain the transition probability matrix
        adj_indices = adj._indices()
        adj_values = adj._values()
        adj_normalized_values = adj_values / out_degree_adj[adj_indices[0, :]]
        adj_normalized = torch.sparse_coo_tensor(adj_indices, adj_normalized_values, adj.size(), device=adj.device)
        
        # Transpose the normalized adjacency matrix for multiplication
        adj_normalized_t = adj_normalized.transpose(0, 1).coalesce()
        # Initialize PageRank values with 1/N for each node
        p = torch.ones(N, dtype=torch.float32, device=adj.device) / N

        while True:
            # Calculate the contribution from dangling nodes (zero out-degree nodes)
            dangling_contrib = torch.sum(p[zero_out_degree]) / N
            # Perform sparse matrix multiplication: adj_normalized_t * p
            # torch.sparse.mm expects a 2D tensor, so p is reshaped to (N, 1)
            # The result is then squeezed back to a 1D tensor
            sparse_product = torch.sparse.mm(adj_normalized_t, p.unsqueeze(1)).squeeze(1)
            # Apply the PageRank iteration formula
            new_p = (1 - d) / N + d * (sparse_product + dangling_contrib)
            # Check for convergence using L1 norm
            if torch.norm(new_p - p, p=1) < eps:
                break
            p = new_p

        return p

    @staticmethod
    def degree_centrality_algorithm(adj: torch.sparse.FloatTensor):
        # Calculate the degree of each node by summing each column
        degree = torch.sparse.sum(adj, dim=0).to_dense()
        N = adj.shape[0]
        # Degree centrality is the degree divided by (N - 1)
        centrality = degree / (N - 1)
        return centrality
