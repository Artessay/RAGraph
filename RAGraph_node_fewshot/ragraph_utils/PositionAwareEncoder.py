import torch

class PositionAwareEncoder:

    @staticmethod
    def encode_position_aware_code(adj: torch.Tensor, num_anchors: int, dis_q: int = 10) -> torch.Tensor:
        distance_matrix = PositionAwareEncoder.floyd_warshall(adj)

        num_nodes = distance_matrix.shape[0]
        # num_anchors = math.log2(num_nodes)
        anchor_index = torch.randint(low=0, high=num_nodes, size=(int(num_anchors),))

        distance_to_centroid = torch.zeros(size=(num_nodes, num_anchors)).to(adj.device)
        for u in range(num_nodes):
            for idx, w in enumerate(anchor_index):
                distance = distance_matrix[u, w]

                # Apply the formula for d2c: 1 / (dis(vu, vw) + 1) if dis(vu, vw) < dis_q, else 0
                if distance < dis_q:
                    distance_to_centroid[u, idx] = 1 / (distance + 1)
                else:
                    distance_to_centroid[u, idx] = 0

        return distance_to_centroid

    @staticmethod
    def floyd_warshall(adj: torch.Tensor):
        """
        Floyd-Warshall algorithm to compute all pairs shortest paths.

        Parameters:
            adj (torch.Tensor): Adjacency matrix of the graph.

        Returns:
            torch.Tensor: Distance matrix of all pairs shortest paths.
        """
        n = adj.shape[0]

        # Initialize distance matrix
        dist = adj.clone()
        dist[adj == 0] = float('inf') # Not connection
        dist[torch.eye(n).bool()] = 0 # Self loop

        for k in range(n):
            # Broadcasting will automatically expand dimensions for comparison
            dist = torch.min(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

        return dist