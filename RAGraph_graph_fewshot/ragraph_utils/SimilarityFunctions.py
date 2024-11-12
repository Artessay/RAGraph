import torch
import torch.nn.functional as F

class SimilarityFunctions:
    @staticmethod
    def calculate_cosine_similarity(search_keys: torch.Tensor, resource_keys: torch.Tensor) -> torch.Tensor:
        # Normalize search_keys (query_num, emb_size)
        search_keys = F.normalize(search_keys, p=2, dim=-1)

        # Normalize resource_keys (resource_num, emb_size)
        resource_keys = F.normalize(resource_keys, p=2, dim=-1)

        # Compute similarities between search_keys and resource_keys, (query_num, resource_num)
        cosine_similarities = torch.matmul(search_keys, resource_keys.t())

        return cosine_similarities    

    @staticmethod
    def calculate_jaccard_similarity(adj: torch.Tensor, v_c: int, v_m: int) -> float:
        # get neighbors
        neighbors_vc = adj[v_c].nonzero(as_tuple=False).squeeze(1)
        neighbors_vm = adj[v_m].nonzero(as_tuple=False).squeeze(1)
        
        # compute intersection and union
        intersection = torch.intersect1d(neighbors_vc, neighbors_vm).size(0)
        union = torch.union1d(neighbors_vc, neighbors_vm).size(0)
        
        # return jaccard similarity
        if union == 0:
            return 0.0  # avoid division by zero
        return intersection / union
    