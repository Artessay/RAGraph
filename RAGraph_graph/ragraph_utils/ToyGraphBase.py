import torch
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from .Augmentation import Augmentation
from .InverseSampling import InverseSampling
from .Propagation import Propagation
from .SimilarityFunctions import SimilarityFunctions
# from .PositionAwareEncoder import PositionAwareEncoder
from .utility import process_tu_dataset

class ToyGraphBase:
    def __init__(self, pretrain_model, num_class, emb_size, query_graph_hop) -> None:
        # 0, 0, 3 for ENZYMES
        # 10, 3, 5 for PROTEINS
        
        # construct phase
        self.num_inverse_sample = 0 # set 0 to disable inverse sampling
        self.num_augment_scale = 0  # set 0 to disable augment
        
        # inference phase
        self.retrieve_num = min(3, num_class + 1)
        self.noise_retrieve_num = 1
        self.noise_std = 0.01

        # self.num_anchors = 10 # log2(19580)
        # self.dis_q = 10

        # self.structure_weight = 0.0
        # self.semantic_weight = 0.999

        # resources
        self.toy_graph_hop = query_graph_hop - 1 # RAG is also 1 hop
        self.pretrain_model = pretrain_model
        
        self.resource_keys = torch.empty(size=(0, emb_size)).cuda()
        self.resource_values = torch.empty(size=(0, emb_size)).cuda()
        self.resource_labels = torch.empty(size=(0, num_class)).cuda()
        # self.resource_positions = torch.empty(size=(0, self.num_anchors)).cuda()

    def build_toy_graph(self, resource_dataset: TUDataset):
        num_classes = resource_dataset.num_classes
        num_node_attributes = resource_dataset.num_node_attributes
        resource_loader = DataLoader(resource_dataset, batch_size=1, shuffle=False)
        for data in resource_loader:
            # print(data)
            assert data.ptr.shape[0] == 2 and len(data.ptr.shape) == 1
            assert len(data.y) == 1

            features, adj = process_tu_dataset(data, num_classes, num_node_attributes)
            self._build_toy_graph_base(features, adj, data.y)

    def retrieve(self, search_keys: Tensor, search_adj: Tensor, add_noise: bool):
        # # (query_num, resource_num)
        # search_positions = PositionAwareEncoder.encode_position_aware_code(search_adj, self.num_anchors, self.dis_q)
        # structure_similarities = SimilarityFunctions.calculate_cosine_similarity(search_positions, self.resource_positions)
        
        # (query_num, resource_num)
        semantic_similarities = SimilarityFunctions.calculate_cosine_similarity(search_keys, self.resource_keys)

        # # (1, num_metric)
        # similarity_weights = torch.tensor([[self.structure_weight, self.semantic_weight]]).cuda()
        # # (num_metric, query_num, resource_num)
        # similarity_matrices = torch.stack([structure_similarities, semantic_similarities], dim=0)
        
        # # (query_num, num_metric)
        # similarity_scores = torch.einsum('ij,jkl->ikl', similarity_weights, similarity_matrices).squeeze(0)

        similarity_scores = semantic_similarities
        similarity_scores = similarity_scores.unsqueeze(0)
        # print("similarity scores", similarity_scores.shape)

        # Get the top-k scores and their indices for each query in one operation
        retrieve_num = 2 * self.retrieve_num if add_noise else self.retrieve_num
        topk_scores, topk_indices = torch.topk(similarity_scores, retrieve_num, largest=True, sorted=True)

        # Retrieve the embeddings and labels corresponding to the top-k indices
        rag_embeddings = self.resource_values[topk_indices]  # (query_num, topk, emb_size)
        rag_labels = self.resource_labels[topk_indices]      # (query_num, topk, 1)
        
        if add_noise:
            rag_embeddings = self._add_noise(rag_embeddings)

        return rag_embeddings, rag_labels

    def show(self):
        print('resource_keys', self.resource_keys.shape)
        print('resource_values', self.resource_values.shape)
        print('resource_labels', self.resource_labels.shape)
        # print('resource positions', self.resource_positions.shape)

        print("label count distribution", torch.sum(self.resource_labels, dim=0))

    def _build_toy_graph_base(self, features, adj, graph_label):
        for aug_features, aug_adj in Augmentation.augment_graph(self.num_augment_scale, features, adj):
            embedddings = self.pretrain_model.inference(aug_features, aug_adj)

            if self.num_inverse_sample > 0:
                # sample subgraph based on probability
                sample_prob = InverseSampling.compute_sample_prob(aug_adj)
                sample_mask = torch.multinomial(sample_prob, num_samples=self.num_inverse_sample, replacement=True)
                
                sample_adj = adj[sample_mask, :][:, sample_mask]
                sample_keys: Tensor = embedddings[sample_mask]
            else:
                sample_adj = aug_adj
                sample_keys: Tensor = embedddings
            
            sample_keys = F.normalize(sample_keys, p=2, dim=-1)
            sample_values = Propagation.aggregate_k_hop_features(sample_adj, sample_keys, self.toy_graph_hop)
            # print(sample_keys.shape)

            # (1, emb_size)
            sample_keys = torch.mean(sample_keys, dim=0).unsqueeze(0)
            # (1, emb_size)
            sample_values = torch.mean(sample_values, dim=0).unsqueeze(0)
            # (1, num_class)
            sample_labels = F.one_hot(graph_label, num_classes=self.resource_labels.shape[1]).cuda()
            # print(sample_labels)
            
            self.resource_keys = torch.cat((self.resource_keys, sample_keys), dim=0)
            self.resource_values = torch.cat((self.resource_values, sample_values), dim=0)
            self.resource_labels = torch.cat((self.resource_labels, sample_labels), dim=0)

            # sample_positions = PositionAwareEncoder.encode_position_aware_code(sample_adj, self.num_anchors, self.dis_q)
            # self.resource_positions = torch.cat((self.resource_positions, sample_positions), dim=0)

    def _add_noise(self, rag_embeddings: torch.Tensor) -> torch.Tensor:
        noise_std = self.noise_std
        noise = torch.normal(mean=0, std=noise_std, size=rag_embeddings.shape).to(rag_embeddings.device)
        return rag_embeddings + noise