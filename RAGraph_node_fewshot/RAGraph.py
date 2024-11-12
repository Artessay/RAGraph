import torch
import torch.nn as nn

from ragraph_utils import ToyGraphBase, Propagation


class RAGraph(nn.Module):
    def __init__(
            self, 
            pretrain_model, 
            resource_dataset, 
            mean_fewshot_logits, 
            emb_size, 
            finetune=True, 
            noise_finetune=False,
            query_graph_hop=3, 
            retrieve_num=5
    ) -> None:
        super(RAGraph, self).__init__()

        self.emb_size = emb_size
        self.pretrain_model = pretrain_model

        dataset_name = resource_dataset.name
        if dataset_name == 'ENZYMES':
            self.retrieve_weight = 0.5
            self.label_weight = 0.5
        elif dataset_name == 'PROTEINS':
            self.retrieve_weight = 0.3
            self.label_weight = 0.8
        else:
            raise NotImplementedError
        
        self.finetune = finetune

        self.noise_finetune = noise_finetune
        if self.noise_finetune:
            assert self.finetune

        self.query_graph_hop = query_graph_hop
        self.toy_graph_base = ToyGraphBase(pretrain_model, len(mean_fewshot_logits), emb_size, self.query_graph_hop, retrieve_num)
        self.toy_graph_base.build_toy_graph(resource_dataset)
        
        
        self.toy_graph_base.show()

    def forward(self, features, adj, mean_fewshot_logits):
        pretrain_embedddings = self.pretrain_model.encode(features, adj)

        add_noise = self.training and self.noise_finetune
        rag_embeddings, rag_labels = self.toy_graph_base.retrieve(pretrain_embedddings, adj, add_noise)
        
        # (batch_size, retrieve_num)
        rag_labels_max_indices = torch.argmax(rag_labels, dim=-1)
        rag_logits = mean_fewshot_logits[rag_labels_max_indices]
        
        # print("rag embeddings:", rag_embeddings.shape)
        # print("rag labels:", rag_labels.shape)
        # print("rag logits:", rag_logits.shape)

        if self.finetune:
            rag_logits = torch.mean(rag_logits, dim=1)
            rag_embedding = torch.sum(rag_embeddings, dim=1)
            
            query_embeddings = Propagation.aggregate_k_hop_features(adj, pretrain_embedddings, self.query_graph_hop)

            hidden_embedding = query_embeddings * (1 - self.retrieve_weight) + rag_embedding * self.retrieve_weight
            
            decode_logits = self.pretrain_model.decode(hidden_embedding, adj)

            # def normal_vector(vector):
            #     return vector / torch.norm(vector)
            
            # decode_logits = normal_vector(decode_logits)
            # rag_logits = rag_logits / torch.norm(rag_logits)

            label_logits = decode_logits * (1 - self.label_weight) + rag_logits * self.label_weight

            return label_logits
        else:
            rag_logits = torch.mean(rag_logits, dim=1)
            
            return rag_logits
