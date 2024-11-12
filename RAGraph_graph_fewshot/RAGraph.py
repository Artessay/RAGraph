import torch
import torch.nn as nn

from ragraph_utils import ToyGraphBase, Propagation, FewShotBase


class RAGraph(nn.Module):
    def __init__(self, pretrain_model, resource_dataset, feture_size, num_class, emb_size, finetune=True, noise_finetune=False) -> None:
        super(RAGraph, self).__init__()

        self.emb_size = emb_size
        self.num_class = num_class
        self.pretrain_model = pretrain_model

        # 0.1, 0.5 for BZR; 0.3, 0.6 for COX2, 0.5, 0.5 for PROTEINS; 0.3, 0.8 for ENZYMES
        dataset_name = resource_dataset.name
        if dataset_name == 'ENZYMES':
            self.retrieve_weight = 0.3
            self.label_weight = 0.8
        elif dataset_name == 'PROTEINS':
            self.retrieve_weight = 0.5
            self.label_weight = 0.5
        elif dataset_name == 'COX2':
            self.retrieve_weight = 0.3
            self.label_weight = 0.6
        elif dataset_name == 'BZR':
            self.retrieve_weight = 0.1
            self.label_weight = 0.5
        else:
            raise NotImplementedError

        self.finetune = finetune

        self.noise_finetune = noise_finetune
        if self.noise_finetune:
            assert self.finetune

        self.query_graph_hop = 1
        self.toy_graph_base = ToyGraphBase(pretrain_model, num_class, emb_size, self.query_graph_hop)
        self.toy_graph_base.build_toy_graph(resource_dataset)
        
        self.toy_graph_base.show()

        self.fewshot_base = FewShotBase(resource_dataset.name, num_class, pretrain_model)

    def forward(self, features, adj, mean_fewshot_logits):
        pretrain_embedddings = self.pretrain_model.encode(features, adj)
        # pretrain_graph_embeddings = torch.mean(pretrain_embedddings, dim=0)
        # print("pretrain graph embedding", pretrain_graph_embeddings.shape)
        
        add_noise = self.training and self.noise_finetune
        rag_embeddings, rag_labels = self.toy_graph_base.retrieve(pretrain_embedddings, adj, add_noise)
        # rag_embeddings, rag_labels = self.toy_graph_base.retrieve(pretrain_graph_embeddings, adj, add_noise)
        
        # (batch_size, retrieve_num)
        rag_labels_max_indices = torch.argmax(rag_labels, dim=-1)
        rag_logits = mean_fewshot_logits[rag_labels_max_indices]

        assert rag_logits.shape[0] == 1
        rag_embeddings = rag_embeddings.squeeze(0)
        rag_logits = rag_logits.squeeze(0)
        
        # print("rag embeddings:", rag_embeddings.shape)
        # print("rag labels:", rag_labels.shape)
        # print("rag logits:", rag_logits.shape)

        if self.finetune:
            rag_logits = torch.mean(rag_logits, dim=1)
            rag_embedding = torch.sum(rag_embeddings, dim=1)
            
            query_embeddings = Propagation.aggregate_k_hop_features(adj, pretrain_embedddings, self.query_graph_hop)
            # query_embedding = torch.mean(query_embeddings, dim=0)
            # print("query embeddings:", query_embeddings.shape)

            hidden_embedding = query_embeddings * (1 - self.retrieve_weight) + rag_embedding * self.retrieve_weight
            # hidden_embedding = query_embedding * (1 - self.retrieve_weight) + rag_embedding * self.retrieve_weight
            
            # print("hidden embedding", hidden_embedding.shape)
            decode_logits = self.pretrain_model.decode(hidden_embedding, adj)
            # print("decode logits", decode_logits.shape)
            
            label_logits = decode_logits * (1 - self.label_weight) + rag_logits * self.label_weight

            label_logits = label_logits.mean(dim=0).unsqueeze(0)
            # print("label logits:", label_logits.shape)

            return label_logits
        else:
            rag_logits = torch.mean(rag_logits, dim=1)
            
            return rag_logits
