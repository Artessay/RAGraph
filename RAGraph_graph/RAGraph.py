import torch
import torch.nn as nn

from ragraph_utils import ToyGraphBase, Propagation, TaskDecoder, FewShotBase


class RAGraph(nn.Module):
    def __init__(self, pretrain_model, resource_dataset, feture_size, num_class, emb_size, finetune=True, noise_finetune=False) -> None:
        super(RAGraph, self).__init__()

        self.emb_size = emb_size
        self.num_class = num_class
        self.pretrain_model = pretrain_model

        # 0.1, 0.5 for BZR; 0.3, 0.6 for COX2, 0.5, 0.5 for PROTEINS; 0.3, 0.8 for ENZYMES
        # self.retrieve_weight = 0.1
        # self.label_weight = 0.5
        # ------------------------
        # self.retrieve_weight = 0.3
        # self.label_weight = 0.6
        # ------------------------
        # self.retrieve_weight = 0.5
        # self.label_weight = 0.5
        # ------------------------
        self.retrieve_weight = 0.3
        self.label_weight = 0.3
        self.finetune = finetune

        self.noise_finetune = noise_finetune
        if self.noise_finetune:
            assert self.finetune

        self.query_graph_hop = 1
        self.toy_graph_base = ToyGraphBase(pretrain_model, num_class, emb_size, self.query_graph_hop)
        self.toy_graph_base.build_toy_graph(resource_dataset)
        
        if self.finetune:
            self.decoder = TaskDecoder(emb_size, emb_size, num_class)
            self.reset_parameters()
        
        self.toy_graph_base.show()

        self.fewshot_base = FewShotBase(resource_dataset.name, num_class, pretrain_model)

    def reset_parameters(self):
        self.decoder.reset_parameters()

    def forward(self, features, adj):
        pretrain_embedddings = self.pretrain_model.inference(features, adj)
        pretrain_graph_embeddings = torch.mean(pretrain_embedddings, dim=0)
        # print("pretrain graph embedding", pretrain_graph_embeddings.shape)
        
        add_noise = self.training and self.noise_finetune
        rag_embeddings, rag_labels = self.toy_graph_base.retrieve(pretrain_graph_embeddings, adj, add_noise)
        # print("rag embeddings:", rag_embeddings.shape)
        # print("rag labels:", rag_labels.shape)

        if self.finetune:
            rag_label = torch.mean(rag_labels, dim=1)
            rag_embedding = torch.sum(rag_embeddings, dim=1)
            
            query_embeddings = Propagation.aggregate_k_hop_features(adj, pretrain_embedddings, self.query_graph_hop)
            query_embedding = torch.mean(query_embeddings, dim=0)

            hidden_embedding = query_embedding * (1 - self.retrieve_weight) + rag_embedding * self.retrieve_weight
            decode_label = self.decoder(hidden_embedding)
            decode_label = torch.softmax(decode_label, dim=1)

            label_logits = decode_label * (1 - self.label_weight) + rag_label * self.label_weight

            return label_logits
        else:
            rag_label = torch.mean(rag_labels, dim=1)
            
            return rag_label
