import torch
import torch.nn as nn
from torch import Tensor
from modules.base_model import BaseModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
import logging
from modules.utils import scatter_add, scatter_sum
from torch_scatter import scatter_softmax

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')

from modules.ragraph_utils.Augmentation import Augmentation
from modules.ragraph_utils.InverseSampling import InverseSampling
from modules.ragraph_utils.SimilarityFunctions import SimilarityFunctions

class RAGraph(BaseModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain', use_RAG=True, use_noise=False, use_LoRA=True, LoRA_rank=16):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.edge_times = [dataset.edge_time_dict[e[0]][e[1]] for e in self.edges.cpu().tolist()]
        self.edge_times = torch.LongTensor(self.edge_times).to(args.device)

        self.phase = phase
        self.use_RAG = use_RAG
        self.use_noise = use_noise and phase == 'finetune'

        data_path = args.data_path
        if 'amazon' in data_path:
            self.retrieve_weight = 0.3


            if self.phase == 'vanilla':
                self.batch_size = 32768
                self.retrieve_num = 50

                self.num_augment_scale = 0
                self.num_inverse_sample = round(0.01 * len(self.adj))
            else:
                self.batch_size = 4096
                self.retrieve_num = 10
                self.noise_retrieve_num = 1

                self.num_augment_scale = 0
                self.num_inverse_sample = 0
                
        elif 'koubei' in data_path:
            self.retrieve_weight = 0.3

            if self.phase == 'vanilla':
                self.batch_size = 512
                self.retrieve_num = 100000

                self.num_augment_scale = 1
                self.num_inverse_sample = round(0.01 * len(self.adj))
            else:
                self.batch_size = 4096 # 8192
                self.retrieve_num = 20 # 1000
                self.noise_retrieve_num = 1
                
                self.num_augment_scale = 0 # 3
                self.num_inverse_sample = 0 # round(0.1 * len(self.adj))
        elif 'taobao' in data_path:
            self.retrieve_weight = 0.3

            if self.phase == 'vanilla':
                self.batch_size = 512
                self.retrieve_num = 100000

                self.num_augment_scale = 1
                self.num_inverse_sample = round(0.01 * len(self.adj))
            else:
                self.batch_size = 4096 # 16384
                self.retrieve_num = 20 # 2000
                self.noise_retrieve_num = 1
                
                self.num_augment_scale = 0 # 1
                self.num_inverse_sample = 0 # round(0.01 * len(self.adj))
        else:
            raise NotImplementedError

        self.resource_graph_radius = args.num_layers

        self.resource_keys = None
        self.resource_times = None
        self.resource_values = None

        if self.phase == 'pretrain' or self.phase == 'for_tune':
            self.user_embedding = nn.Parameter(init(torch.empty(self.num_users, self.emb_size)))
            self.item_embedding = nn.Parameter(init(torch.empty(self.num_items, self.emb_size)))

            if self.phase == 'for_tune':
                self.emb_gate = self.random_gate
            elif self.phase == 'pretrain':
                self.emb_gate = lambda x: x # no gating

        # load gating weights from pretrained model
        elif self.phase == 'vanilla':
            pre_user_emb, pre_item_emb = pretrained_model.generate()

            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(False)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(False)

            if self.use_RAG:
                self._make_resource_graph(pretrained_model)

            self.emb_gate = lambda x: x # no gating

        # load gating weights from pretrained model
        elif self.phase == 'finetune':
            pre_user_emb, pre_item_emb = pretrained_model.generate()

            if self.use_RAG:
                self._make_resource_graph(pretrained_model)

            # LoRA finetune
            self.use_LoRA = use_LoRA
            if use_LoRA:
                self.user_embedding = nn.Parameter(pre_user_emb).detach().requires_grad_(False)
                # self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)

                self.item_embedding = nn.Parameter(pre_item_emb).detach().requires_grad_(False)
                # self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

                U, S, V = torch.svd(self.user_embedding)
                U_r = U[:, :LoRA_rank]
                S_r = S[:LoRA_rank]
                V_r = V[:, :LoRA_rank]

                Sigma_r = torch.diag(S_r)

                self.user_embedding_A = U_r @ Sigma_r  # N x r
                self.user_embedding_B = V_r.t()  # r x M

                self.user_embedding_A = self.user_embedding_A.detach().requires_grad_(True)
                self.user_embedding_B = self.user_embedding_B.detach().requires_grad_(True)


                U, S, V = torch.svd(self.item_embedding)
                U_r = U[:, :LoRA_rank]
                S_r = S[:LoRA_rank]
                V_r = V[:, :LoRA_rank]

                Sigma_r = torch.diag(S_r)

                self.item_embedding_A = U_r @ Sigma_r  # N x r
                self.item_embedding_B = V_r.t()  # r x M

                self.item_embedding_A = self.item_embedding_A.detach().requires_grad_(True)
                self.item_embedding_B = self.item_embedding_B.detach().requires_grad_(True)

                self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
                self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

                self.lora_dropout = nn.Dropout(args.emb_dropout)
            else:
                # note grad
                self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
                self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)

            self.gating_weight = nn.Parameter(init(torch.empty(args.emb_size, args.emb_size)))
            self.gating_bias = nn.Parameter(init(torch.empty(1, args.emb_size)))

            self.emb_dropout = nn.Dropout(args.emb_dropout)

            self.emb_gate = lambda x: self.emb_dropout(torch.mul(x, torch.sigmoid(torch.matmul(x, self.gating_weight) + self.gating_bias)))
        
        self.edge_dropout = EdgelistDrop()

        logger.info(f"Max Time Step: {self.edge_times.max()}")
    
    def random_gate(self, x):
        gating_weight = F.normalize(torch.randn((args.emb_size, args.emb_size)).to(args.device))
        gating_bias = F.normalize(torch.randn((1, args.emb_size)).to(args.device))

        gate = torch.sigmoid(torch.matmul(x, gating_weight) + gating_bias)

        return torch.mul(x, gate)
    
    def _make_resource_graph(self, pretrained_model: BaseModel):
        pre_user_emb, pre_item_emb = pretrained_model.generate()
        all_emb = torch.cat([pre_user_emb, pre_item_emb], dim=0)
        
        # aggregate embeddings in bipartition resource graphs
        res_emb = [all_emb]
        for _ in range(self.resource_graph_radius):
            all_emb = self._agg(res_emb[-1], self.edges, self.edge_norm)
            res_emb.append(all_emb)
        dual_res_emb = res_emb[0::2]

        all_logits = sum(dual_res_emb)

        # inverse sampling
        sample_prob = InverseSampling.compute_sample_prob(self.adj)

        num_loop = 1 + self.num_augment_scale
        for i in range(0, num_loop):
            # augment graph
            if i > 0:
                aug_keys = Augmentation.augment_features(all_emb, sample_prob)
                aug_values = Augmentation.augment_features(all_logits, sample_prob)

            else:
                aug_keys: Tensor = all_emb
                aug_values: Tensor = all_logits

            if self.num_inverse_sample > 0:
                sample_mask = torch.multinomial(sample_prob, num_samples=self.num_inverse_sample, replacement=True)

                sample_keys: Tensor = aug_keys[sample_mask]
                sample_values: Tensor = aug_values[sample_mask]
            else:
                sample_keys: Tensor = aug_keys
                sample_values: Tensor = aug_values
        
            if self.resource_keys is not None:
                self.resource_keys = torch.cat((self.resource_keys, sample_keys), dim=0)
                self.resource_values = torch.cat((self.resource_values, sample_values), dim=0)
            else:
                self.resource_keys = sample_keys
                self.resource_values = sample_values

        # self.resource_keys = all_emb
        # self.resource_values = aug_values
        # self.resource_times = None # self.edge_times

    def _agg(self, all_emb, edges, edge_norm):
        src_emb = all_emb[edges[:, 0]]

        # bi-norm
        src_emb = src_emb * edge_norm.unsqueeze(1)

        # conv
        dst_emb = scatter_sum(src_emb, edges[:, 1], dim=0, dim_size=self.num_users+self.num_items)
        return dst_emb
    
    def _edge_binorm(self, edges):
        user_degs = scatter_add(torch.ones_like(edges[:, 0]), edges[:, 0], dim=0, dim_size=self.num_users)
        user_degs = user_degs[edges[:, 0]]
        item_degs = scatter_add(torch.ones_like(edges[:, 1]), edges[:, 1], dim=0, dim_size=self.num_items)
        item_degs = item_degs[edges[:, 1]]
        norm = torch.pow(user_degs, -0.5) * torch.pow(item_degs, -0.5)
        return norm
    
    def _relative_edge_time_encoding(self, edges, edge_times, max_step=None):
        # for each node, normalize edge_times according to its neighbors
        # edge_times: [E]
        # rescal to 0-1
        edge_times = edge_times.float()
        if max_step is None:
            max_step = edge_times.max()
        edge_times = (edge_times - edge_times.min()) / (max_step - edge_times.min())
        # edge_times = torch.exp(edge_times)
        # edge_times = torch.sigmoid(edge_times)
        dst_nodes = edges[:, 1]
        time_norm = scatter_softmax(edge_times, dst_nodes, dim_size=self.num_users+self.num_items)
        # time_norm = time_norm[dst_nodes]
        return time_norm

    def forward(self, edges, edge_norm, edge_times, max_time_step=None):
        time_norm = self._relative_edge_time_encoding(edges, edge_times, max_step=max_time_step)
        edge_norm = edge_norm * 1/2 + time_norm * 1/2

        if self.phase == 'finetune' and self.use_LoRA:
            user_embedding = self.user_embedding + self.lora_dropout(self.user_embedding_A @ self.user_embedding_B)
            item_embedding = self.item_embedding + self.lora_dropout(self.item_embedding_A @ self.item_embedding_B)
        else:
            user_embedding = self.user_embedding
            item_embedding = self.item_embedding

        all_emb = torch.cat([user_embedding, item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)

        # aggregate representations of neighbors
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        # aggregate representations of resource graphs
        if self.use_RAG and (self.phase == 'vanilla' or self.phase == 'finetune'):
            query_emb = res_emb[0]
            add_noise = self.use_noise and self.training # add noise only when training

            # need to split batch
            batch_size = self.batch_size
            query_num = query_emb.shape[0]
            # logger.info(f"[RAG] Query Num: {query_num}")

            rag_emb = torch.empty((query_emb.shape[0], self.resource_values.shape[1]), 
                dtype=query_emb.dtype, device=query_emb.device)

            for start in range(0, query_num, batch_size):
                end = min(start + batch_size, query_num)
                batch_emb = query_emb[start:end]

                # (query_num, resource_num)
                semantic_similarities = SimilarityFunctions.calculate_cosine_similarity(batch_emb, self.resource_keys)

                # (query_num, num_metric)
                # similarity_scores = torch.einsum('ij,jkl->ikl', similarity_weights, similarity_matrices).squeeze(0)
                similarity_scores = semantic_similarities

                # Get the top-k scores and their indices for each query in one operation
                retrieve_num = self.retrieve_num + self.noise_retrieve_num if add_noise else self.retrieve_num
                topk_scores, topk_indices = torch.topk(similarity_scores, retrieve_num, largest=True, sorted=True)

                # (query_num, topk, emb_size)
                batch_rag_emb = self.resource_values[topk_indices]  
    
                if add_noise:
                    noise_indices = torch.randint(0, self.resource_values.shape[0], (batch_emb.shape[0], self.noise_retrieve_num))
                    noise_rag_emb = self.resource_values[noise_indices]
                    batch_rag_emb = torch.cat([batch_rag_emb, noise_rag_emb], dim=1)

                # (query_num, emb_size)
                batch_rag_emb = batch_rag_emb.mean(dim=1)

                rag_emb[start:end] = batch_rag_emb


            res_emb = sum(res_emb)
            res_emb = (1 - self.retrieve_weight) * res_emb + self.retrieve_weight * rag_emb
        else:
            res_emb = sum(res_emb)
        
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb
    
    def cal_loss(self, batch_data):
        # randomly dropout some edges in graph
        edges, dropout_mask = self.edge_dropout(self.edges, 1-args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]
        edge_times = self.edge_times[dropout_mask]

        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm, edge_times)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict
    
    @torch.no_grad()
    def generate(self, max_time_step=None):
        return self.forward(self.edges, self.edge_norm, self.edge_times, max_time_step=max_time_step)
    
    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
    
    def _reg_loss(self, users, pos_items, neg_items):
        if self.phase == 'finetune' and self.use_LoRA:
            user_embedding = self.user_embedding + self.lora_dropout(self.user_embedding_A @ self.user_embedding_B)
            item_embedding = self.item_embedding + self.lora_dropout(self.item_embedding_A @ self.item_embedding_B)
        else:
            user_embedding = self.user_embedding
            item_embedding = self.item_embedding

        u_emb = user_embedding[users]
        pos_i_emb = item_embedding[pos_items]
        neg_i_emb = item_embedding[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss