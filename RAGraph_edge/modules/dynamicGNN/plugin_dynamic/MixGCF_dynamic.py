import torch
import torch.nn as nn
from utils.parse_args import args
from modules.utils import EdgelistDrop
from modules.base_model import BaseModel
import logging
from modules.utils import scatter_add, scatter_sum
from torch.nn import GRUCell
from torch.nn import GRU
from copy import deepcopy

init = nn.init.xavier_uniform_
logger = logging.getLogger('train_logger')


class BaseModel_1(BaseModel):
    def __init__(self, dataset):
        super().__init__(dataset)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.n_negs = args.n_negs

        self.phase = "vanilla"
        self.edge_dropout = EdgelistDrop()

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
    
    def negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        # print("user gcn emb", user_gcn_emb.shape)
        # print("neg_candidates", neg_candidates.shape)
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        # print(f"p_e: {p_e.shape}")

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        # print(f"seed: {seed.shape}")
        n_e = item_gcn_emb[neg_candidates].view(batch_size, args.n_negs, -1, args.emb_size)  # [batch_size, n_negs, n_hops, channel]
        # print(f"n_e: {n_e.shape}")
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 1-args.edge_dropout, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]
        if self.phase not in ['vanilla']:
            edge_times = self.edge_times[dropout_mask]
        else:
            edge_times = None

        # forward
        # neg_items: B, n_negs
        users, pos_items, neg_items = batch_data
        # print(f"neg_items: {neg_items.shape}")
        user_emb, item_emb, res_emb = self.forward(edges, edge_norm, edge_times, return_res_emb=True)
        user_stack_emb, item_stack_emb = torch.stack(res_emb, dim=1).split([self.num_users, self.num_items], dim=0)
        neg_item_emb = self.negative_sampling(user_stack_emb, item_stack_emb, users, neg_items, pos_items).sum(dim=1)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        rec_loss = self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        loss = rec_loss + reg_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
        }
        return loss, loss_dict
    
    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_norm)
    
    @torch.no_grad()
    def rating(self, user_emb, item_emb):
        return torch.matmul(user_emb, item_emb.t())
    
    def _reg_loss(self, users, pos_items, neg_items):
        u_emb = self.user_embedding[users]
        pos_i_emb = self.item_embedding[pos_items]
        neg_i_emb = self.item_embedding[neg_items]
        reg_loss = (1/2)*(u_emb.norm(2).pow(2) +
                          pos_i_emb.norm(2).pow(2) +
                          neg_i_emb.norm(2).pow(2))/float(len(users))
        return reg_loss

class MixGCF_evolveGCN_O(BaseModel_1):
    def __init__(self, dataset, pretrained_model=None, last_emb=None):
        super().__init__(dataset)

        self.gru_cell = GRUCell(input_size=self.emb_size, hidden_size=self.emb_size)
        
        if pretrained_model is not None:
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
        else:
            last_user_emb, last_item_emb = last_emb.split([self.num_users, self.num_items], dim=0)
            self.user_embedding = nn.Parameter(last_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(last_item_emb).requires_grad_(True)

    def forward(self, edges, edge_norm, edge_times=None, return_res_emb=False):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.gru_cell(all_emb, all_emb)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        user_res_emb, item_res_emb = sum(res_emb).split([self.num_users, self.num_items], dim=0)
        if return_res_emb:
            return user_res_emb, item_res_emb, res_emb
        return user_res_emb, item_res_emb


class MixGCF_evolveGCN_H(BaseModel_1):
    def __init__(self, dataset, pretrained_model=None, last_emb=None):
        super().__init__(dataset)
        self.recurrent_layer = GRU(input_size=self.emb_size,
                                   hidden_size=self.emb_size,
                                   num_layers=1)
        self.last_emb = last_emb
        
        if pretrained_model is not None:
            pre_user_emb, pre_item_emb = pretrained_model.generate()
            self.user_embedding = nn.Parameter(pre_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(pre_item_emb).requires_grad_(True)
        else:
            last_user_emb, last_item_emb = last_emb.split([self.num_users, self.num_items], dim=0)
            self.user_embedding = nn.Parameter(last_user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(last_item_emb).requires_grad_(True)

        self.edge_dropout = EdgelistDrop()

    def forward(self, edges, edge_norm, edge_times=None, return_res_emb=False):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb, _ = self.recurrent_layer(all_emb.unsqueeze(0), self.last_emb.unsqueeze(0))
        all_emb = all_emb.squeeze(0)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        user_res_emb, item_res_emb = sum(res_emb).split([self.num_users, self.num_items], dim=0)
        if return_res_emb:
            return user_res_emb, item_res_emb, res_emb
        return user_res_emb, item_res_emb

@torch.no_grad()
def average_state_dict(state_dict1: dict, state_dict2: dict, weight: float) -> dict:
    # Average two model.state_dict() objects.
    # out = (1-w)*dict1 + w*dict2
    assert 0 <= weight <= 1
    d1 = deepcopy(state_dict1)
    d2 = deepcopy(state_dict2)
    out = dict()
    for key in d1.keys():
        assert isinstance(d1[key], torch.Tensor)
        param1 = d1[key].detach().clone()
        assert isinstance(d2[key], torch.Tensor)
        param2 = d2[key].detach().clone()
        out[key] = (1 - weight) * param1 + weight * param2
    return out

class MixGCF_roland(BaseModel_1):
    def __init__(self, dataset, pretrain_model=None, meta_model=None):
        super().__init__(dataset)

        # this is maintained and updated for each t, initialized by pretrained model
        self.gru = GRUCell(self.emb_size, self.emb_size)

        self.meta_model = meta_model

        # t1, initialize meta model with pretrained model
        if pretrain_model is not None:
            user_emb, item_emb = pretrain_model.generate()
            self.user_embedding = nn.Parameter(user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(item_emb).requires_grad_(True)
        
        elif meta_model is not None:
            self.load_state_dict(meta_model.state_dict(), strict=False)
            user_emb, item_emb = meta_model.generate_lgn()
            self.user_embedding = nn.Parameter(user_emb).requires_grad_(True)
            self.item_embedding = nn.Parameter(item_emb).requires_grad_(True)

    def update_meta_model(self, model, meta_sd):
        if "gru.weight_ih" not in meta_sd:
            sd = model.state_dict()
            nsd = {
                "user_embedding": sd["user_embedding"],
                "item_embedding": sd["item_embedding"],
            }
        else:
            nsd = model.state_dict()
        print("nsd", nsd.keys())
        print("meta sd", meta_sd.keys())
        new_sd = average_state_dict(nsd, meta_sd, 0.9)
        # print(self.meta_model_sd)
        # print(last_model.state_dict())
        print("loading state dict:", new_sd.keys())
        print("last model state dict:", model.state_dict().keys())
        model.load_state_dict(new_sd, strict=False)
        return model
    
    def forward(self, edges, edge_norm, edge_times=None, return_res_emb=False):
        last_user_emb, last_item_emb = self.meta_model.generate_lgn(return_layers=True)
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            last_emb_i = torch.cat([last_user_emb[l+1], last_item_emb[l+1]], dim=0)
            all_emb = self.gru(all_emb, last_emb_i)
            res_emb.append(all_emb)

        if return_res_emb:
            user_res_emb, item_res_emb = sum(res_emb).split([self.num_users, self.num_items], dim=0)
            return user_res_emb, item_res_emb, res_emb
        
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb

    def forward_lgn(self, edges, edge_norm, edge_times=None, return_layers=False):
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(res_emb[-1], edges, edge_norm)
            res_emb.append(all_emb)

        if not return_layers:
            res_emb = sum(res_emb)
            user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        else:
            user_res_emb, item_res_emb = [], []
            for emb in res_emb:
                u_emb, i_emb = emb.split([self.num_users, self.num_items], dim=0)
                user_res_emb.append(u_emb)
                item_res_emb.append(i_emb)
        return user_res_emb, item_res_emb

    @torch.no_grad()
    def generate_lgn(self, return_layers=False):
        return self.forward_lgn(self.edges, self.edge_norm, return_layers=return_layers)