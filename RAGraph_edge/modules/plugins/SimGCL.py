import torch
import torch.nn as nn
from modules.plugins.GraphProPluginModel import GraphProPluginModel
from utils.parse_args import args
import torch.nn.functional as F
from modules.utils import EdgelistDrop
from modules.utils import scatter_add, scatter_sum

init = nn.init.xavier_uniform_

def cal_infonce(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)

class SimGCL(GraphProPluginModel):
    def __init__(self, dataset, pretrained_model=None, phase='pretrain'):
        super().__init__(dataset, pretrained_model, phase)
        self.adj = self._make_binorm_adj(dataset.graph)
        self.edges = self.adj._indices().t()
        self.edge_norm = self.adj._values()

        self.eps = args.eps
        self.lbd = args.lbd

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

    def forward(self, edges, edge_norm, edge_times=None, perturbed=False):
        if self.phase not in ['vanilla']:
            time_norm = self._relative_edge_time_encoding(edges, edge_times)
            edge_norm = edge_norm * 1/2 + time_norm * 1/2
        all_emb = torch.cat([self.user_embedding, self.item_embedding], dim=0)
        all_emb = self.emb_gate(all_emb)
        res_emb = [all_emb]
        for l in range(args.num_layers):
            all_emb = self._agg(all_emb, edges, edge_norm)
            if perturbed:
                random_noise = torch.rand_like(all_emb).to(all_emb.device)
                all_emb += torch.sign(all_emb) * F.normalize(random_noise, dim=-1) * self.eps
            res_emb.append(all_emb)
        res_emb = sum(res_emb)
        user_res_emb, item_res_emb = res_emb.split([self.num_users, self.num_items], dim=0)
        return user_res_emb, item_res_emb
    
    def cal_cl_loss(self, idx, edges, edge_norm, edge_times=None):
        u_idx = torch.unique(torch.Tensor(idx[0]).type(torch.long)).to(args.device)
        i_idx = torch.unique(torch.Tensor(idx[1]).type(torch.long)).to(args.device)
        user_view_1, item_view_1 = self.forward(edges, edge_norm, edge_times=edge_times, perturbed=True)
        user_view_2, item_view_2 = self.forward(edges, edge_norm, edge_times=edge_times,perturbed=True)
        user_cl_loss = cal_infonce(user_view_1[u_idx], user_view_2[u_idx], 0.2)
        item_cl_loss = cal_infonce(item_view_1[i_idx], item_view_2[i_idx], 0.2)
        return user_cl_loss + item_cl_loss
    
    def cal_loss(self, batch_data):
        edges, dropout_mask = self.edge_dropout(self.edges, 0.5, return_mask=True)
        edge_norm = self.edge_norm[dropout_mask]
        if self.phase not in ['vanilla']:
            edge_times = self.edge_times[dropout_mask]
        else:
            edge_times = None
            
        # forward
        users, pos_items, neg_items = batch_data
        user_emb, item_emb = self.forward(edges, edge_norm, edge_times=edge_times)
        batch_user_emb = user_emb[users]
        pos_item_emb = item_emb[pos_items]
        neg_item_emb = item_emb[neg_items]
        rec_loss = torch.tensor(0.0) # self._bpr_loss(batch_user_emb, pos_item_emb, neg_item_emb)
        reg_loss = args.weight_decay * self._reg_loss(users, pos_items, neg_items)

        cl_loss = self.lbd * self.cal_cl_loss([users,pos_items], edges, edge_norm, edge_times=edge_times)

        loss = rec_loss + reg_loss + cl_loss
        loss_dict = {
            "rec_loss": rec_loss.item(),
            "reg_loss": reg_loss.item(),
            "cl_loss": cl_loss.item()
        }
        return loss, loss_dict
    
    @torch.no_grad()
    def generate(self):
        return self.forward(self.edges, self.edge_norm, edge_times=self.edge_times)
    
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