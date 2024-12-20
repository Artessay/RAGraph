import torch
import torch.nn.functional as F
from torch import nn
from typing import Optional

def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


class EdgelistDrop(nn.Module):
    def __init__(self):
        super(EdgelistDrop, self).__init__()

    def forward(self, edgeList, keep_rate, return_mask=False):
        if keep_rate == 1.0:
            return edgeList, torch.ones(edgeList.size(0)).type(torch.bool)
        edgeNum = edgeList.size(0)
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newEdgeList = edgeList[mask, :]
        if return_mask:
            return newEdgeList, mask
        else:
            return newEdgeList

class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate, return_mask=False):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        mask = (torch.rand(edgeNum) + keep_rate).floor().type(torch.bool)
        newVals = vals[mask]  # / keep_rate
        newIdxs = idxs[:, mask]
        if return_mask:
            return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape), mask
        else:
            return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


def reg_params(model):
    reg_loss = 0
    for W in model.parameters():
        reg_loss += W.norm(2).square()
    return reg_loss

def cal_infonce(view1, view2, temperature, b_cos = True):
    if b_cos:
        view1, view2 = F.normalize(view1, dim=1), F.normalize(view2, dim=1)
    pos_score = (view1 * view2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)
    ttl_score = torch.matmul(view1, view2.transpose(0, 1))
    ttl_score = torch.exp(ttl_score / temperature).sum(dim=1)
    cl_loss = -torch.log(pos_score / ttl_score+10e-6)
    return torch.mean(cl_loss)
