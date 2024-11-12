import torch
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
