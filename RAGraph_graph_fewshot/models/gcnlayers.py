import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp
from layers import GCN, AvgReadout
import tqdm
import numpy as np


class GcnLayers(torch.nn.Module):
    def __init__(self, n_in, n_h,num_layers_num,dropout):
        super(GcnLayers, self).__init__()

        self.act=torch.nn.ReLU()
        self.num_layers_num=num_layers_num
        self.g_net, self.bns = self.create_net(n_in,n_h,self.num_layers_num)

        self.dropout=torch.nn.Dropout(p=dropout)

    def create_net(self,input_dim, hidden_dim,num_layers):

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()

        for i in range(num_layers):

            if i:
                nn = GCN(hidden_dim, hidden_dim)
            else:
                nn = GCN(input_dim, hidden_dim)
            conv = nn
            bn = torch.nn.BatchNorm1d(hidden_dim)

            self.convs.append(conv)
            self.bns.append(bn)

        return self.convs, self.bns


    def forward(self, seq, adj,sparse,LP=False):
        graph_output = torch.squeeze(seq,dim=0)
        for i in range(self.num_layers_num):
            input=(graph_output,adj)
            graph_output = self.convs[i](input)
            
            if LP:
                assert False
                graph_output = self.bns[i](graph_output)
                graph_output = self.dropout(graph_output)
                
        return graph_output.unsqueeze(dim=0)
    
    @torch.no_grad()
    def encode(self, seq, adj):
        graph_output = torch.squeeze(seq, dim=0)

        graph_input = (graph_output, adj)
        graph_output = self.convs[0](graph_input)

        # LP = True
        # graph_output = self.bns[0](graph_output)
        # graph_output = self.dropout(graph_output)

        return graph_output
    
    def decode(self, embeddings, adj):
        graph_output = embeddings

        graph_input = (graph_output, adj)
        graph_output = self.convs[1](graph_input)

        # LP = True
        # graph_output = self.bns[1](graph_output)
        # graph_output = self.dropout(graph_output)

        return graph_output

