import torch
import torch.nn as nn
import torch.nn.functional as F
from models import DGI, GraphCL, Lp, GcnLayers
from layers import AvgReadout 
import numpy as np

def get_subgraph_3(feature, adj):
    adj_3hop = torch.matmul(adj, torch.matmul(adj, adj)).squeeze()
    #adj_3hop = torch.matmul(adj, adj).squeeze()
    adj_3hop[adj_3hop > 0] = 1  #保留距离为3以内的节点

    #print("3s adj", adj_3hop.shape)
    index = torch.nonzero(adj_3hop, as_tuple=False)
    #print("3s index", index.shape)

    
    res = torch.zeros(feature.size(0), feature.size(1)).cuda() 
    cnt = torch.zeros(feature.size(0)).cuda() 
    for i in range(index.size(0)):
        src, dst = index[i][0], index[i][1]
        res[src] += feature[dst]  # 对距离为3以内的节点的特征向量进行累加
        cnt[src] += 1
    for i in range(feature.size(0)):
        res[i] /= cnt[i]

    return res

class PrePrompt(nn.Module):
    def __init__(self, n_in, n_h, activation,num_layers_num,p):
        super(PrePrompt, self).__init__()
        self.dgi = DGI(n_h)
        self.graphcledge = GraphCL(n_in, n_h, activation)
        self.graphclmask = GraphCL(n_in, n_h, activation)
        self.lp = Lp(n_in, n_h)
        self.gcn = GcnLayers(n_in, n_h,num_layers_num,p)
        self.read = AvgReadout()
        
        self.loss = nn.BCEWithLogitsLoss()


    def forward(self, seq1, seq2, seq3, seq4, adj, aug_adj1edge, aug_adj2edge, 
                sparse, msk, samp_bias1, samp_bias2,
                lbl,sample):
        negative_sample = torch.tensor(sample, dtype=int).cuda()
        seq1 = torch.squeeze(seq1,0)
        seq2 = torch.squeeze(seq2,0)
        seq3 = torch.squeeze(seq3,0)
        seq4 = torch.squeeze(seq4,0)
        logits3 = self.lp(self.gcn,seq1,adj,sparse)
        lploss = compareloss(logits3,negative_sample,temperature=1.5)
        lploss.requires_grad_(True)
        
        ret = lploss
        return ret

    def embed(self, seq, adj, sparse, msk,LP):
        h_1 = self.gcn(seq, adj, sparse, LP)
        h = h_1.squeeze()
        sub_3_feature = get_subgraph_3(h, adj)
        c = self.read(sub_3_feature, msk)
        return h.detach(), c.detach()
    
    def inference(self, features, adj):
        h, _ = self.embed(features, adj, False, None, False)
        return h


def mygather(feature, index):
    input_size=index.size(0)
    index = index.flatten()
    index = index.reshape(len(index), 1)
    index = torch.broadcast_to(index, (len(index), feature.size(1)))

    res = torch.gather(feature, dim=0, index=index)
    #print("res", res.shape)
    return res.reshape(input_size,-1,feature.size(1))


def compareloss(feature,tuples,temperature):
    h_tuples=mygather(feature,tuples) #negative
    #print("tuples",h_tuples.shape)
    temp = torch.arange(0, len(tuples))
    temp = temp.reshape(-1, 1)
    temp = torch.broadcast_to(temp, (temp.size(0), tuples.size(1)))
    temp=temp.cuda()  
    h_i = mygather(feature, temp) #positive


    sim = F.cosine_similarity(h_i, h_tuples, dim=2)
    # print("sim",sim)
    exp = torch.exp(sim)
    exp = exp / temperature
    exp = exp.permute(1, 0)
    numerator = exp[0].reshape(-1, 1)
    denominator = exp[1:exp.size(0)]
    denominator = denominator.permute(1, 0)
    denominator = denominator.sum(dim=1, keepdim=True)

    # print("numerator",numerator)
    # print("denominator",denominator)
    res = -1 * torch.log(numerator / denominator)
    return res.mean()


def prompt_pretrain_sample(adj,n):
    #print("adj.shape", adj.shape)
    nodenum=adj.shape[0]
    n = min(n, nodenum)
    indices=adj.indices
    indptr=adj.indptr
    res=np.zeros((nodenum,1+n))
    whole=np.array(range(nodenum))
    # print("#############")
    # print("start sampling disconnected tuples")
    for i in range(nodenum):
        nonzero_index_i_row=indices[indptr[i]:indptr[i+1]]
        zero_index_i_row=np.setdiff1d(whole,nonzero_index_i_row)
        np.random.shuffle(nonzero_index_i_row)
        np.random.shuffle(zero_index_i_row)
        if np.size(nonzero_index_i_row)==0:
            res[i][0] = i
        else:
            res[i][0]=nonzero_index_i_row[0]
        res[i][1:1+n]=zero_index_i_row[0:n]
    return res.astype(int)


