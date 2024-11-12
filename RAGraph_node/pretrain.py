import numpy as np
import scipy.sparse as sp

from preprompt import PrePrompt
import preprompt
from utils import process
import aug
import os
import argparse
import torch
import torch.nn as nn

from ragraph_utils import seed_everything


parser = argparse.ArgumentParser("RAGraph")

parser.add_argument('--dataset', type=str, default="ENZYMES", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
device = torch.device("cuda")

print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
seed = args.seed
seed_everything(seed)

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
# training params

batch_size = 16 if args.dataset == 'ENZYMES' else 8
nb_epochs = 1000
# nb_epochs = 10

patience = 10
lr = 0.001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = False


nonlinearity = 'prelu'  # special name to separate parameters

dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
ft_size = dataset.num_node_attributes # feature size


model = PrePrompt(ft_size, hid_units, nonlinearity,1,0.3).cuda()
# model = model.to(device)

best = 1e9

for epoch in range(nb_epochs):
    seed_everything(seed)
    loss = 0
    regloss = 0
    for step, data in enumerate(loader):
        features, adj, nodelabels= process.process_tu(data, ft_size)

        negetive_sample = preprompt.prompt_pretrain_sample(adj, 100)


        nb_nodes = features.shape[0]  # node number
        # ft_size = features.shape[1]  # node features dim
        nb_classes = nodelabels.shape[1]  # classes = 6

        features = torch.FloatTensor(features[np.newaxis])

        '''
        # ------------------------------------------------------------
        # edge node mask subgraph
        # ------------------------------------------------------------
        '''
        aug_features1edge = features
        aug_features2edge = features

        aug_adj1edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges
        aug_adj2edge = aug.aug_random_edge(adj, drop_percent=drop_percent)  # random drop edges

        '''
        # ------------------------------------------------------------
        '''

        adj = process.normalize_adj(adj + sp.eye(adj.shape[0]))
        aug_adj1edge = process.normalize_adj(aug_adj1edge + sp.eye(aug_adj1edge.shape[0]))
        aug_adj2edge = process.normalize_adj(aug_adj2edge + sp.eye(aug_adj2edge.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
            sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)


        else:
            adj = adj.todense()
            aug_adj1edge = aug_adj1edge.todense()
            aug_adj2edge = aug_adj2edge.todense()

        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
            aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])
            
        labels = torch.FloatTensor(nodelabels[np.newaxis])

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        if step==0:
            features = features.cuda()
            aug_features1edge = aug_features1edge.cuda()
            aug_features2edge = aug_features2edge.cuda()
            if sparse:
                sp_adj = sp_adj.cuda()
                sp_aug_adj1edge = sp_aug_adj1edge.cuda()
                sp_aug_adj2edge = sp_aug_adj2edge.cuda()
            else:
                adj = adj.cuda()
                aug_adj1edge = aug_adj1edge.cuda()
                aug_adj2edge = aug_adj2edge.cuda()
            labels = labels.cuda()
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()

        model.train()
        optimiser.zero_grad()
        idx = np.random.permutation(nb_nodes)
        shuf_fts = features[:, idx, :]
        lbl_1 = torch.ones(1, nb_nodes)
        lbl_2 = torch.zeros(1, nb_nodes)
        lbl = torch.cat((lbl_1, lbl_2), 1)
        if torch.cuda.is_available():
            shuf_fts = shuf_fts.cuda()
            lbl = lbl.cuda()
            
        logit = model(features, shuf_fts, aug_features1edge, aug_features2edge,
                       sp_adj if sparse else adj,
                    sp_aug_adj1edge if sparse else aug_adj1edge,
                    sp_aug_adj2edge if sparse else aug_adj2edge,
                    sparse, None, None, None, lbl=lbl,sample=negetive_sample)
        loss = loss + logit
        showloss = loss/(step+1)

    loss = loss / step
    print('Loss:[{:.4f}]'.format(loss.item()))
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1
    if cnt_wait == patience:
        print('Early stopping!')
        break
    loss.backward()
    optimiser.step()
    print('Loading {}th epoch'.format(best_t))
