import torch
import random
import argparse

parser = argparse.ArgumentParser("RAGraph")
parser.add_argument('--dataset', type=str, default="ENZYMES", help='data')
parser.add_argument('--seed', type=int, default=2024, help='seed')
parser.add_argument('--k_shot', type=int, default=5, help='k-shot for val and test sets')
args = parser.parse_args()

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

# Set random seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)

dataset = TUDataset(root='data', name=args.dataset,use_node_attr=True)
# train_dataset = dataset[:int(0.5 * len(dataset))]
# val_dataset = dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))]
# test_dataset = dataset[int(0.8 * len(dataset)):]

# Function to get k-shot nodes from each class
def get_k_shot_nodes(dataset, k):
    class_to_nodes = {}
    for graph in dataset:
        labels = graph.y.numpy()
        for idx, label in enumerate(labels):
            if label not in class_to_nodes:
                class_to_nodes[label] = []
            class_to_nodes[label].append(graph) # ((graph, idx))
    
    k_shot_data = []
    for label, nodes in class_to_nodes.items():
        selected_nodes = random.sample(nodes, min(k, len(nodes)))
        k_shot_data.extend(selected_nodes)
    
    return k_shot_data

# Create k-shot validation and test datasets
# val_dataset = get_k_shot_nodes(dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))], args.k_shot)
# test_dataset = get_k_shot_nodes(dataset[int(0.8 * len(dataset)):], args.k_shot)
val_dataset = dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))]
test_dataset = dataset[int(0.8 * len(dataset)):]

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

test_adj = torch.load(f"data/fewshot_{args.dataset}/5shot_{args.dataset}/testadj.pt")
testfeature = torch.load(f"data/fewshot_{args.dataset}/5shot_{args.dataset}/testemb.pt")
test_lbls = torch.load(f"data/fewshot_{args.dataset}/5shot_{args.dataset}/testlabels.pt")

print(testfeature.shape)
print(test_adj.shape)
print(test_lbls.shape)

import scipy.sparse as sp
feture_size = dataset.num_node_attributes
num_class = dataset.num_features - feture_size

import os
os.makedirs(f"data/rag_{args.dataset}", exist_ok=True)

# import numpy as np
# # Process a (subset of) a TU dataset into standard form
# def process_tu(data, class_num):
#     nb_graphs = data.num_graphs
#     ft_size = data.num_features


#     num = range(class_num)

#     labelnum=range(class_num, ft_size)
    
#     for g in range(nb_graphs):
#         if g == 0:
#             features = data[g].x[:, num]
#             rawlabels = data[g].x[:, labelnum]
#             e_ind = data[g].edge_index
#             coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
#                                 shape=(features.shape[0], features.shape[0]))
#             adjacency = coo.todense()
#         else:
#             tmpfeature = data[g].x[:, num]
#             features = np.row_stack((features, tmpfeature))
#             tmplabel = data[g].x[:, labelnum]
#             rawlabels = np.row_stack((rawlabels, tmplabel))
#             e_ind = data[g].edge_index
#             coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
#                                 shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
#             # print("coo",coo)
#             tmpadj = coo.todense()
#             zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
#             tmpadj1 = np.column_stack((adjacency, zero))
#             tmpadj2 = np.column_stack((zero.T, tmpadj))
#             adjacency = np.row_stack((tmpadj1, tmpadj2))


#     nodelabels =rawlabels
#     adj = sp.csr_matrix(adjacency)

#     return features, adj, nodelabels


# def normalize_adj(adj):
#     """Symmetrically normalize adjacency matrix."""
#     adj = sp.coo_matrix(adj)
#     rowsum = np.array(adj.sum(1))
#     d_inv_sqrt = np.power(rowsum, -0.5).flatten()
#     d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
#     d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
#     return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

from utils.process import process_tu, normalize_adj

def save_data(loader, split_name):
    for data in loader:
        features, adj, nodelabels = process_tu(data, feture_size)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()

        features = torch.FloatTensor(features)
        adj = torch.FloatTensor(adj)
        nodelabels = torch.FloatTensor(nodelabels)
        nodelabels = torch.argmax(nodelabels, dim=1)

        torch.save(features, f"data/rag_{args.dataset}/{split_name}emb.pt")
        torch.save(adj, f"data/rag_{args.dataset}/{split_name}adj.pt")
        torch.save(nodelabels, f"data/rag_{args.dataset}/{split_name}labels.pt")

        print(features.shape)
        print(adj.shape)
        print(nodelabels.shape)

print()
print(len(val_dataset))
save_data(val_loader, "val")

print()
print(len(test_dataset))
save_data(test_loader, "test")