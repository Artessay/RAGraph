import torch
import numpy as np
import scipy.sparse as sp

def seed_everything(seed: int):  
    import random, os
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


# Process a (subset of) a TU dataset into standard form
def process_tu_dataset(data, class_num, node_class):
    nb_graphs = data.num_graphs
    # print("len",nb_graphs)

    node_class_num=range(node_class)

    # print("data",data)
    labels = np.zeros((nb_graphs,class_num))

    # features = np.zeros((nb_graphs, nb_nodes, ft_size))
    # adjacency = np.zeros((nb_graphs, nb_nodes, nb_nodes))
    # labels = np.zeros(nb_graphs)
    # sizes = np.zeros(nb_graphs, dtype=np.int32)
    # masks = np.zeros((nb_graphs, nb_nodes))
    # zero = np.zeros((nb_nodes, nb_nodes))
    for g in range(nb_graphs):
        if g == 0:
            # sizes = data[g].x.shape[0]
            features = data[g].x[ :,node_class_num]
            
            rawlabels = data[g].y[0]
            # masks[g, :sizes[g]] = 1.0
            e_ind = data[g].edge_index
            # print("e_ind",e_ind)
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(features.shape[0], features.shape[0]))
            # print("coo",coo)
            adjacency = coo.todense()
        else:
            tmpfeature = data[g].x[ :,node_class_num]
            features = np.row_stack((features,tmpfeature))
            tmplabel = data[g].y[0]
            rawlabels = np.row_stack((rawlabels,tmplabel))
            e_ind = data[g].edge_index
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])), shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
            # print("coo",coo)
            tmpadj = coo.todense()
            zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
            tmpadj1 = np.column_stack((adjacency,zero))
            tmpadj2 = np.column_stack((zero.T,tmpadj))
            adjacency = np.row_stack((tmpadj1,tmpadj2))

    for x in range(nb_graphs):
        if nb_graphs == 1:
            labels[0][rawlabels.item()]=1
            break
        labels[x][rawlabels[x][0]] = 1
    
    adj = sp.csr_matrix(adjacency)

    # postprocess
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(adj).cuda()

    return features, adj