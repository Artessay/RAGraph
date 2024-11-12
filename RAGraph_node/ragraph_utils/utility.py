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
def process_tu_dataset(data, num_node_attributes):
    nb_graphs = data.num_graphs
    ft_size = data.num_features

    num = range(num_node_attributes)

    labelnum=range(num_node_attributes, ft_size)
    
    for g in range(nb_graphs):
        if g == 0:
            features = data[g].x[:, num]
            rawlabels = data[g].x[:, labelnum]
            e_ind = data[g].edge_index
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(features.shape[0], features.shape[0]))
            adjacency = coo.todense()
        else:
            tmpfeature = data[g].x[:, num]
            features = np.row_stack((features, tmpfeature))
            tmplabel = data[g].x[:, labelnum]
            rawlabels = np.row_stack((rawlabels, tmplabel))
            e_ind = data[g].edge_index
            coo = sp.coo_matrix((np.ones(e_ind.shape[1]), (e_ind[0, :], e_ind[1, :])),
                                shape=(tmpfeature.shape[0], tmpfeature.shape[0]))
            # print("coo",coo)
            tmpadj = coo.todense()
            zero = np.zeros((adjacency.shape[0], tmpfeature.shape[0]))
            tmpadj1 = np.column_stack((adjacency, zero))
            tmpadj2 = np.column_stack((zero.T, tmpadj))
            adjacency = np.row_stack((tmpadj1, tmpadj2))


    node_labels =rawlabels
    adj = sp.csr_matrix(adjacency)

    # postprocess
    adj = normalize_adj(adj + sp.eye(adj.shape[0])).todense()
    features = torch.FloatTensor(features).cuda()
    adj = torch.FloatTensor(adj).cuda()
    node_labels = torch.FloatTensor(node_labels).cuda()


    return features, adj, node_labels