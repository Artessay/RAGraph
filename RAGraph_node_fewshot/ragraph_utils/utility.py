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


def fewshot_mean(
        fewshot_logits: torch.Tensor, 
        fewshot_labels: torch.Tensor, 
):
    # 计算每个标签的平均 logits
    unique_labels = fewshot_labels.unique()
    mean_fewshot_logits = []

    for label in unique_labels:
        # 选择对应于当前标签的 logits
        mask = (fewshot_labels == label)
        mean_logits = fewshot_logits[mask].mean(dim=0)
        mean_fewshot_logits.append(mean_logits)

    # 将平均 logits 转换为张量
    mean_fewshot_logits = torch.stack(mean_fewshot_logits)

    return mean_fewshot_logits, unique_labels
    

def fewshot_logits_map(
        fewshot_logits: torch.Tensor, 
        fewshot_labels: torch.Tensor, 
):
    mean_fewshot_logits, unique_labels = fewshot_mean(fewshot_logits, fewshot_labels)
    return {label.item(): logit for label, logit in zip(unique_labels, mean_fewshot_logits)}

def fewshot_predict_loss(
        fewshot_logits: torch.Tensor, 
        fewshot_labels: torch.Tensor, 
        logits: torch.Tensor,
        labels: torch.Tensor) -> torch.Tensor:
    label_to_logit = fewshot_logits_map(fewshot_logits, fewshot_labels)

    # 使用张量操作构建 gold_logits
    gold_logits_list = [label_to_logit[label.item()] for label in labels]
    gold_logits = torch.stack(gold_logits_list).to(logits.device)
    
    return torch.nn.functional.mse_loss(logits, gold_logits)

def fewshot_mean_logits(
        fewshot_logits: torch.Tensor, 
        fewshot_labels: torch.Tensor, 
):
    # 将 unique_labels 转换为一个字典以加快查找速度
    label_to_logit = fewshot_logits_map(fewshot_logits, fewshot_labels)
    
    mean_fewshot_logits = torch.stack([
        label_to_logit[label] for label in range(len(label_to_logit))
    ])

    return mean_fewshot_logits

def fewshot_predict_logits(
        mean_fewshot_logits: torch.Tensor, 
        logits: torch.Tensor,
) -> torch.Tensor:
    similarity = torch.nn.functional.cosine_similarity(logits.unsqueeze(1), mean_fewshot_logits.unsqueeze(0), dim=-1)
    return similarity

def fewshot_predict_labels(
        fewshot_logits: torch.Tensor, 
        fewshot_labels: torch.Tensor, 
        logits: torch.Tensor) -> torch.Tensor:
    mean_fewshot_logits, unique_labels = fewshot_mean(fewshot_logits, fewshot_labels)

    # 计算相似度（这里使用余弦相似度）
    similarity = torch.nn.functional.cosine_similarity(logits.unsqueeze(1), mean_fewshot_logits.unsqueeze(0), dim=-1)

    # 获取最相似的标签索引
    _, predicted_indices = similarity.max(dim=1)

    # 根据索引获取预测标签
    predicted_labels = unique_labels[predicted_indices]

    return predicted_labels

def fewshot_predict_labels_by_mean(
        mean_fewshot_logits: torch.Tensor, 
        logits: torch.Tensor) -> torch.Tensor:
    
    # 计算相似度（这里使用余弦相似度）
    similarity = torch.nn.functional.cosine_similarity(logits.unsqueeze(1), mean_fewshot_logits.unsqueeze(0), dim=-1)

    # 获取最相似的标签索引
    _, predicted_indices = similarity.max(dim=1)

    return predicted_indices
