import os
import json
import torch
import argparse
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from preprompt import PrePrompt
from RAGraph import RAGraph
from ragraph_utils import seed_everything, process_tu_dataset
from ragraph_utils import fewshot_mean_logits, fewshot_predict_labels_by_mean

parser = argparse.ArgumentParser("RAGraph")

parser.add_argument('--dataset', type=str, default="ENZYMES", help='data')

args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'
args.val_name = f'modelset/noval_rag_{args.dataset}.pkl'

print('-' * 100)
print(args)
print('-' * 100)

# training params
batch_size = 1 # 8
nb_epochs = 1000
patience = 100
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
sparse = False
useMLP = False
LP = False
nonlinearity = 'prelu'  # special name to separate parameters

dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True)
feature_size = dataset.num_node_attributes
num_classes = dataset.num_classes

pretrain_model = PrePrompt(feature_size, hid_units, nonlinearity, 2, 0.3)
pretrain_model.load_state_dict(torch.load(args.save_name))
pretrain_model = pretrain_model.cuda()


fewshot_adj: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}_graph/testset/adj.pt").cuda()
fewshot_feature: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}_graph/testset/feature.pt").cuda()
fewshot_labels: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}_graph/testset/labels.pt").cuda()
fewshot_graph_len = torch.load(f"data/fewshot_{args.dataset}_graph/testset/graph_len.pt").cuda()

print("fewshot_adj", fewshot_adj.shape)
print("fewshot_feature", fewshot_feature.shape)
print("fewshot_labels", fewshot_labels.shape)
print("fewshot_graph_len", fewshot_graph_len.shape)

fewshot_node_logits: torch.Tensor = pretrain_model.inference(fewshot_feature, fewshot_adj)
print("fewshot_node_logits", fewshot_node_logits.shape)

graph_start_index = 0
fewshot_logits = torch.zeros(fewshot_labels.shape[0], hid_units).cuda()
for i, graph_len in enumerate(fewshot_graph_len):
    graph_end_index = graph_start_index + graph_len.int().item()
    fewshot_logits[i] = fewshot_node_logits[graph_start_index:graph_end_index].mean(0)
    graph_start_index = graph_end_index

mean_fewshot_logits = fewshot_mean_logits(fewshot_logits, fewshot_labels)


shotnum = 5
test_times = 5
accuracy_list = []
for i in range(test_times):
    seed_everything(i)
    print('-' * 100)

    # shuffle database
    dataset = dataset.shuffle()
    train_val_dataset = dataset[:int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]

    rag_model = RAGraph(pretrain_model, resource_dataset=train_val_dataset,
                        feture_size=feature_size, num_class=num_classes, emb_size=hid_units,
                        finetune=False).cuda()

    print('-' * 100)

    print("tasknum:", i+1)
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        features, adj = process_tu_dataset(data, num_classes, feature_size)
        
        logits = rag_model(features, adj, mean_fewshot_logits)
        predict_label = fewshot_predict_labels_by_mean(mean_fewshot_logits, logits)
        graph_label = data.y.cuda()

        # print(predict_label)
        # print(graph_label)

        correct += torch.sum(predict_label == graph_label).item()
        total += len(predict_label)
    accuracy = 100 * correct / total
    print("accuracy: {:.4f}".format(accuracy))
    accuracy_list.append(accuracy) 

print('-' * 100)
print("shotnum",shotnum)
accs = np.array(accuracy_list)
mean_acc = accs.mean()
std_acc = accs.std()
print('Mean:[{:.4f}]'.format(mean_acc))
print('Std :[{:.4f}]'.format(std_acc))
print('-' * 100)

os.makedirs("results", exist_ok=True)
with open(f"results/vanilla_rag_{args.dataset}_shot{shotnum}.json", "w") as f:
    json.dump({
        "mean": mean_acc,
        "std": std_acc,
        "accuracy": accuracy_list
    }, f, indent=4)
