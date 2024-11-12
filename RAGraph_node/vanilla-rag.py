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

parser = argparse.ArgumentParser("RAGraph")

parser.add_argument('--dataset', type=str, default="ENZYMES", help='data')
parser.add_argument('--seed', type=int, default=39, help='seed')

args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'
seed_everything(args.seed)

print('-' * 100)
print(args)
print('-' * 100)

# training params
batch_size = 256
hid_units = 256
nonlinearity = 'prelu'  # special name to separate parameters

dataset = TUDataset(root='data', name=args.dataset, use_node_attr=True) 

print("dataset", dataset)
feature_size = dataset.num_node_attributes
nb_classes = 3

pretrain_model = PrePrompt(feature_size, hid_units, nonlinearity, 1, 0.3)
pretrain_model.load_state_dict(torch.load(args.save_name))
pretrain_model = pretrain_model.cuda()

# shotnum = 5
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
                        feture_size=feature_size, num_class=nb_classes, emb_size=hid_units,
                        finetune=False).cuda()

    print('-' * 100)

    print("tasknum:", i+1)
    # fewshot_adj: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodeadj.pt").cuda()
    # fewshot_feature: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodeemb.pt").cuda()
    # fewshot_label: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodelabels.pt").type(torch.long).squeeze().cuda()
    
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        features, adj, node_labels= process_tu_dataset(data, feature_size)
        
        logits = rag_model(features, adj)

        predict_labels = torch.argmax(logits, dim=1)
        node_labels = torch.argmax(node_labels, dim=1)

        correct += torch.sum(predict_labels == node_labels).item()
        total += len(node_labels)
    accuracy = 100 * correct / total
    print("accuracy: {:.4f}".format(accuracy))
    accuracy_list.append(accuracy) 

print('-' * 100)
# print("shotnum",shotnum)
accs = np.array(accuracy_list)
mean_acc = accs.mean()
std_acc = accs.std()
print('Mean:[{:.4f}]'.format(mean_acc))
print('Std :[{:.4f}]'.format(std_acc))
print('-' * 100)

os.makedirs("results", exist_ok=True)
with open(f"results/vanilla_rag_{args.dataset}.json", "w") as f:
    json.dump({
        "mean": mean_acc,
        "std": std_acc,
        "accuracy": accuracy_list
    }, f, indent=4)
