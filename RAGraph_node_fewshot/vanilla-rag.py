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

parser.add_argument('--dataset', type=str, default="PROTEINS", help='data')
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

pretrain_model = PrePrompt(feature_size, hid_units, nonlinearity, 2, 0.3)
pretrain_model.load_state_dict(torch.load(args.save_name))
pretrain_model = pretrain_model.cuda()

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

    fewshot_adj: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodeadj.pt").cuda()
    fewshot_feature: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodeemb.pt").cuda()
    fewshot_labels: torch.Tensor = torch.load(f"data/fewshot_{args.dataset}/{shotnum}shot_{args.dataset}/{i}/nodelabels.pt").squeeze().cuda()
    
    fewshot_logits = pretrain_model.inference(fewshot_feature, fewshot_adj)
    mean_fewshot_logits = fewshot_mean_logits(fewshot_logits, fewshot_labels)

    rag_model = RAGraph(pretrain_model, resource_dataset=train_val_dataset,
                        mean_fewshot_logits=mean_fewshot_logits, emb_size=hid_units,
                        finetune=False).cuda()

    print('-' * 100)

    print("tasknum:", i+1)
    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        features, adj, node_labels= process_tu_dataset(data, feature_size)

        logits = rag_model(features, adj, mean_fewshot_logits)
        predict_labels = fewshot_predict_labels_by_mean(mean_fewshot_logits, logits)

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
with open(f"results/vanilla_rag_{args.dataset}_shot{shotnum}.json", "w") as f:
    json.dump({
        "mean": mean_acc,
        "std": std_acc,
        "accuracy": accuracy_list
    }, f, indent=4)
