import os
import json
import torch
import argparse
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

from preprompt import PrePrompt
from RAGraph import RAGraph
from ragraph_utils import seed_everything, process_tu_dataset
from ragraph_utils import fewshot_mean_logits, fewshot_predict_labels_by_mean, fewshot_predict_logits


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
lr = 0.0001
l2_coef = 0.0
drop_prob = 0.0
hid_units = 256
nonlinearity = 'prelu'  # special name to separate parameters

downstream_lr = 0.0001
downstream_epochs = 50
patience = 10

dataset_name = args.dataset
dataset = TUDataset(root='data', name=dataset_name, use_node_attr=True)
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

def calculate_mean_logits(pretrain_model):
    fewshot_node_logits: torch.Tensor = pretrain_model.inference(fewshot_feature, fewshot_adj)
    # print("fewshot_node_logits", fewshot_node_logits.shape)

    graph_start_index = 0
    fewshot_logits = torch.zeros(fewshot_labels.shape[0], hid_units).cuda()
    for i, graph_len in enumerate(fewshot_graph_len):
        graph_end_index = graph_start_index + graph_len.int().item()

        # 请仔细检查下面这行代码
        fewshot_logits[i, :] = fewshot_node_logits[graph_start_index:graph_end_index].mean(0)
        graph_start_index = graph_end_index

    mean_fewshot_logits = fewshot_mean_logits(fewshot_logits, fewshot_labels)
    return mean_fewshot_logits

shotnum = 5
test_times = 5
accuracy_list = []
for i in range(test_times):
    seed_everything(i)
    print('-' * 100)

    # shuffle database
    dataset = dataset.shuffle()
    train_dataset = dataset[:int(0.5 * len(dataset))]
    val_dataset = dataset[int(0.5 * len(dataset)):int(0.8 * len(dataset))]
    test_dataset = dataset[int(0.8 * len(dataset)):]

    rag_model = RAGraph(pretrain_model, resource_dataset=train_dataset,
                        feture_size=feature_size, num_class=num_classes, emb_size=hid_units,
                        finetune=True, noise_finetune=False).cuda()

    print('-' * 100)

    print("tasknum:", i+1)
    
    # finetune
    rag_model.train()
    pretrain_model.gcn.convs[0].eval()
    best_loss = float('inf')
    finetune_model_name = f"modelset/finetune_rag_model_{args.dataset}_shot{shotnum}_{i}.pkl"
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(rag_model.parameters(), lr=downstream_lr)
    
    best_epoch = 0
    trigger_times = 0
    for epoch in range(downstream_epochs):
        total_loss = 0
        for data in tqdm(val_loader, desc=f'epoch {epoch}', ncols=80):
            features, adj = process_tu_dataset(data, num_classes, feature_size)
        
            opt.zero_grad()

            mean_fewshot_logits = calculate_mean_logits(pretrain_model)
            # print("mean_fewshot_logits", mean_fewshot_logits.shape)

            logits = rag_model(features, adj, mean_fewshot_logits)
            predict_logits = fewshot_predict_logits(mean_fewshot_logits, logits)

            graph_label = data.y
            graph_label = torch.nn.functional.one_hot(graph_label, num_classes=num_classes).float().cuda()

            # print("predict_logits", predict_logits.shape)
            # print("graph_label", graph_label.shape)

            def orthogonal_loss(mean_fewshot_logits):
                loss = 0.0
                num_classes = mean_fewshot_logits.size(0)

                # 计算不同类别之间的内积
                for i in range(num_classes):
                    for j in range(i + 1, num_classes):
                        inner_product = torch.dot(mean_fewshot_logits[i], mean_fewshot_logits[j])
                        loss += inner_product ** 2  # 最小化内积的平方

                return loss

            # if dataset_name == 'BZR':
            #     loss = torch.nn.functional.cross_entropy(predict_logits, graph_label) + 0.0005 * orthogonal_loss(mean_fewshot_logits)
            # else:
            #     loss = torch.nn.functional.cross_entropy(predict_logits, graph_label)
            loss = torch.nn.functional.cross_entropy(predict_logits, graph_label)

            total_loss += loss.item()
            loss.backward()
            opt.step()
            
        epoch_loss = total_loss / len(val_loader)
        print("epoch:", epoch, "loss:", epoch_loss)
        if epoch_loss < best_loss:
            best_epoch = epoch
            best_loss = epoch_loss
            trigger_times = 0
            torch.save(rag_model.state_dict(), finetune_model_name)
        else:
            trigger_times += 1
            if trigger_times >= patience:
                print(f'early stop at epoch: {epoch}!')
                break

    # inference
    print('-' * 100)
    print("best_epoch:", best_epoch)
    print("best_loss:", best_loss)
    rag_model.load_state_dict(torch.load(finetune_model_name))
    rag_model.eval()
    rag_model.toy_graph_base.build_toy_graph(val_dataset)
    rag_model.toy_graph_base.show()

    mean_fewshot_logits = calculate_mean_logits(pretrain_model)

    correct = 0
    total = 0
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    for data in test_loader:
        features, adj = process_tu_dataset(data, num_classes, feature_size)
        
        logits = rag_model(features, adj, mean_fewshot_logits)
        # print(logits)

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
with open(f"results/finetune_rag_{args.dataset}_shot{shotnum}.json", "w") as f:
    json.dump({
        "mean": mean_acc,
        "std": std_acc,
        "accuracy": accuracy_list
    }, f, indent=4)
