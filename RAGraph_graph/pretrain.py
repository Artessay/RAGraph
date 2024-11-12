import numpy as np
import scipy.sparse as sp
import random
from preprompt import PrePrompt
import preprompt
from utils import process
import aug
import os
import argparse
from downprompt import downprompt,averageemb,predict
import csv
parser = argparse.ArgumentParser("My DGI")

parser.add_argument('--dataset', type=str, default="ENZYMES", help='data')
parser.add_argument('--aug_type', type=str, default="edge", help='aug type: mask or edge')
parser.add_argument('--drop_percent', type=float, default=0.1, help='drop percent')
parser.add_argument('--seed', type=int, default=39, help='seed')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
args = parser.parse_args()
args.save_name = f'modelset/model_{args.dataset}.pkl'
args.val_name = f'modelset/noval_graphcl_{args.dataset}.pkl'

# world_size = torch.cuda.device_count()
# local_rank = args.local_rank
# dist_backend = 'nccl'
# dist.init_process_group(backend=dist_backend)
print('-' * 100)
print(args)
print('-' * 100)

dataset = args.dataset
aug_type = args.aug_type
drop_percent = args.drop_percent
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu) 
seed = args.seed
random.seed(seed)
np.random.seed(seed)

import torch
import torch.nn as nn
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
# training params

batch_size = 8
nb_epochs = 10 # 1000 # 5
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

loader = DataLoader(dataset,batch_size=batch_size,shuffle=True,drop_last=True)

ft_size = dataset.num_node_attributes
class_num = dataset.num_classes

model = PrePrompt(ft_size, hid_units, nonlinearity,1,0.3)
model = model.cuda()

best = 1e9


for epoch in range(nb_epochs):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    loss = 0
    regloss = 0
    for step, data in enumerate(loader):

        features,adj =  process.process_tu(data,class_num,ft_size)

        negetive_sample = preprompt.prompt_pretrain_sample(adj,50)


        nb_nodes = features.shape[0]  # node number

        features = torch.FloatTensor(features[np.newaxis])

        '''
        # ------------------------------------------------------------
        # edge node mask subgraph
        # ------------------------------------------------------------
        '''
        # print("Begin Aug:[{}]".format(args.aug_type))
        # if args.aug_type == 'edge':

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

        # aug_adj1mask = process.normalize_adj(aug_adj1mask + sp.eye(aug_adj1mask.shape[0]))
        # aug_adj2mask = process.normalize_adj(aug_adj2mask + sp.eye(aug_adj2mask.shape[0]))

        if sparse:
            sp_adj = process.sparse_mx_to_torch_sparse_tensor(adj)
            sp_aug_adj1edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj1edge)
            sp_aug_adj2edge = process.sparse_mx_to_torch_sparse_tensor(aug_adj2edge)

            # sp_aug_adj1mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj1mask)
            # sp_aug_adj2mask = process.sparse_mx_to_torch_sparse_tensor(aug_adj2mask)

        else:
            adj = (adj + sp.eye(adj.shape[0])).todense()
            aug_adj1edge = (aug_adj1edge + sp.eye(aug_adj1edge.shape[0])).todense()
            aug_adj2edge = (aug_adj2edge + sp.eye(aug_adj2edge.shape[0])).todense()

            # aug_adj1mask = (aug_adj1mask + sp.eye(aug_adj1mask.shape[0])).todense()
            # aug_adj2mask = (aug_adj2mask + sp.eye(aug_adj2mask.shape[0])).todense()

        # '''
        # ------------------------------------------------------------
        # mask
        # ------------------------------------------------------------
        '''

        # '''
        # ------------------------------------------------------------
        if not sparse:
            adj = torch.FloatTensor(adj[np.newaxis])
            aug_adj1edge = torch.FloatTensor(aug_adj1edge[np.newaxis])
            aug_adj2edge = torch.FloatTensor(aug_adj2edge[np.newaxis])

        optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2_coef)
        if torch.cuda.is_available() and step==0:
            print('Using CUDA')
            # model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
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
        b_xent = nn.BCEWithLogitsLoss()
        xent = nn.CrossEntropyLoss()
        # cnt_wait = 0
        # # best = 1e9
        # best_t = 0

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
    loss = loss / (step+1)
    print('Loss:[{:.4f}]'.format(loss.item()))
    if loss < best:
        best = loss
        best_t = epoch
        cnt_wait = 0
        torch.save(model.state_dict(), args.save_name)
    else:
        cnt_wait += 1
        print("cnt_wait",cnt_wait)

    if cnt_wait == patience:
        print('Early stopping!')
        break
    loss.backward()
    optimiser.step()
    print('Loading {}th epoch'.format(best_t))




model.load_state_dict(torch.load(args.save_name))

dgiprompt = model.dgi.prompt
graphcledgeprompt = model.graphcledge.prompt
lpprompt = model.lp.prompt

list = [5]
downlrlist = [0.1]

for downlr in downlrlist:

    for shotnum in list:
        tot = torch.zeros(1)
        tot = tot.cuda()
        accs = []
        for seed in range(36,37):
            test_adj = torch.load(
                f"data/fewshot_{args.dataset}_graph/testset/adj.pt").cuda()
            testfeature = torch.load(
                f"data/fewshot_{args.dataset}_graph/testset/feature.pt").cuda()
            testlbls = torch.load(
                f"data/fewshot_{args.dataset}_graph/testset/labels.pt").cuda()
            testgraphlen = torch.load(
                f"data/fewshot_{args.dataset}_graph/testset/graph_len.pt").cuda()
            # print("lbls", lbls)
            testfeature, _ = model.embed(testfeature, test_adj, sparse, None, LP)
            testfeature = testfeature.squeeze()

            print('-' * 100)
            print("test_lbls", testlbls.shape)
            print("testlbl", testlbls)

            b_xent = nn.BCEWithLogitsLoss()
            xent = nn.CrossEntropyLoss()

            cnt_wait = 0
            
            best_t = 0
            test_times = 100
            #test_times = 1
            for i in range(test_times):
                np.random.seed(seed)
                torch.manual_seed(seed)
                torch.cuda.manual_seed(seed)
                print("tasknum",i)
                best = 1e9
                cnt_wait = 0
                # pretrain_lbls = torch.LongTensor(shotnum * class_num, ).cuda()
                # pretrain_embs = torch.FloatTensor(shotnum * class_num, hid_units).cuda()
                print("downlr",downlr)
                pretrain_adj = torch.load(
                    f"data/fewshot_{args.dataset}_graph/{shotnum}shot_{args.dataset}_graph/{i}/adj.pt")
                prefeature = torch.load(
                    f"data/fewshot_{args.dataset}_graph/{shotnum}shot_{args.dataset}_graph/{i}/feature.pt")
                lbls = torch.load(
                    f"data/fewshot_{args.dataset}_graph/{shotnum}shot_{args.dataset}_graph/{i}/labels.pt").cuda()
                prelbls = lbls.cpu().numpy()
                prelbls = torch.LongTensor(prelbls).cuda()
                graphlen = torch.load(
                    f"data/fewshot_{args.dataset}_graph/{shotnum}shot_{args.dataset}_graph/{i}/graph_len.pt")
                prefeature = torch.FloatTensor(prefeature).cuda()
                pretrain_adj = torch.FloatTensor(pretrain_adj).cuda()
                pretrain_embs, _ = model.embed(prefeature, pretrain_adj, sparse, None, LP)
                # tmp_embs = prompt * tmp_embs
                pretrain_embs = pretrain_embs.squeeze()
                print("true", lbls)
                # print("pretrain_emb",pretrain_embs)

                log = downprompt(dgiprompt, graphcledgeprompt, lpprompt, hid_units, class_num)
                # opt = torch.optim.Adam(log.parameters(),downstreamprompt.parameters(),lr=0.01, weight_decay=0.0)
                opt = torch.optim.Adam(log.parameters(), lr=downlr)
                log.cuda()
                pat_steps = 0
                best_acc = torch.zeros(1)
                best_acc = best_acc.cuda()
                for epoch_num in range(300):
                    log.train()
                    opt.zero_grad()
                    train_embs = log(pretrain_embs,graphlen)
                    c_embedding = averageemb(lbls, train_embs, class_num)
                    logits = predict(shotnum * class_num,class_num,train_embs,c_embedding).cuda()

                    loss = xent(logits, prelbls)
                    if loss < best:
                        best = loss
                        torch.save(log.state_dict(), args.val_name)
                        best_t = epoch_num
                        cnt_wait =0 
                    else:
                        cnt_wait = cnt_wait+1
                    
                    if cnt_wait == patience:
                        cnt_wait = 0
                        break
                    loss.backward()
                    opt.step()

                print("best epoch",best_t)

                log.load_state_dict(torch.load(args.val_name))
                # log.eval()
                train_embs = log(pretrain_embs,graphlen)
                c_embedding = averageemb(lbls,train_embs,class_num)
                test_embs = log(testfeature,testgraphlen)
                # c_embedding = averageemb(test_lbls,test_embs,class_num)
                logits = predict(testlbls.shape[0],class_num,test_embs,c_embedding)
                # print("logits",logits)
                # print(log.a)
                preds = torch.argmax(logits, dim=1)
                print("preds",preds)
                # print("test_lbls",test_lbls)
                acc = torch.sum(preds == testlbls).float() / testlbls.shape[0]
                accs.append(acc * 100)
                # print('acc:[{:.4f}]'.format(acc)
                
                tot += acc
                print("current total acc",tot/(i+1))

        print('-' * 100)

        print("downlr",downlr)
        print('Average accuracy:[{:.4f}]'.format(tot.item() / test_times))
        accs = torch.stack(accs)
        print('Mean:[{:.4f}]'.format(accs.mean().item()))
        print('Std :[{:.4f}]'.format(accs.std().item()))
        print('-' * 100)
        row = [args.save_name,downlr,shotnum, accs.mean().item(),accs.std().item()]
        out = open(f"data/model_{args.dataset}_val.csv", "a", newline="")
        csv_writer = csv.writer(out, dialect="excel")
        csv_writer.writerow(row)
        # print("test_lablels", test_lbls)