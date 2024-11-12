import torch
import torch.nn as nn
import torch.nn.functional as F


class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3, ft_in, nb_classes):
        super(downprompt, self).__init__()
        
        self.downprompt = downstreamprompt(ft_in)
        
        self.nb_classes = nb_classes

        self.leakyrelu = nn.ELU()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)

        self.nodelabelprompt = weighted_prompt(3)

        self.dffprompt = weighted_feature(2)

    def forward(self, seq, graph_len):
        rawret2 = self.downprompt(seq)

        #rawret = self.dffprompt(rawret1 ,rawret2)
        rawret = rawret2
        # rawret = seq
        rawret = rawret.cuda()
        
        graph_embedding=split_and_batchify_graph_feats(rawret, graph_len)

        return graph_embedding
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)



def predict(graphnum,nb_classes,rawret,ave):
    # print(graphnum)
    ret = torch.FloatTensor(graphnum, nb_classes).cuda()
    
    for x in range(0, graphnum):
        ret[x][0] = torch.cosine_similarity(rawret[x], ave[0], dim=0)
        ret[x][1] = torch.cosine_similarity(rawret[x], ave[1], dim=0)
        if nb_classes == 6:
            ret[x][2] = torch.cosine_similarity(rawret[x], ave[2], dim=0)
            ret[x][3] = torch.cosine_similarity(rawret[x], ave[3], dim=0)
            ret[x][4] = torch.cosine_similarity(rawret[x], ave[4], dim=0)
            ret[x][5] = torch.cosine_similarity(rawret[x], ave[5], dim=0)
    
    ret = F.log_softmax(ret, dim=1)
    # print("retshape2",ret.shape)
    return ret


def averageemb(labels, rawret, nb_class):
    retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0]), int(rawret.shape[1])).cuda()
    # retlabel = torch.FloatTensor(nb_class, int(rawret.shape[0] / nb_class), int(rawret.shape[1])).cuda()
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    # print("labels",retlabel.shape)
    # print("rawret",rawret.shape)
    for x in range(0, rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2] = rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
        if labels[x].item() == 3:
            retlabel[3][cnt4] = rawret[x]
            cnt4 = cnt4 + 1
        if labels[x].item() == 4:
            retlabel[4][cnt5] = rawret[x]
            cnt5 = cnt5 + 1
        if labels[x].item() == 5:
            retlabel[5][cnt6] = rawret[x]
            cnt6 = cnt6 + 1
        if labels[x].item() == 6:
            retlabel[6][cnt7] = rawret[x]
            cnt7 = cnt7 + 1
    retlabel = torch.mean(retlabel, dim=1)
    return retlabel



def split_and_batchify_graph_feats(batched_graph_feats, graph_sizes):

    cnt = 0 

    result = torch.FloatTensor(graph_sizes.shape[0], batched_graph_feats.shape[1]).cuda()

    for i in range(graph_sizes.shape[0]):
        # print("i",i)
        current_graphlen = int(graph_sizes[i].item())
        graphlen = range(cnt,cnt+current_graphlen)
        # print("graphlen",graphlen)
        result[i] = torch.sum(batched_graph_feats[graphlen,:], dim=0)
        cnt = cnt + current_graphlen
    # print("resultsum",cnt)    
    return result

class weighted_prompt(nn.Module):
    def __init__(self, weightednum):
        super(weighted_prompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.9)
        self.weight[0][1].data.fill_(0.9)
        self.weight[0][2].data.fill_(0.1)

    def forward(self, graph_embedding):
        graph_embedding = torch.mm(self.weight, graph_embedding)
        return graph_embedding


class weighted_feature(nn.Module):
    def __init__(self, weightednum):
        super(weighted_feature, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, weightednum), requires_grad=True)
        self.act = nn.ELU()
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(1)
        self.weight[0][1].data.fill_(0)

    def forward(self, graph_embedding1, graph_embedding2):
        
        # weight = F.softmax(self.weight, dim=1)
        # print("weight",weight)
        graph_embedding = self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)


class downstreamprompt(nn.Module):
    def __init__(self, hid_units):
        super(downstreamprompt, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(1, hid_units), requires_grad=True)
        self.reset_parameters()
        self.act = nn.ELU()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)


    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding = self.weight * graph_embedding
        return graph_embedding
    


def distance2center(input,center):
    n = input.size(0)
    m = input.size(1)
    k = center.size(0)
    input_power = torch.sum(input * input, dim=1, keepdim=True).expand(n, k)
    center_power = torch.sum(center * center, dim=1).expand(n, k)
    temp1=input_power+center_power
    temp2=2*torch.mm(input,center.transpose(0,1))
    distance = input_power + center_power - 2 * torch.mm(input, center.transpose(0, 1))
    return distance



def onehot(label, nb_classes):
    ret = torch.zeros(label.shape[0], nb_classes).cuda()
    for x in range(0, label.shape[0]):
        ret[x][int(label[x].item())] = 1
    return ret