import torch
import torch.nn as nn
import torch.nn.functional as F


class downprompt(nn.Module):
    def __init__(self, prompt1, prompt2, prompt3, ft_in, nb_classes, feature, labels):
        super(downprompt, self).__init__()
        for m in self.modules():
            self.weights_init(m)

        self.labels = labels
        self.downprompt = downstreamprompt(ft_in)
        self.leakyrelu = nn.ELU()
        self.prompt = torch.cat((prompt1, prompt2, prompt3), 0)
        self.nodelabelprompt = weighted_prompt(3)

        self.dffprompt = weighted_feature(2)

        feature = feature.squeeze().cuda()

        self.one = torch.ones(1,ft_in).cuda()

        self.ave = averageemb(labels=self.labels, rawret=feature)
        
    def forward(self, seq, train=0):
        rawret = self.downprompt(seq)

        # rawret = seq
        rawret = rawret.cuda()
        # rawret = torch.stack((rawret,rawret,rawret,rawret,rawret,rawret))
        if train == 1:
            self.ave = averageemb(labels=self.labels, rawret=rawret)


        ret = torch.FloatTensor(seq.shape[0],3).cuda()
        # print("avesize",self.ave.size(),"ave",self.ave)
        # print("rawret=", rawret[1])
        # print("aveemb", self.ave)
        for x in range(0,seq.shape[0]):
            ret[x][0] = torch.cosine_similarity(rawret[x], self.ave[0], dim=0)
            ret[x][1] = torch.cosine_similarity(rawret[x], self.ave[1], dim=0)
            ret[x][2] = torch.cosine_similarity(rawret[x], self.ave[2], dim=0)

        ret = F.softmax(ret, dim=1)

        # probability: (batch_size, num_class)
        return ret

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)




def averageemb(labels,rawret):
    #print("rawret",rawret.shape)
    retlabel = torch.FloatTensor(3,int(rawret.shape[0]/2),int(rawret.shape[1])).cuda()

    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    #print("labels",labels.shape)
    #print("retlabel",retlabel.shape)
    for x in range(0,rawret.shape[0]):
        if labels[x].item() == 0:
            retlabel[0][cnt1] = rawret[x]
            cnt1 = cnt1 + 1
        if labels[x].item() == 1:
            retlabel[1][cnt2]= rawret[x]
            cnt2 = cnt2 + 1
        if labels[x].item() == 2:
            retlabel[2][cnt3] = rawret[x]
            cnt3 = cnt3 + 1
    retlabel = torch.mean(retlabel,dim=1)
    return retlabel

class weighted_prompt(nn.Module):
    def __init__(self,weightednum):
        super(weighted_prompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(0.9)
        self.weight[0][1].data.fill_(0.9)
        self.weight[0][2].data.fill_(0.1)
    def forward(self, graph_embedding):
        # print("weight",self.weight)
        graph_embedding=torch.mm(self.weight,graph_embedding)
        return graph_embedding
    



class weighted_feature(nn.Module):
    def __init__(self,weightednum):
        super(weighted_feature, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,weightednum), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        # torch.nn.init.xavier_uniform_(self.weight)

        self.weight[0][0].data.fill_(1)
        self.weight[0][1].data.fill_(0)
    def forward(self, graph_embedding1,graph_embedding2):
        # print("weight",self.weight)
        graph_embedding= self.weight[0][0] * graph_embedding1 + self.weight[0][1] * graph_embedding2
        return self.act(graph_embedding)
    

class downstreamprompt(nn.Module):
    def __init__(self,hid_units):
        super(downstreamprompt, self).__init__()
        self.act = nn.ELU()
        self.weight= nn.Parameter(torch.FloatTensor(1,hid_units), requires_grad=True)
        self.reset_parameters()
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, graph_embedding):
        graph_embedding=self.weight * graph_embedding

        return self.act(graph_embedding)