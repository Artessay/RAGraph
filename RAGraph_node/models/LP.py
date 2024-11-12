import torch
import torch.nn as nn

class Lp(nn.Module):
    def __init__(self, n_in, n_h):
        super(Lp, self).__init__()
        self.sigm = nn.ELU()
        self.prompt = nn.Parameter(torch.FloatTensor(1, n_h), requires_grad=True)

        self.reset_parameters()



    def forward(self,gcn,seq,adj,sparse):
        h_1 = gcn(seq,adj,sparse,True)
        ret = self.sigm(h_1.squeeze(dim=0))
        return ret

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.prompt)
