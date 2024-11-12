import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self, in_features, out_features, num_heads=1, dropout=0.0, alpha=0.2):
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.dropout = dropout

        # Define the weight matrices for each attention head
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features * num_heads)))
        nn.init.xavier_uniform_(self.W.data)

        # Define the attention mechanism parameters
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data)

        self.leaky_relu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj):

        N = x.size(0)

        # Linear transformation
        h = torch.mm(x, self.W).view(N, self.num_heads, self.out_features)

        # Compute attention coefficients
        a_input = torch.cat([h.repeat(1, 1, N).view(N*N*self.num_heads, -1), 
                             h.repeat(1, N, 1).view(N*N*self.num_heads, -1)], dim=1)
        e = self.leaky_relu(torch.matmul(a_input, self.a).squeeze(1).view(-1, N*N, self.num_heads))

        # Masked softmax to handle the masked nodes in the adjacency matrix
        zero_vec = -9e15*torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = self.dropout(attention)

        # Apply attention to the input features
        h_prime = torch.matmul(attention, h.permute(1, 0, 2))
        h_prime = h_prime.permute(1, 0, 2).contiguous().view(N, -1)

        return h_prime