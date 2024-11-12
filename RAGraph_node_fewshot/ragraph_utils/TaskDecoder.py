import torch.nn as nn

class TaskDecoder(nn.Module):
    def __init__(self, input_dim, hiddden_dim, output_dim):
        super(TaskDecoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hiddden_dim)
        self.act = nn.LeakyReLU()
        self.fc2 = nn.Linear(hiddden_dim, output_dim)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return x