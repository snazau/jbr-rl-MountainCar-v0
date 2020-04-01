import torch
from torch import nn
import torch.nn.functional as F


input_nodes = 2
output_nodes = 3


class DQN(nn.Module):
    def __init__(self, hidden_size_1, hidden_size_2):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_nodes, hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2, output_nodes)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.layer3(x)
        return x
