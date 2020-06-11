import torch
from builtins import super
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
"""
Forward model in the cycle loss.

Input: s_t, a_t
Output: s_{t+1}
"""

class Fwd_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 96)
        self.fc2 = nn.Linear(96, 256)
        self.fc3 = nn.Linear(256, 384)
        self.fc4 = nn.Linear(384, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, 200)
        
        self.fc1_1 = nn.Linear(3, 24)
        self.fc2_1 = nn.Linear(24, 48)
        self.fc3_1 = nn.Linear(48, 100)

        # Did not implement dim reduction via Bayes because the dim is low in this vanilla case.
        self.fc_out_1 = nn.Linear(300, 100)
    def forward(self, x1, x2):
        # Stream 1
        x1 = F.elu(self.fc1(x1))
        x1 = F.elu(self.fc2(x1))
        x1 = F.elu(self.fc3(x1))
        x1 = F.elu(self.fc4(x1))
        x1 = F.elu(self.fc5(x1))
        latent1 = F.elu(self.fc6(x1))
        # Stream 2
        x2 = F.elu(self.fc1_1(x2))
        x2 = F.elu(self.fc2_1(x2))
        latent2 = F.elu(self.fc3_1(x2))
        # Concatenate the output latent tensors
        # TODO: Verify the dimensions. Assuming the batch size dimension is 0
        latent = torch.cat((latent1, latent2), 1)
        out = self.fc_out_1(latent)
        return out

