import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import torch.nn.functional as F
"""
Reimplementation of the inverse model in the paper "Combining Self-Supervised Learning and Imitation for Vision-Based
Rope Manipulation".

Input: Current state represented by an image or a state tensor
       Target state represented by an image or a state tensor
Output: Predicted action that can bring the input state to the target state (inverse model)

Network Architecture: Two streams.
                      If the input data are images, then each stream: Conv96, Conv256, Conv384, Conv384, Conv256, Conv200. 
                      SHARE WEIGHTS between the two streams.

We will not use images as states (for now).
Given we have access to the ground truth rope links locations in Blender:

TODO: Better SA formulations"?

States: 90x2. 90 links in a rope, and each link has x and y.
Actions: 1x3. 1 time step, dx and dy for the HELD link position.

For details, refer to https://arxiv.org/pdf/1703.02018.pdf
"""
class Inv_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(180, 96)
        self.fc2 = nn.Linear(96, 256)
        self.fc3 = nn.Linear(256, 384)
        self.fc4 = nn.Linear(384, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, 200)

        # Did not implement dim reduction via Bayes because the dim is low in this vanilla case.
        self.fc_out_1 = nn.Linear(400, 3)
    def forward(self, x1, x2):
        # Stream 1
        x1 = F.elu(self.fc1(x1))
        x1 = F.elu(self.fc2(x1))
        x1 = F.elu(self.fc3(x1))
        x1 = F.elu(self.fc4(x1))
        x1 = F.elu(self.fc5(x1))
        latent1 = F.elu(self.fc6(x1))
        # Stream 2
        x2 = F.elu(self.fc1(x2))
        x2 = F.elu(self.fc2(x2))
        x2 = F.elu(self.fc3(x2))
        x2 = F.elu(self.fc4(x2))
        x2 = F.elu(self.fc5(x2))
        latent2 = F.elu(self.fc6(x2))
        # Concatenate the output latent tensors
        # TODO: Verify the dimensions. Assuming the batch size dimension is 0
        latent = torch.cat((latent1, latent2), 1)
        out = self.fc_out_1(latent)
        return out



    
