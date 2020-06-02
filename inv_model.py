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
        self.fc1 = nn.Linear(100, 96)
        self.fc2 = nn.Linear(96, 256)
        self.fc3 = nn.Linear(256, 384)
        self.fc4 = nn.Linear(384, 384)
        self.fc5 = nn.Linear(384, 256)
        self.fc6 = nn.Linear(256, 200)

        # Implemented dim reduction via Bayes as classification.
        # "Velocity" action output
        self.fc_out_1 = nn.Linear(400, 18)
        # X action output --- discretize [-3, 3] to an array evenly separated by 0.1
        self.fc_out_2 = nn.Linear(418, 61)
        # Y action output --- discretize [-3, 3] to an array evenly separated by 0.1
        self.fc_out_3 = nn.Linear(418+61, 61)
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
        latent = torch.cat((latent1, latent2), 1)
        out_frame = F.elu(self.fc_out_1(latent))
        softmax = nn.Softmax(dim=1)
        # Convert to a distribution
        out_frame = softmax(out_frame)
        # Concatenate the latent tensors with the first action pred
        lat_frame = torch.cat((latent, out_frame), 1)
        out_x = F.elu(self.fc_out_2(lat_frame))
        out_x = softmax(out_x)
        lat_frame_x = torch.cat((lat_frame, out_x), 1)
        out_y = F.elu(self.fc_out_3(lat_frame_x))
        out_y = softmax(out_y)
        return out_frame, out_x, out_y



    
