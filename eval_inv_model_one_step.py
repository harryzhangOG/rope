from inv_model import *
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.utils.data as Data
import torch.nn.functional as F
import os

"""
Evaluate the performance of the trained inverse model.
- Load the trained checkpoint
- Pass in a pair of states
- Output the predicted state
- TODO: Evaluate the performance in Blender.
"""
def eval_inv(ckpt, s1, s2):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    inv_model = Inv_Model()
    inv_model.to(device)
    checkpoint = torch.load(ckpt, map_location=device)
    inv_model.load_state_dict(checkpoint['model_state_dict'])
    inv_model.eval()
    # Cast to Torch tensors
    with torch.no_grad():
        s1 = torch.from_numpy(s1).to(device)
        s1 = torch.reshape(s1, (s1.shape[0], -1))
        s2 = torch.from_numpy(s2).to(device)
        s2 = torch.reshape(s2, (s2.shape[0], -1))
        output_action = inv_model(s1.float(), s2.float())
        return output_action

if __name__ == "__main__":
    # Checkpoint path
    ckpt = 'inv_model_ckpt_one_step.pth'
    # Test initial state
    s1 = np.load(os.path.join('states_actions', 's_test.npy'))
    # Test target state
    s2 = np.load(os.path.join('states_actions', 'sp1_test.npy'))
    a = np.load(os.path.join('states_actions', 'a_test.npy'))
    output_action = (eval_inv(ckpt, s1, s2)).numpy()
    print("Predicted action: ", output_action)
    print("Ground truth action: ", a)