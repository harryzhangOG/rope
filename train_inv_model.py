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
Training details from the paper:

ADAM Optimizer, lr=1e-4

"""
def train():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Reproducable
    torch.manual_seed(1)
    # Inverse model net
    inv_model = Inv_Model()
    inv_model.to(device)
    optimizer = optim.Adam(inv_model.parameters(), lr=1e-3)
    loss_function = nn.MSELoss()
    # TODO: Load Training and Testing data
    path = os.path.join(os.getcwd(), 'states_actions')
    X1 = np.load(os.path.join(path, 's.npy'))
    X2 = np.load(os.path.join(path, 'sp1.npy'))
    Y = np.load(os.path.join(path, 'a.npy'))
    train_x1 = torch.from_numpy(X1).to(device)
    train_x1 = torch.reshape(train_x1, (X1.shape[0], -1))
    train_x2 = torch.from_numpy(X1).to(device)
    train_x2 = torch.reshape(train_x2, (X2.shape[0], -1))
    train_y = torch.from_numpy(Y).to(device)
    print(train_y.shape)
    
    EPOCHS = 2000
    train_loss = []
    val_loss = []
    # TODO: Finalize how many training points/validation points we want
    holdout = 900 
    for e in range(EPOCHS):
        inv_model.zero_grad()
        optimizer.zero_grad()
        # Train on first HOLDOUT points
        inv_model.train()
        outputs = inv_model(train_x1[:holdout].float(), train_x2[:holdout].float())
        loss = loss_function(outputs, train_y[:holdout].float())
        
        loss.backward()
        # Training loss
        tloss = loss.item()
        optimizer.step()
        train_loss.append(tloss)
        # Validate on the rest
        inv_model.eval()
        val_outputs = inv_model(train_x1[holdout:].float(), train_x2[holdout:].float())
        vloss = loss_function(val_outputs, train_y[holdout:].float()).item()
        val_loss.append(vloss)
        if (e % 100) == 0:
            print("Epoch: ", e, " Training Loss: ", tloss)
            print("Epoch: ", e, " Validation Loss: ", vloss)
    torch.save({'epoch': e,
                'model_state_dict': inv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'inv_model_ckpt.pth')
    return train_loss, val_loss

if __name__ == "__main__":
    train_loss, val_loss = train()
    EPOCHS = 2000
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Loss vs. Epochs")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(np.linspace(0, EPOCHS - 1, EPOCHS), train_loss, np.linspace(0, EPOCHS - 1, EPOCHS), val_loss)
    ax.legend(['Training Loss', 'Validation Loss'])
    plt.show()

