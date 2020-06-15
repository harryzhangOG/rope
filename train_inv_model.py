from inv_model import *
from fwd_model import *
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
    # Forward model net
    fwd_model = Fwd_Model()
    fwd_model.to(device)
    ckpt = 'inv_model_ckpt.pth'
#    checkpoint = torch.load(ckpt, map_location=device)
#    inv_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_1 = optim.Adam(inv_model.parameters(), lr=4e-4)
    optimizer_2 = optim.Adam(fwd_model.parameters(), lr=4e-3)
    
    lr_scheduler_1 = torch.optim.lr_scheduler.StepLR(optimizer_1, step_size=50, gamma=0.8)
    lr_scheduler_2 = torch.optim.lr_scheduler.StepLR(optimizer_2, step_size=50, gamma=0.8)

    loss_function = nn.MSELoss()
    # TODO: Load Training and Testing data
    path = os.path.join(os.getcwd(), 'states_actions')
    X1 = np.load(os.path.join(path, 's.npy'))
    print(X1.shape)
    X2 = np.load(os.path.join(path, 'sp1.npy'))
    Y = np.load(os.path.join(path, 'a.npy'))
    # TODO: Finalize how many training points/validation points we want
    holdout = 45000 
    x1 = torch.from_numpy(X1).to(device)
    x1 = torch.reshape(x1, (X1.shape[0], -1))
    x2 = torch.from_numpy(X2).to(device)
    x2 = torch.reshape(x2, (X2.shape[0], -1))
    y = torch.from_numpy(Y).to(device)
    train_dataset = Data.TensorDataset(x1[:holdout], x2[:holdout], y[:holdout])
    train_dataloader = Data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    val_dataset = Data.TensorDataset(x1[holdout:], x2[holdout:], y[holdout:])
    val_dataloader = Data.DataLoader(train_dataset, batch_size=100, shuffle=True)
    
    EPOCHS = 2000

    train_loss_inv = []
    val_loss_fwd = []
    train_loss_fwd = []
    val_loss_inv = []

    for e in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y = batch
            inv_model.zero_grad()
            fwd_model.zero_grad()
            optimizer_1.zero_grad()
            optimizer_2.zero_grad()
            # Train on first HOLDOUT points
            inv_model.train()
            fwd_model.train()

            outputs_1 = inv_model(train_x1.float(), train_x2.float())
            outputs_2 = fwd_model(train_x1.float(), train_y.float())
            loss_1 = loss_function(outputs_1, train_y.float())
            loss_2 = loss_function(outputs_2, train_x2.float())
            
            #loss.backward()
            loss_1.backward()
            loss_2.backward()

            loss = loss_1 + loss_2

            # Training loss
            tloss = loss.item()
            optimizer_1.step()
            optimizer_2.step()

            train_loss_inv.append(loss_1.item())
            train_loss_fwd.append(loss_2.item())
            if (i % 40) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Training Loss Inv: ", loss_1.item())
                print("Epoch: ", e, "Iteration: ", i, " Training Loss Fwd: ", loss_2.item())
        for i, batch in enumerate(val_dataloader, 0):
            # Validate on the rest
            inv_model.eval()
            fwd_model.eval()

            val_x1, val_x2, val_y = batch
            val_outputs_1 = inv_model(val_x1.float(), val_x2.float())
            val_outputs_2 = fwd_model(val_x1.float(), val_y.float())

            vloss_1 = loss_function(val_outputs_1, val_y.float()).item()
            vloss_2 = loss_function(val_outputs_2, val_x2.float()).item()

            vloss = vloss_1 + vloss_2

            val_loss_inv.append(vloss_1)
            val_loss_fwd.append(vloss_2)

            if (i % 40) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Validation Loss Inv: ", vloss_1)
                print("Epoch: ", e, "Iteration: ", i, " Validation Loss Fwd: ", vloss_2)
        # Adjust learning rate
        lr_scheduler_1.step()
        lr_scheduler_2.step()
        for param_group in optimizer_1.param_groups:
            print("Current learning rate 1: ", param_group['lr'])
        for param_group in optimizer_2.param_groups:
            print("Current learning rate 2: ", param_group['lr'])

    torch.save({'model_state_dict': inv_model.state_dict(),
                'optimizer_state_dict': optimizer_1.state_dict()
               }, 'inv_model_ckpt_3d.pth')
    torch.save({'model_state_dict': fwd_model.state_dict(),
                'optimizer_state_dict': optimizer_2.state_dict()
               }, 'fwd_model_ckpt_3d.pth')
    return train_loss_inv, val_loss_inv, train_loss_fwd, val_loss_fwd

if __name__ == "__main__":
    train_loss_inv, val_loss_inv, train_loss_fwd, val_loss_fwd = train()
    np.save('trainloss_inv.npy', train_loss_inv)
    np.save('valloss_inv.npy', val_loss_inv)
    np.save('trainloss_fwd.npy', train_loss_fwd)
    np.save('valloss_fwd.npy', val_loss_fwd)
    EPOCHS = 2000
  #  fig, ax = plt.subplots(figsize=(12, 7))
  #  ax.set_title("Loss vs. Epochs")
  #  ax.set_xlabel('Epochs')
  #  ax.set_ylabel('Loss')
  #  ax.plot(np.linspace(0, len(train_loss) - 1, len(train_loss)), train_loss, np.linspace(0, len(val_loss) - 1, len(val_loss)), val_loss)
  #  ax.legend(['Training Loss', 'Validation Loss'])
# #   plt.show()
  #  fig.savefig('loss.png')
