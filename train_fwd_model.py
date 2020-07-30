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
        device = torch.device("cuda:1")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Reproducable
    torch.manual_seed(1)
    # Forward model net
    fwd_model = Fwd_Model()
    fwd_model.to(device)
    #ckpt = 'fwd_model_ckpt_3d.pth'
    #checkpoint = torch.load(ckpt, map_location=device)
    #fwd_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.SGD(fwd_model.parameters(), lr=0.001, momentum=0.9)
    #optimizer = optim.Adam(fwd_model.parameters(), lr=1e-2)
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)

    loss_function = nn.MSELoss()
    # TODO: Load Training and Testing data
    path = os.path.join(os.getcwd(), 'states_actions')
    X1 = np.load(os.path.join(path, 's_spring.npy'))
    print(X1.shape)
    X2 = np.load(os.path.join(path, 'sp1_spring.npy'))
    Y = np.load(os.path.join(path, 'a_spring.npy'))
    # TODO: Finalize how many training points/validation points we want
    holdout = 30000
    x1 = torch.from_numpy(X1[:, 0, :]).to(device)
    x1 = torch.reshape(x1, (X1.shape[0], -1))
    x2 = torch.from_numpy(X2[:,0,:]).to(device)
    x2 = torch.reshape(x2, (X2.shape[0], -1))
    y = torch.from_numpy(Y).to(device)
    train_dataset = Data.TensorDataset(x1[:holdout], x2[:holdout], y[:holdout])
    train_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_dataset = Data.TensorDataset(x1[holdout:], x2[holdout:], y[holdout:])
    val_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    
    EPOCHS = 1000

    val_loss_fwd = []
    train_loss_fwd = []

    for e in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y = batch
            fwd_model.zero_grad()
            optimizer.zero_grad()
            # Train on first HOLDOUT points
            fwd_model.train()

            outputs = fwd_model(train_x1.float(), train_y.float())
            loss = loss_function(outputs, train_x2.float())
            
            loss.backward()

            # Training loss
            tloss = loss.item()
            optimizer.step()

            train_loss_fwd.append(loss.item())
            if (i % 40) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Training Loss Fwd: ", loss.item())
        for i, batch in enumerate(val_dataloader, 0):
            # Validate on the rest
            fwd_model.eval()

            val_x1, val_x2, val_y = batch
            val_outputs = fwd_model(val_x1.float(), val_y.float())

            vloss = loss_function(val_outputs, val_x2.float()).item()

            val_loss_fwd.append(vloss)

            if (i % 40) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Validation Loss Fwd: ", vloss)
        # Adjust learning rate
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': fwd_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'fwd_model_ckpt_free_end_only.pth')
    return train_loss_fwd, val_loss_fwd

if __name__ == "__main__":
    train_loss_fwd, val_loss_fwd = train()
    np.save('trainloss_fwd_4.npy', train_loss_fwd)
    np.save('valloss_fwd_4.npy', val_loss_fwd)
    EPOCHS = 2000
