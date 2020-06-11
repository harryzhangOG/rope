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
        device = torch.device("cuda:4")
        print("Using GPU Cuda 0")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Reproducable
    torch.manual_seed(1)
    # Inverse model net
    inv_model = Inv_Model()
    inv_model.to(device)
#    ckpt = 'inv_model_ckpt.pth'
#    checkpoint = torch.load(ckpt, map_location=device)
#    inv_model.load_state_dict(checkpoint['model_state_dict'])
#    optimizer = optim.Adam(inv_model.parameters(), lr=1e-3, weight_decay=1e-4)
    optimizer = optim.SGD(inv_model.parameters(), lr=5e-4, momentum=0.9, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.7)
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
    
    EPOCHS = 2500
    train_loss = []
    val_loss = []
    for e in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y = batch
#            inv_model.zero_grad()
            optimizer.zero_grad()
            # Train on first HOLDOUT points
            inv_model.train()
            outputs1, outputs2 = inv_model(train_x1.float(), train_x2.float(), train_y.float())
            loss1 = loss_function(outputs1, train_y.float())
            loss2 = loss_function(outputs2, train_x2.float())

            loss = loss1 + loss2
            
            #loss.backward()
            torch.autograd.backward([loss1, loss2])
            # Training loss
            tloss = loss.item()
            optimizer.step()
            train_loss.append(tloss)
            if (i % 20) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Training Loss: ", tloss)
        for i, batch in enumerate(val_dataloader, 0):
            # Validate on the rest
            inv_model.eval()
            val_x1, val_x2, val_y = batch
            val_outputs1, val_outputs2 = inv_model(val_x1.float(), val_x2.float(), val_y.float())
            vloss1 = loss_function(val_outputs1, val_y.float()).item()
            vloss2 = loss_function(val_outputs2, val_x2.float()).item()
            
            vloss = vloss1 + vloss2

            val_loss.append(vloss)
            if (i % 20) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Validation Loss: ", vloss)
        # Adjust learning rate
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': inv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'inv_model_ckpt_3d.pth')
    return train_loss, val_loss

if __name__ == "__main__":
    train_loss, val_loss = train()
    save_dir = os.path.join(os.getcwd(), 'multistep_results')
    np.save(os.path.join(save_dir, 'trainloss_3d.npy'), train_loss)
    np.save(os.path.join(save_dir, 'valloss_3d.npy'), val_loss)
    EPOCHS = 2500
  #  fig, ax = plt.subplots(figsize=(12, 7))
  #  ax.set_title("Loss vs. Epochs")
  #  ax.set_xlabel('Epochs')
  #  ax.set_ylabel('Loss')
  #  ax.plot(np.linspace(0, len(train_loss) - 1, len(train_loss)), train_loss, np.linspace(0, len(val_loss) - 1, len(val_loss)), val_loss)
  #  ax.legend(['Training Loss', 'Validation Loss'])
# #   plt.show()
  #  fig.savefig('loss.png')
