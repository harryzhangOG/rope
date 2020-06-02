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
    optimizer = optim.Adam(inv_model.parameters(), lr=1e-4)
    loss_function = nn.CrossEntropyLoss()
    # TODO: Load Training and Testing data
    path = os.path.join(os.getcwd(), 'states_actions')
    X1 = np.load(os.path.join(path, 's.npy'))
    X2 = np.load(os.path.join(path, 'sp1.npy'))
    Y = np.load(os.path.join(path, 'a_encoded.npy'))
    # TODO: Finalize how many training points/validation points we want
    holdout = 900 
    x1 = torch.from_numpy(X1).to(device)
    x1 = torch.reshape(x1, (X1.shape[0], -1))
    x2 = torch.from_numpy(X2).to(device)
    x2 = torch.reshape(x2, (X2.shape[0], -1))
    y = torch.from_numpy(Y).to(device)
    train_dataset = Data.TensorDataset(x1[:holdout], x2[:holdout], y[:holdout])
    train_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_dataset = Data.TensorDataset(x1[holdout:], x2[holdout:], y[holdout:])
    val_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    
    EPOCHS = 1000
    train_loss = []
    val_loss = []
    for e in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y = batch
            print(train_x2)
            inv_model.zero_grad()
            optimizer.zero_grad()
            # Train on first HOLDOUT points
            inv_model.train()
            output_frame, output_x, output_y = inv_model(train_x1.float(), train_x2.float())
            loss = loss_function(outputs, train_y.float())
            
            loss.backward()
            # Training loss
            tloss = loss.item()
            optimizer.step()
            train_loss.append(tloss)
            #if (i % 5) == 0:
            #    print("Epoch: ", e, "Iteration: ", i, " Training Loss: ", tloss)
        for i, batch in enumerate(val_dataloader, 0):
            # Validate on the rest
            inv_model.eval()
            val_x1, val_x2, val_y = batch
            val_outputs = inv_model(val_x1.float(), val_x2.float())
            vloss = loss_function(val_outputs, val_y.float()).item()
            val_loss.append(vloss)
            #if (i % 5) == 0:
            #    print("Epoch: ", e, "Iteration: ", i, " Validation Loss: ", vloss)

    torch.save({'model_state_dict': inv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'inv_model_ckpt.pth')
    return train_loss, val_loss

if __name__ == "__main__":
    train_loss, val_loss = train()
    EPOCHS = 1000
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_title("Loss vs. Epochs")
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.plot(np.linspace(0, EPOCHS - 1, EPOCHS), train_loss, np.linspace(0, EPOCHS - 1, EPOCHS), val_loss)
    ax.legend(['Training Loss', 'Validation Loss'])
    plt.show()
    fig.savefig('loss.png')

