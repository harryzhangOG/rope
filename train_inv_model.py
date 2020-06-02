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
    Y = np.load(os.path.join(path, 'a_enc.npy'))
    # Post-process the encoded action to make it compatible with pytorch
    Y_frame = np.vstack(list(Y[:, 0]))
    Y_x = np.vstack(list(Y[:, 1]))
    Y_y = np.vstack(list(Y[:, 2]))
    # TODO: Finalize how many training points/validation points we want
    holdout = 5500 

    print("NumPy Data loaded")

    x1 = torch.from_numpy(X1).to(device)
    x1 = torch.reshape(x1, (X1.shape[0], -1))
    x2 = torch.from_numpy(X2).to(device)
    x2 = torch.reshape(x2, (X2.shape[0], -1))
    y_frame = torch.from_numpy(Y_frame).to(device)
    y_x = torch.from_numpy(Y_x).to(device)
    y_y = torch.from_numpy(Y_y).to(device)

    print("PyTorch Data loaded")

    train_dataset = Data.TensorDataset(x1[:holdout], x2[:holdout], y_frame[:holdout], y_x[:holdout], y_y[:holdout])
    train_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    val_dataset = Data.TensorDataset(x1[holdout:], x2[holdout:], y_frame[holdout:], y_x[holdout:], y_y[holdout:])
    val_dataloader = Data.DataLoader(train_dataset, batch_size=200, shuffle=True)
    
    EPOCHS = 1000
    train_loss = []
    val_loss = []
    for e in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y_frame, train_y_x, train_y_y = batch
            inv_model.zero_grad()
            optimizer.zero_grad()
            # Train on first HOLDOUT points
            inv_model.train()
            output_frame, output_x, output_y = inv_model(train_x1.float(), train_x2.float())

            # Convert to labels because CE loss expects class labels
            train_y_frame_label = torch.max(train_y_frame, 1)[1]
            train_y_x_label = torch.max(train_y_x, 1)[1]
            train_y_y_label = torch.max(train_y_y, 1)[1]

            loss_1 = loss_function(output_frame, train_y_frame_label)
            loss_2 = loss_function(output_x, train_y_x_label)
            loss_3 = loss_function(output_y, train_y_y_label)

            # Aggregate the loss
            loss = loss_1 + loss_2 + loss_3
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
            val_x1, val_x2, val_y_frame, val_y_x, val_y_y = batch
            val_output_frame, val_output_x, val_output_y = inv_model(val_x1.float(), val_x2.float())

            # Convert to labels because CE loss expects class labels
            val_y_frame_label = torch.max(val_y_frame, 1)[1]
            val_y_x_label = torch.max(val_y_x, 1)[1]
            val_y_y_label = torch.max(val_y_y, 1)[1]

            vloss1 = loss_function(val_output_frame, val_y_frame_label)
            vloss2 = loss_function(val_output_x, val_y_x_label)
            vloss3 = loss_function(val_output_y, val_y_y_label)

            vloss = vloss1 + vloss2 + vloss3

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

