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
    ckpt = 'inv_model_ckpt.pth'
    checkpoint = torch.load(ckpt, map_location=device)
    inv_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = optim.Adam(inv_model.parameters(), lr=1e-4, weight_decay=1e-4)
    loss_function = nn.CrossEntropyLoss()
    # TODO: Load Training and Testing data
    path = os.path.join(os.getcwd(), 'states_actions')

    np_load_old = np.load
    # modify the default parameters of np.load
    np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

    X1 = np.load(os.path.join(path, 's.npy'))
    print(X1.shape)
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
    
    EPOCHS = 2500
    train_loss = []
    val_loss = []
    train_acc_frame = []
    train_acc_x = []
    train_acc_y = []
    val_acc_frame = []
    val_acc_x = []
    val_acc_y = []

    for e in range(EPOCHS):
        correct_hits_frame = 0
        correct_hits_x = 0
        correct_hits_y = 0
        total = 0
        for i, batch in enumerate(train_dataloader, 0):
            train_x1, train_x2, train_y_frame, train_y_x, train_y_y = batch
            inv_model.zero_grad()
            optimizer.zero_grad()
            # Train on first HOLDOUT points
            inv_model.train()
            output_frame, output_x, output_y = inv_model(train_x1.float(), train_x2.float())
            pred_frame, pred_x, pred_y = torch.max(output_frame.data, 1)[1], torch.max(output_x.data, 1)[1], torch.max(output_y.data, 1)[1]
            
            total += output_frame.size(0)

            # Convert to labels because CE loss expects class labels
            train_y_frame_label = torch.max(train_y_frame, 1)[1]
            train_y_x_label = torch.max(train_y_x, 1)[1]
            train_y_y_label = torch.max(train_y_y, 1)[1]

            loss_1 = loss_function(output_frame, train_y_frame_label)
            loss_2 = loss_function(output_x, train_y_x_label)
            loss_3 = loss_function(output_y, train_y_y_label)
            
            correct_hits_frame += (pred_frame==train_y_frame_label).sum().item()
            correct_hits_x += (pred_frame==train_y_x_label).sum().item()
            correct_hits_y += (pred_frame==train_y_y_label).sum().item()

            # Aggregate the loss
            loss = loss_1 + loss_2 + loss_3
            loss.backward()
            # Training loss
            tloss = loss.item()
            optimizer.step()
            train_loss.append(tloss)
            if (i % 10) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Training Loss: ", tloss)

        train_acc_frame.append(correct_hits_frame / total)
        print('Training accuracy of frame on epoch ',e+1,'= ',str((correct_hits_frame/total)*100))
        train_acc_x.append(correct_hits_x / total)
        print('Training accuracy of x on epoch ',e+1,'= ',str((correct_hits_x/total)*100))
        train_acc_y.append(correct_hits_y / total)
        print('Training accuracy of y on epoch ',e+1,'= ',str((correct_hits_y/total)*100))

        correct_hits_frame = 0
        correct_hits_x = 0
        correct_hits_y = 0
        total = 0
        for i, batch in enumerate(val_dataloader, 0):
            # Validate on the rest
            inv_model.eval()
            val_x1, val_x2, val_y_frame, val_y_x, val_y_y = batch
            val_output_frame, val_output_x, val_output_y = inv_model(val_x1.float(), val_x2.float())
            val_pred_frame, val_pred_x, val_pred_y = torch.max(val_output_frame.data, 1)[1], torch.max(val_output_x.data, 1)[1], torch.max(val_output_y.data, 1)[1]

            total += val_output_frame.size(0)

            # Convert to labels because CE loss expects class labels
            val_y_frame_label = torch.max(val_y_frame, 1)[1]
            val_y_x_label = torch.max(val_y_x, 1)[1]
            val_y_y_label = torch.max(val_y_y, 1)[1]

            vloss1 = loss_function(val_output_frame, val_y_frame_label)
            vloss2 = loss_function(val_output_x, val_y_x_label)
            vloss3 = loss_function(val_output_y, val_y_y_label)

            vloss = vloss1 + vloss2 + vloss3

            correct_hits_frame += (val_pred_frame==val_y_frame_label).sum().item()
            correct_hits_x += (val_pred_x==val_y_x_label).sum().item()
            correct_hits_y += (val_pred_y==val_y_y_label).sum().item()

            val_loss.append(vloss.item())
            if (i % 10) == 0:
                print("Epoch: ", e, "Iteration: ", i, " Validation Loss: ", vloss.item())


        val_acc_frame.append(correct_hits_frame / total)
        print('Validation accuracy of frame on epoch ',e+1,'= ',str((correct_hits_frame/total)*100))
        val_acc_x.append(correct_hits_x / total)
        print('Validation accuracy of x on epoch ',e+1,'= ',str((correct_hits_x/total)*100))
        val_acc_y.append(correct_hits_y / total)
        print('Validation accuracy of y on epoch ',e+1,'= ',str((correct_hits_y/total)*100))

    torch.save({'model_state_dict': inv_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'inv_model_ckpt.pth')

    np.load = np_load_old

    return train_loss, val_loss, train_acc_frame, train_acc_x, train_acc_y, val_acc_frame, val_acc_x, val_acc_y

if __name__ == "__main__":
    train_loss, val_loss, train_acc_frame, train_acc_x, train_acc_y, val_acc_frame, val_acc_x, val_acc_y = train()
    
    save_dir = os.path.join(os.getcwd(), 'multistep_results')
    np.save(os.path.join(save_dir, 'trainloss.npy'), train_loss)
    np.save(os.path.join(save_dir, 'valloss.npy'), val_loss)
    np.save(os.path.join(save_dir, 'train_acc_frame.npy'), train_acc_frame)
    np.save(os.path.join(save_dir, 'train_acc_x.npy'), train_acc_x)
    np.save(os.path.join(save_dir, 'train_acc_y.npy'), train_acc_y)
    np.save(os.path.join(save_dir, 'val_acc_frame.npy'), val_acc_frame)
    np.save(os.path.join(save_dir, 'val_acc_x.npy'), val_acc_x)
    np.save(os.path.join(save_dir, 'val_acc_y.npy'), val_acc_y)

    EPOCHS = 2500
  #  fig, ax = plt.subplots(figsize=(12, 7))
  #  ax.set_title("Loss vs. Epochs")
  #  ax.set_xlabel('Epochs')
  #  ax.set_ylabel('Loss')
  #  ax.plot(np.linspace(0, len(train_loss) - 1, len(train_loss)), train_loss, np.linspace(0, len(val_loss) - 1, len(val_loss)), val_loss)
  #  ax.legend(['Training Loss', 'Validation Loss'])
# #   plt.show()
  #  fig.savefig('loss.png')

