import torch
import numbers
import numpy as np
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image

import torchvision.models as models
import os
import natsort

class Fwd_Model_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net50 = models.resnet50(pretrained=False)
        num_ftrs = self.net50.fc.in_features
        self.net50.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, 10))
        self.fc_out = nn.Linear(20, 3)

    def forward(self, x1, x2):
        latent1 = self.net50(x1)
        latent2 = self.net50(x2)
        latent = torch.cat((latent1, latent2), 1)
        out = self.fc_out(latent)
        return out

class TrainDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir_s = os.path.join(main_dir, 'images/s')
        self.img_dir_sp1 = os.path.join(main_dir, 'images/s')
        self.transform = transform
        all_imgs_s = os.listdir(self.img_dir_s)[:-1]
        all_imgs_sp1 = os.listdir(self.img_dir_sp1)[1:]
        self.total_imgs_s = natsort.natsorted(all_imgs_s)[:holdout]
        self.total_imgs_sp1 = natsort.natsorted(all_imgs_sp1)[:holdout]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'a_spring.npy'))[:holdout]).to(device)

    def __len__(self):
        return len(self.total_imgs_s)

    def __getitem__(self, idx):
        img_loc_s = os.path.join(self.img_dir_s, self.total_imgs_s[idx])
        img_loc_sp1 = os.path.join(self.img_dir_sp1, self.total_imgs_sp1[idx])
        image_s = Image.open(img_loc_s).convert("RGB")
        image_sp1 = Image.open(img_loc_sp1).convert("RGB")
        tensor_image_s = self.transform(image_s)
        tensor_image_sp1 = self.transform(image_sp1)
        return tensor_image_s, tensor_image_sp1, self.total_labels[idx]

class ValDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir_s = os.path.join(main_dir, 'images/s')
        self.img_dir_sp1 = os.path.join(main_dir, 'images/s')
        self.transform = transform
        all_imgs_s = os.listdir(self.img_dir_s)[:-1]
        all_imgs_sp1 = os.listdir(self.img_dir_sp1)[1:]
        self.total_imgs_s = natsort.natsorted(all_imgs_s)[holdout:]
        self.total_imgs_sp1 = natsort.natsorted(all_imgs_sp1)[holdout:]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'a_spring.npy'))[holdout:]).to(device)

    def __len__(self):
        return len(self.total_imgs_s)

    def __getitem__(self, idx):
        img_loc_s = os.path.join(self.img_dir_s, self.total_imgs_s[idx])
        img_loc_sp1 = os.path.join(self.img_dir_sp1, self.total_imgs_sp1[idx])
        image_s = Image.open(img_loc_s).convert("RGB")
        image_sp1 = Image.open(img_loc_sp1).convert("RGB")
        tensor_image_s = self.transform(image_s)
        tensor_image_sp1 = self.transform(image_sp1)
        return tensor_image_s, tensor_image_sp1, self.total_labels[idx]

def train():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50 based fwd model
    net50 = Fwd_Model_ResNet()
    net50.to(device)

    cost = nn.MSELoss()
    optimizer = torch.optim.SGD(net50.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 2

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'mpc_policy_sa'))
    holdout = 18

    train_dataset = TrainDataset(path, transform, holdout, device)
    val_dataset = ValDataset(path, transform, holdout, device)

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            net50.train()
            train_x1, train_x2, train_y = batch
            train_x1 = train_x1.to(device)
            train_x2 = train_x2.to(device)
            train_y = train_y.to(device)
            net50.zero_grad()
            optimizer.zero_grad()

            outputs = net50(train_x1, train_x2)
            loss = cost(outputs, train_y.float())
            loss.backward()

            # Training loss
            tloss = loss.item()
            optimizer.step()

            trainLoss.append(tloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Training Loss: %.5f' % (epoch+1, i, tloss))
        for i, batch in enumerate(val_dataloader, 0):
            net50.eval()

            val_x1, val_x2, val_y = batch
            val_x1 = val_x1.to(device)
            val_x2 = val_x2.to(device)
            val_y = val_y.to(device)
            val_outputs = net50(val_x1, val_x2)
            vloss = cost(val_outputs, val_y.float()).item()
            valLoss.append(vloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'mpc_res_model.pth')
    return trainLoss, valLoss

if __name__ == "__main__":
    trainLoss, valLoss = train()
    if not os.path.exists("./results_mpc_res"):
        os.makedirs('./results_mpc_res')
    save_dir = os.path.join(os.getcwd(), 'results_mpc_res')
    np.save(os.path.join(save_dir, 'trainloss_resnet.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet.npy'), valLoss)
    EPOCHS = 2000

                
