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

class TrainDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir = os.path.join(main_dir, 'images')
        self.transform = transform
        all_imgs = os.listdir(self.img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)[:holdout]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'a.npy'))[:holdout]).to(device)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_labels[idx]

class ValDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir = os.path.join(main_dir, 'images')
        self.transform = transform
        all_imgs = os.listdir(self.img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)[holdout:]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'a.npy'))[holdout:]).to(device)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_labels[idx]

def train():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50
    net50 = models.resnet50(pretrained=False)
    num_ftrs = net50.fc.in_features
    net50.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, 3))
    state_dict = torch.load('resnet50_model_old.pth')['model_state_dict']
    net50.load_state_dict(state_dict)
    net50.to(device)

    cost = nn.MSELoss()
    optimizer = torch.optim.SGD(net50.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 200

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'whip_policy_sa'))
    holdout = 800

    train_dataset = TrainDataset(path, transform, holdout, device)
    val_dataset = ValDataset(path, transform, holdout, device)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            net50.train()
            train_x, train_y = batch
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            net50.zero_grad()
            optimizer.zero_grad()

            outputs = net50(train_x)
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

            val_x, val_y = batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_outputs = net50(val_x)
            vloss = cost(val_outputs, val_y.float()).item()
            valLoss.append(vloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'resnet50_model.pth')
    return trainLoss, valLoss

if __name__ == "__main__":
    trainLoss, valLoss = train()
    if not os.path.exists("./results_whip"):
        os.makedirs('./results_whip')
    save_dir = os.path.join(os.getcwd(), 'results_whip')
    np.save(os.path.join(save_dir, 'trainloss_resnet.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet.npy'), valLoss)
    EPOCHS = 2000

                
