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
import sys
#sys.path.append('/usr/local/lib/python3.7/site-packages')
import natsort

class TrainDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir = os.path.join(main_dir, 'seg')
        self.transform = transform
        all_imgs = os.listdir(self.img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)[:holdout]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'whip_actions.npy'))[:holdout]).to(device)

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
        self.img_dir = os.path.join(main_dir, 'seg')
        self.transform = transform
        all_imgs = os.listdir(self.img_dir)
        self.total_imgs = natsort.natsorted(all_imgs)[holdout:]
        self.total_labels = torch.from_numpy(np.load(os.path.join(self.main_dir, 'whip_actions.npy'))[holdout:]).to(device)

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.img_dir, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image, self.total_labels[idx]

class DistModel(nn.Module):
    def __init__(self, device, ac_dim):
        import itertools
        super(DistModel, self).__init__()
        self.resnet_mean = models.resnet18(pretrained=False)
        num_ftrs = self.resnet_mean.fc.in_features
        self.resnet_mean.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, ac_dim))

        self.logstd = nn.Parameter(torch.zeros(ac_dim, dtype=torch.float32, device=device))
        self.logstd.to(device)
        #self.optimizer = torch.optim.SGD(itertools.chain([self.logstd], self.resnet_mean.parameters()), 
        #                                 lr=1e-3, weight_decay=1e-4, momentum=0.9)
        self.optimizer1000 = torch.optim.Adam(itertools.chain([self.logstd], self.resnet_mean.parameters()), 
                                         lr=1e-3)
        self.optimizer2000 = torch.optim.Adam(itertools.chain([self.logstd], self.resnet_mean.parameters()), 
                                         lr=5e-4)
        # lr starts with 1e-3, second round starts w 5e-4, third rounds starts with 1e-4
    def forward(self, obs):
        mean = self.resnet_mean(obs)
        return torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(torch.exp(self.logstd)))


def train_1000():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50
    net50 = DistModel(device, 12)
    #state_dict = torch.load('resnet_ur5_model_physical_stable.pth')['model_state_dict']
    #net50.load_state_dict(state_dict)
    net50.resnet_mean.to(device)

    cost = nn.MSELoss()
    #optimizer = torch.optim.SGD(net50.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    optimizer = net50.optimizer1000
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 1000

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'whip_snake_dataset'))
    holdout = 150

    train_dataset = TrainDataset(path, transform, holdout, device)
    val_dataset = ValDataset(path, transform, holdout, device)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            net50.train()
            train_x, train_y = batch
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            net50.zero_grad()
            optimizer.zero_grad()

            outputs = net50(train_x).rsample()
            loss = cost(outputs, train_y.float())
            loss.backward()

            # Training loss
            tloss = loss.item()
            optimizer.step()

            trainLoss.append(tloss)
            if i % 5 == 0:
                print('[Epoch %d, Iteration %d] Training Loss: %.5f' % (epoch+1, i, tloss))
        for i, batch in enumerate(val_dataloader, 0):
            net50.eval()

            val_x, val_y = batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_outputs = net50(val_x).sample()
            vloss = cost(val_outputs, val_y.float()).item()
            valLoss.append(vloss)
            if i % 5 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'resnet_ur5_model_physical_snake.pth')
    if not os.path.exists("./results_snake_ur5_physical"):
        os.makedirs('./results_snake_ur5_physical')
    save_dir = os.path.join(os.getcwd(), 'results_snake_ur5_physical')
    np.save(os.path.join(save_dir, 'trainloss_resnet1000.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet1000.npy'), valLoss)
    return trainLoss, valLoss

def train_2000():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50
    net50 = DistModel(device, 12)
    state_dict = torch.load('resnet_ur5_model_physical_snake.pth')['model_state_dict']
    net50.load_state_dict(state_dict)
    net50.resnet_mean.to(device)

    cost = nn.MSELoss()
    #optimizer = torch.optim.SGD(net50.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    optimizer = net50.optimizer2000
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 1000

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'whip_snake_dataset'))
    holdout = 150

    train_dataset = TrainDataset(path, transform, holdout, device)
    val_dataset = ValDataset(path, transform, holdout, device)

    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            net50.train()
            train_x, train_y = batch
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            net50.zero_grad()
            optimizer.zero_grad()

            outputs = net50(train_x).rsample()
            loss = cost(outputs, train_y.float())
            loss.backward()

            # Training loss
            tloss = loss.item()
            optimizer.step()

            trainLoss.append(tloss)
            if i % 5 == 0:
                print('[Epoch %d, Iteration %d] Training Loss: %.5f' % (epoch+1, i, tloss))
        for i, batch in enumerate(val_dataloader, 0):
            net50.eval()

            val_x, val_y = batch
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            val_outputs = net50(val_x).rsample()
            vloss = cost(val_outputs, val_y.float()).item()
            valLoss.append(vloss)
            if i % 5 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'resnet_ur5_model_physical_snake_2000.pth')
    save_dir = os.path.join(os.getcwd(), 'results_snake_ur5_physical')
    np.save(os.path.join(save_dir, 'trainloss_resnet2000.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet2000.npy'), valLoss)
    return trainLoss, valLoss

if __name__ == "__main__":
    trainLoss, valLoss = train_1000()
    trainLoss, valLoss = train_2000()

                
