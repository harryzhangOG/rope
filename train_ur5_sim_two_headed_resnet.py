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
sys.path.append('/usr/local/lib/python3.7/site-packages')
import natsort

class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class TrainDataset(Dataset):
    def __init__(self, main_dir, img_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir = os.path.join(main_dir, img_dir)
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
    def __init__(self, main_dir, img_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir = os.path.join(main_dir, img_dir)
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

class DistModel(nn.Module):
    def __init__(self, device, ac_dim):
        import itertools
        super(DistModel, self).__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(3, 64, 10),  # 64@96*96
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64@48*48
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    # 128@42*42
            nn.MaxPool2d(2),   # 128@21*21
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), # 128@18*18
            nn.MaxPool2d(2), # 128@9*9
            nn.Conv2d(128, 3, 4),
            nn.ReLU(),   # 256@6*6
        )
        self.liner = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.resnet_mean = models.resnet18(pretrained=False)
        num_ftrs = self.resnet_mean.fc.in_features
        self.resnet_mean.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, ac_dim))

        self.logstd = nn.Parameter(torch.zeros(ac_dim, dtype=torch.float32, device=device))
        self.logstd.to(device)
        #self.optimizer = torch.optim.SGD(itertools.chain([self.logstd], self.resnet_mean.parameters()), 
        #                                 lr=1e-3, weight_decay=1e-4, momentum=0.9)
        self.optimizer = torch.optim.Adam(itertools.chain([self.logstd], self.resnet_mean.parameters()), 
                                         lr=1e-3)
    def forward(self, obs1, obs2):
        out1 = self.shared(obs1)
        out2 = self.shared(obs2)
        mean = self.resnet_mean(out1 + out2)
        return torch.distributions.MultivariateNormal(mean, scale_tril=torch.diag(torch.exp(self.logstd)))


def train():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50
    #net50 = models.resnet50(pretrained=False)
    #num_ftrs = net50.fc.in_features
    #net50.fc = nn.Sequential(nn.Dropout(0.55), nn.Linear(num_ftrs, 3))
    #state_dict = torch.load('resnet50_model_old.pth')['model_state_dict']
    #net50.load_state_dict(state_dict)
    #net50.to(device)
    net50 = DistModel(device, 6)
    # state_dict = torch.load('resnet_ur5_model.pth')['model_state_dict']
    # net50.load_state_dict(state_dict)
    net50.resnet_mean.to(device)

    cost = nn.MSELoss()
    #optimizer = torch.optim.SGD(net50.parameters(), lr=1e-3, weight_decay=1e-4, momentum=0.9)
    optimizer = net50.optimizer
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 500

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'whip_ur5_sa'))
    img_dir_1 = 'images'
    img_dir_2 = 'images_2'
    holdout = 4

    train_dataset_1 = TrainDataset(path, img_dir_1, transform, holdout, device)
    train_dataset_2 = TrainDataset(path, img_dir_2, transform, holdout, device)
    train_dataset = ConcatDataset(train_dataset_1, train_dataset_2)
    val_dataset_1 = ValDataset(path, img_dir_1, transform, holdout, device)
    val_dataset_2 = ValDataset(path, img_dir_2, transform, holdout, device)
    val_dataset = ConcatDataset(val_dataset_1, val_dataset_2)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            net50.train()
            train_1, train_2 = batch
            train_x = [train_1[0], train_2[0]]
            train_y = train_1[1]
            train_x[0] = train_x[0].to(device)
            train_x[1] = train_x[1].to(device)
            train_y = train_y.to(device)
            net50.zero_grad()
            optimizer.zero_grad()

            outputs = net50(train_x[0], train_x[1]).rsample()
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

            val_1, val_2 = batch
            val_x = [val_1[0], val_2[0]]
            val_y = val_1[1]
            val_x[0] = val_x[0].to(device)
            val_x[1] = val_x[1].to(device)
            val_y = val_y.to(device)
            val_outputs = net50(val_x[0], val_x[1]).rsample()
            vloss = cost(val_outputs, val_y.float()).item()
            valLoss.append(vloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': net50.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'resnet_ur5_model.pth')
    return trainLoss, valLoss

if __name__ == "__main__":
    trainLoss, valLoss = train()
    if not os.path.exists("./results_whip_ur5"):
        os.makedirs('./results_whip_ur5')
    save_dir = os.path.join(os.getcwd(), 'results_whip_ur5')
    np.save(os.path.join(save_dir, 'trainloss_resnet.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet.npy'), valLoss)
    EPOCHS = 1000

                
