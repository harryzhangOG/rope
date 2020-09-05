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
import future

class Encoder_ResNet(nn.Module):
    def __init__(self, latent_dim=4):
        super(Encoder_ResNet, self).__init__()
        self.encoder_net = models.resnet18(pretrained=True)
        num_ftrs = self.encoder_net.fc.in_features
        self.encoder_net.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, latent_dim))

    def forward(self, x):
        latent = self.encoder_net(x)
        return latent

class Encoder(nn.Module):
    def __init__(self, latent_dim=4, channel_dim=3):
        super(Encoder, self).__init__()
        self.z_dim = latent_dim
        self.model = nn.Sequential(
                nn.Conv2d(channel_dim, 128, 3, 1, 1),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(256, 256, 3, 1, 1),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.LeakyReLU(0.3, inplace=True),
                # 128 x 16 x 16
                nn.Conv2d(512, 256, 4, 2, 1),
                nn.LeakyReLU(0.3, inplace=True),
                nn.Conv2d(256, 256, 4, 2, 1),
                nn.LeakyReLU(0.3, inplace=True),
                # 256 x 16 x 16
        )
        self.out = nn.Linear(256*16*16, self.z_dim)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.shape[0], -1)
        x = self.out(x)
        return x


class Transition_Model(nn.Module):
    def __init__(self, latent_dim=4):
        super(Transition_Model, self).__init__()
        self.fc1 = nn.Linear(latent_dim+3, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, latent_dim)
    def forward(self, latent, a):
        in_cat = torch.cat((latent, a), -1)
        x = F.relu(self.fc1(in_cat))
        x = F.relu(self.fc2(x))
        out_latent = self.fc3(x)
        return out_latent


class TrainDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir_s = os.path.join(main_dir, 'images/s')
        self.img_dir_sp1 = os.path.join(main_dir, 'images/s')
        self.transform = transform
        self.device = device
        all_imgs_s = os.listdir(self.img_dir_s)
        for f in all_imgs_s:
            if f[-8:-4] == '1001':
                all_imgs_s.remove(f)
        all_imgs_sp1 = os.listdir(self.img_dir_sp1)
        for f in all_imgs_sp1:
            if f[-8:-4] == '0001':
                all_imgs_sp1.remove(f)
        self.total_imgs_s = natsort.natsorted(all_imgs_s)[:holdout]
        self.total_imgs_sp1 = natsort.natsorted(all_imgs_sp1)[:holdout]
        self.mean = np.array([0.5, 0.5, 0])
        self.std = np.array([0.5, 0.5, 1])
        self.total_labels = np.load(os.path.join(self.main_dir, 'a_spring.npy'))[:holdout]

    def __len__(self):
        return len(self.total_imgs_s)

    def __getitem__(self, idx):
        img_loc_s = os.path.join(self.img_dir_s, self.total_imgs_s[idx])
        img_loc_sp1 = os.path.join(self.img_dir_sp1, self.total_imgs_sp1[idx])
        image_s = Image.open(img_loc_s).convert("RGB")
        image_sp1 = Image.open(img_loc_sp1).convert("RGB")
        image_s = np.array(image_s).astype('float32')/255
        image_sp1 = np.array(image_sp1).astype('float32')/255
        tensor_image_s = self.transform(image_s)
        tensor_image_sp1 = self.transform(image_sp1)
        return tensor_image_s, tensor_image_sp1, torch.from_numpy((self.total_labels[idx] - self.mean) / self.std).to(self.device)

class ValDataset(Dataset):
    def __init__(self, main_dir, transform, holdout, device):
        self.main_dir = main_dir
        self.img_dir_s = os.path.join(main_dir, 'images/s')
        self.img_dir_sp1 = os.path.join(main_dir, 'images/s')
        self.transform = transform
        self.device = device
        all_imgs_s = os.listdir(self.img_dir_s)
        for f in all_imgs_s:
            if f[-8:-4] == '1001':
                all_imgs_s.remove(f)
        all_imgs_sp1 = os.listdir(self.img_dir_sp1)
        for f in all_imgs_sp1:
            if f[-8:-4] == '0001':
                all_imgs_sp1.remove(f)
        self.total_imgs_s = natsort.natsorted(all_imgs_s)[holdout:]
        self.total_imgs_sp1 = natsort.natsorted(all_imgs_sp1)[holdout:]
        self.mean = np.array([0.5, 0.5, 0])
        self.std = np.array([0.5, 0.5, 1])
        self.total_labels = np.load(os.path.join(self.main_dir, 'a_spring.npy'))[holdout:]

    def __len__(self):
        return len(self.total_imgs_sp1)

    def __getitem__(self, idx):
        img_loc_s = os.path.join(self.img_dir_s, self.total_imgs_s[idx])
        img_loc_sp1 = os.path.join(self.img_dir_sp1, self.total_imgs_sp1[idx])
        image_s = Image.open(img_loc_s).convert("RGB")
        image_sp1 = Image.open(img_loc_sp1).convert("RGB")
        image_s = np.array(image_s).astype('float32')/255
        image_sp1 = np.array(image_sp1).astype('float32')/255
        tensor_image_s = self.transform(image_s)
        tensor_image_sp1 = self.transform(image_sp1)
        return tensor_image_s, tensor_image_sp1, torch.from_numpy((self.total_labels[idx] - self.mean) / self.std).to(self.device)

def compute_cpc_loss(obs, obs_pos, encoder, trans, actions, device):
    bs = obs.shape[0]

    z, z_pos = encoder(obs.float()), encoder(obs_pos.float())  # b x z_dim
    z_next = trans(z, actions.float())

    neg_dot_products = torch.mm(z_next, z.t()) # b x b
    neg_dists = -((z_next ** 2).sum(1).unsqueeze(1) - 2* neg_dot_products + (z ** 2).sum(1).unsqueeze(0))
    idxs = np.arange(bs)
    # Set to minus infinity entries when comparing z with z - will be zero when apply softmax
    neg_dists[idxs, idxs] = float('-inf') # b x b+1

    pos_dot_products = (z_pos * z_next).sum(dim=1) # b
    pos_dists = -((z_pos ** 2).sum(1) - 2* pos_dot_products + (z_next ** 2).sum(1))
    pos_dists = pos_dists.unsqueeze(1) # b x 1

    dists = torch.cat((neg_dists, pos_dists), dim=1) # b x b+1
    dists = F.log_softmax(dists, dim=1) # b x b+1
    loss = -dists[:, -1].mean() # Get last column with is the true pos sample

    return loss

def train():
    normalize = transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    transform = transforms.Compose([transforms.ToTensor(), normalize])

    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("Using ", device)

    # ResNet 50 based fwd model
    encoder = Encoder_ResNet()
    #encoder = Encoder()
    transition = Transition_Model()
    encoder.to(device)
    transition.to(device)
    parameters = list(encoder.parameters())+list(transition.parameters())

    optimizer = torch.optim.Adam(parameters, lr=5e-3, weight_decay=2e-3)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)

    trainLoss = []
    valLoss = []
    EPOCHS = 2

    # Load data
    path = os.path.join(os.path.join(os.getcwd(), 'mpc_policy_sa'))
    holdout = 10

    train_dataset = TrainDataset(path, transform, holdout, device)
    val_dataset = ValDataset(path, transform, holdout, device)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=False, drop_last=True)

    for epoch in range(EPOCHS):
        for i, batch in enumerate(train_dataloader, 0):
            encoder.train()
            transition.train()
            train_o, train_op1, train_a = batch
            train_o = train_o.to(device)
            train_op1 = train_op1.to(device)
            train_a = train_a.to(device)
            optimizer.zero_grad()

            loss = compute_cpc_loss(train_o, train_op1, encoder, transition, train_a, device)
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 20)

            # Training loss
            tloss = loss.item()
            optimizer.step()

            trainLoss.append(tloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Training Loss: %.5f' % (epoch+1, i, tloss))
        for i, batch in enumerate(val_dataloader, 0):
            encoder.eval()
            transition.eval()

            val_o, val_op1, val_a = batch
            val_o = val_o.to(device)
            val_op1 = val_op1.to(device)
            val_a = val_a.to(device)
            vloss = compute_cpc_loss(val_o, val_op1, encoder, transition, val_a, device)
            vloss = vloss.item()
            valLoss.append(vloss)
            if i % 10 == 0:
                print('[Epoch %d, Iteration %d] Validation Loss: %.5f' % (epoch+1, i, vloss))
            
        lr_scheduler.step()
        for param_group in optimizer.param_groups:
            print("Current learning rate: ", param_group['lr'])

    torch.save({'model_state_dict': encoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'mpc_encoder_model.pth')
    torch.save({'model_state_dict': transition.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
               }, 'mpc_transition_model.pth')
    return trainLoss, valLoss

if __name__ == "__main__":
    trainLoss, valLoss = train()
    if not os.path.exists("./results_mpc_res"):
        os.makedirs('./results_mpc_res')
    save_dir = os.path.join(os.getcwd(), 'results_mpc_res')
    np.save(os.path.join(save_dir, 'trainloss_resnet.npy'), trainLoss)
    np.save(os.path.join(save_dir, 'valloss_resnet.npy'), valLoss)
    EPOCHS = 2000

                
