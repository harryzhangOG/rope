import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
obstacle = False
device = torch.device('cpu')
if obstacle:
    class FC_Net(nn.Module):
        def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(8, 200)
                self.fc2 = nn.Linear(200, 150)
                self.fc3 = nn.Linear(150, 100)
                self.fc4 = nn.Linear(100, 50)
                self.fc5 = nn.Linear(50, 40)
                self.fc6 = nn.Linear(40, 30)
                self.fc7 = nn.Linear(30, 20)
                self.fc8 = nn.Linear(20, 10)
                self.fc9 = nn.Linear(10, 10)
                self.fc10 = nn.Linear(10, 10)
        def forward(self, x):
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = F.tanh(self.fc3(x))
            x = F.tanh(self.fc4(x))
            x = F.tanh(self.fc5(x))
            x = F.tanh(self.fc6(x))
            x = F.tanh(self.fc7(x))
            x = F.tanh(self.fc8(x))
            x = F.tanh(self.fc9(x))
            out = self.fc10(x)
            return out
    fc_net = FC_Net()
    fc_net.load_state_dict(torch.load('baseline_ckpt.pth', map_location=device))
    test_x = np.array([0, 20, 25, 30, 35, 50, np.random.uniform(0.4, 0.5), np.random.uniform(-0.2, 0.3)])
    x = torch.from_numpy(test_x)
    y = fc_net(x.float()).detach().numpy()
    print("X: ", test_x)
    print("Y: ", y)
else:
    class FC_Net(nn.Module):
        def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(4, 200)
                self.fc2 = nn.Linear(200, 150)
                self.fc3 = nn.Linear(150, 100)
                self.fc4 = nn.Linear(100, 80)
                self.fc5 = nn.Linear(80, 70)
                self.fc6 = nn.Linear(70, 56)
        def forward(self, x):
            x = F.tanh(self.fc1(x))
            x = F.tanh(self.fc2(x))
            x = F.tanh(self.fc3(x))
            x = F.tanh(self.fc4(x))
            x = F.tanh(self.fc5(x))
            out = self.fc6(x)
            return out
    fc_net = FC_Net()
    fc_net.load_state_dict(torch.load('baseline_ckpt_wrap.pth', map_location=device))
    dx = np.random.uniform(-1, 1)
    dy = np.random.uniform(-1, 1)
    test_x = np.array([-13.225 + dx, 0 + dy, -10 + dx, -2.7 + dy])
    x = torch.from_numpy(test_x)
    y = fc_net(x.float()).detach().numpy()
    print("X: ", test_x)
    print("Y: ", y)
    np.save('test_y.npy', y)
