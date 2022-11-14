import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import yaml
mparams = yaml.safe_load(open("params.yaml"))["train"]["model"]
oparams = yaml.safe_load(open("params.yaml"))["train"]["optimizer"]

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, mparams["conv1_filters"], 3, 1)
        self.conv2 = nn.Conv2d(32, mparams["conv2_filters"], 3, 1)
        self.drop1 = nn.Dropout(mparams["dropout1_prob"])
        self.drop2 = nn.Dropout(mparams["dropout2_prob"])
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.drop1(x)
        x = torch.flatten(x,1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def get_training():
    model = Net()
    optimizer = optim.Adadelta(model.parameters(), lr = oparams["lr"])
    print(model,optimizer)
    return model,optimizer
