import torch
import yaml
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

params = yaml.safe_load(open("params.yaml"))["train"]

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train = datasets.MNIST("data", train=True, transform=transform)
test = datasets.MNIST("data", train=False, transform=transform)

train_loader = DataLoader(train,batch_size=params["batchsize"])
test_loader = DataLoader(test,batch_size=params["batchsize"])
