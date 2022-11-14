# Training code w
import torch
import torch.nn.functional as F
from preprocess import train_loader, test_loader
from model.model import get_training
import yaml

params = yaml.safe_load(open("params.yaml"))["train"]

# Training Loop

model, optimizer = get_training()

def train(model, train_loader,optimizer, epoch, device='cpu'):
    model.train()
    for batch_idx, (data,target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 49:
            print(f"Train {epoch} - {batch_idx}, Loss - {loss.item()}")

def save(model):
    torch.save(model.state_dict(),f'{params["model_save_location"]}/mnist_cnn.pt')


for epoch in range(params["epochs"]):
    train(model,train_loader,optimizer,epoch)
    save(model)
