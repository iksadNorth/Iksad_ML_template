# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config.adapter import adapter

from tqdm import tqdm
# %%
@adapter
def train(train_loader, net, criterion, optimizer, device, scheduler):
    batch_size = train_loader.batch_size 
    length = len(train_loader)
    outputs_stacked, labels_stacked, loss_stacked = [], [], 0.0
    
    net.to(device)
    net.train()
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        scheduler.step()
        if i % 50 == 0:
            print(f'{i}/{length} train_loss : {loss / batch_size:2.4f}')
        
        outputs = outputs.argmax(dim=-1)
        outputs_stacked.extend(outputs)
        labels_stacked.extend(labels)
        loss_stacked += loss

    loss_stacked /= batch_size
    loss_stacked /= len(train_loader)
    
    return torch.tensor(outputs_stacked), torch.tensor(labels_stacked), loss_stacked

# %%
@torch.no_grad()
@adapter
def infer(valid_loader, net, device):
    outputs_stacked, labels_stacked, inputs_stacked = [], [], []
    
    net.to(device)
    net.eval()
    for i, data in enumerate(valid_loader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = net(inputs)
        
        outputs = outputs.argmax(dim=-1)
        outputs_stacked.extend(outputs)
        labels_stacked.extend(labels)
        inputs_stacked.extend(inputs)
    
    return torch.tensor(outputs_stacked), torch.tensor(labels_stacked), torch.stack(inputs_stacked, dim=0)

