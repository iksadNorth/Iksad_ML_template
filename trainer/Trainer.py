# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config.adapter import adapter

from tqdm import tqdm
import wandb

from torchmetrics import functional as remote_metrics
# %%
@adapter
def train(train_loader, net, criterion, optimizer, device, scheduler, print_frequency=1):
    period = print_frequency
    batch_size = train_loader.batch_size 
    length = len(train_loader)
    outputs_stacked, labels_stacked = [], []
    
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
        
        outputs = outputs.argmax(dim=-1)
        outputs_stacked.extend(outputs)
        labels_stacked.extend(labels)
        
        if i % period == 0:
            accuracy = remote_metrics.accuracy(labels, outputs)
            print(f'{i}/{length} train_loss : {loss / batch_size:2.4f} accuracy : {accuracy:2.4%}')
            wandb.log({'loss': loss, 'accuracy': accuracy})
    
    return torch.tensor(outputs_stacked), torch.tensor(labels_stacked)

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

