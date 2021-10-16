import torch
import torchvision
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.optim import lr_scheduler

import numpy as np
import os
import matplotlib.pyplot as plt
import time
import copy
from torchsummary import summary
import subprocess
import gc
import sys
from tqdm import tqdm
gc.collect()
torch.cuda.empty_cache()

# subprocess.run(['clear'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device available: {device}","\n","--"*40, )

mean= np.array([0.48145466, 0.4578275, 0.40821073])
std= np.array([0.26862954, 0.26130258, 0.27577711])

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]), 
    'val': transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)]),
    }


image_datasets = {}
image_datasets['train'] = torchvision.datasets.CIFAR10(root='C:\\Users\\akash\\Documents\\VSCODE\\datasets\\cifar10', train=True, download=True, transform = data_transforms['train'])
image_datasets['val'] = torchvision.datasets.CIFAR10(root='C:\\Users\\akash\\Documents\\VSCODE\\datasets\\cifar10', train=False, download=True, transform = data_transforms['val'])

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

data_loaders = {x: torch.utils.data.DataLoader(image_datasets[x], shuffle=True, batch_size = 100) 
                for x in ['train', 'val']}

model = torchvision.models.resnet50(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Linear(in_features = 2048, out_features=len(image_datasets['train'].classes))

model.to(device=device)

learning_rate = 0.001
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

#exponential lr scheduler

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size = 7, gamma=0.1)


def train_model(model, 
                criterion,
                optimizer,
                scheduler,
                num_epochs=2):
    
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-'*10)
        
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            #iterate over the data
            
            for images, labels in tqdm(data_loaders[phase]):
                # print(f'\n {phase}')
                images = images.to(device)
                labels = labels.to(device)
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    _,predictions = torch.max(outputs, dim=-1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        
                #statistics
                running_loss +=loss.item()*images.size(0)
                running_corrects += torch.sum(predictions == labels.data)

            if phase == 'train':
                scheduler.step()
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            print('{} Loss: {:.4f}, Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            #deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            del images
            del labels
            torch.cuda.empty_cache()
        print()
        
    time_elapsed = time.time()-since
    minutes = time_elapsed//60
    seconds = time_elapsed %60
    
    print(f'Training complete in {minutes}m {seconds:.4f}s\n\n')
    print("Best Validation Accuracy Achieved: {:.4f}".format(best_acc))
    
    #Load best model weights
    model.load_state_dict(best_model_wts)
    print("Returned the best model", "\n", "-"*20)
    return model


trained_model = train_model(model, criterion,optimizer, exp_lr_scheduler, num_epochs=1)
