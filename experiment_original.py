import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, models, transforms
import time
import os
import copy
import wandb
import math
import sys


def experiment(config=None):
    
    pathDataset = 'C:/Users/56950/Documents/DCC/Research/KunischPatterns/KunischDataset/'

    train_dataset = torchvision.datasets.ImageFolder(pathDataset + 'train', 
                                                    transform = transforms.Compose([
                                                        transforms.RandomVerticalFlip(),
                                                        transforms.RandomHorizontalFlip(),
                                                        transforms.RandomResizedCrop(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std = [0.229, 0.224, 0.225])]))

    val_dataset = torchvision.datasets.ImageFolder(pathDataset + 'test',
                                                    transform = transforms.Compose([ transforms.Resize(256),
                                                                    transforms.CenterCrop(224),
                                                                    transforms.ToTensor(),
                                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                                        std = [0.229, 0.224, 0.225])]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.batch_size)

    class_names = train_dataset.classes

    device = ('cuda' if torch.cuda.is_available() else 'cpu')

    if config.network == 'resnet18':
        model_ft = models.resnet18(pretrained=True)
    elif config.network == 'resnet34':
        model_ft = models.resnet34(pretrained=True)
    elif config.network == 'resnet50':
        model_ft = models.resnet50(pretrained=True)

    num_ft = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ft, 6)

    model_ft = model_ft.to(device)
    criterion = nn.CrossEntropyLoss()

    if config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model_ft.parameters(), lr = config.learning_rate)
    elif config.optimizer == 'adam':
        optimizer = torch.optim.Adam(model_ft.parameters(), lr = config.learning_rate)

    #best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    wandb.watch(model_ft, criterion, log="all")

    for epoch in range(config.epochs):
        print(f'Epoch {epoch}/{config.epochs-1}')
        print('-' * 10)

        model_ft.train()

        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds ==  labels.data)
        
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)

        print('Train Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))
        wandb.log({'train_epoch':epoch+1, 'train_loss': epoch_loss, 'train_acc': epoch_acc})

        #Validation
        model_ft.eval()
        running_loss = 0.0
        running_corrects = 0.0

        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                outputs = model_ft(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_dataset)
        epoch_acc = running_corrects / len(val_dataset)
        print('Val Loss: {:.4f}  Acc: {:.4f}'.format(epoch_loss, epoch_acc))

        wandb.log({'test_epoch':epoch+1, 'test_loss': epoch_loss, 'test_acc': epoch_acc})

        if epoch_acc > best_acc:
            best_acc = epoch_acc
    
    wandb.log({'best_acc': best_acc})
        #    best_model_wts = copy.deepcopy(model.state_dict())
    
    #print('Best accuracy: {:.4f}'.format(best_acc))

    #model.load_state_dict(best_model_wts)


hyperparameter_defaults = dict(
    batch_size = 32,
    learning_rate = 0.001,
    epochs = 300,
    network = "resnet18",
    optimizer= "adam"
)

resume = sys.argv[-1] == "--resume"
wandb.init(config=hyperparameter_defaults, project='kunisch', entity='isipiran', resume=resume)
config = wandb.config

if __name__ == '__main__':
    train(config)

""" sweep_config = {
    'method': 'random',
    'metric': {'goal': 'maximize', 'name': 'test_acc'},
    'parameters': {
        'batch_size': {
            'distribution': 'q_log_uniform',
            'max': math.log(256),
            'min': math.log(32),
            'q': 1
        },
        'epochs': {'value': 300},
        'learning_rate': {'distribution': 'uniform',
            'max': 0.1,
            'min': 0},
        'optimizer': {'values': ['adam', 'sgd']},
        'network': {'values': ['resnet18','resnet34', 'resnet50']}
    }
}

#sweep_id = wandb.sweep(sweep_config, project="kunisch")
#wandb.agent(sweep_id, function=train, count=25)

print(sweep_config) """