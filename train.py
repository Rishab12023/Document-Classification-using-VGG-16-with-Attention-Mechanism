import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
import torch.optim.lr_scheduler as lr_scheduler

from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from network import AttnVGG

from earlyStopping import EarlyStopping
from utilities import *
# from transforms import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    
    root_dir = "/home/rishab/alexnet_attention/train"


    batch_size = 16
    learning_rate = 0.001

    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(root=root_dir,transform=transform)

    train_size = 0.8 
    train_data, val_data = train_test_split(dataset, train_size=train_size, shuffle=True)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False)

    

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    lr_lambda = lambda epoch : np.power(0.1, epoch//10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    writer = SummaryWriter("logs")

    model = AttnVGG(num_classes=16,normalize_attn=True)
    es = EarlyStopping(min_delta=0.00000001)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_val_loss = float("inf")
    num_epochs = 30


    # Training loop
    for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for images, labels in train_loader:
        inputs = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs,_,_= model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(train_loader)

    writer.add_scalar("Training Loss" , avg_train_loss,epoch)

    # Adjusting Learning Rate
    scheduler.step()

    model.eval()
    total = 0
    correct = 0
    val_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            inputs = images.to(device)
            labels = labels.to(device)
            outputs,_,_ = model(inputs)
            loss = criterion(outputs,labels)
            val_loss += loss.item()*images.size(0)
            _,predict = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            all_preds.extend(predict.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader.dataset)
    accuracy_val = 100*correct / total

    print(f'Epoch [{epoch+1}/{num_epochs}], Traning Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss: .4f}, Validation Accuracy,{accuracy_val:.2f}%')

    acc,val_mean_sensitivity,val_mean_precision,CM = compute_metrics(all_preds,all_labels,num_classes=16,epoch=epoch)
    writer.add_scalar('val/accuracy', acc*100, epoch)
    writer.add_scalar('val/mean_recall', val_mean_sensitivity,epoch)
    writer.add_scalar('val/precision_mel',val_mean_precision, epoch)
    writer.add_scalar("Validation Loss",val_loss,epoch)
    fig = plt.figure(figsize=(20,10))
    sns.heatmap(CM, annot=True, cmap="coolwarm")

    # Add the figure to the SummaryWriter
    writer.add_figure("heatmap", fig,global_step=epoch)

    # writer.close()

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        checkpoint_path = '/home/rishab/alexnet_attention/saved_model_1'
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path ,'best_model.pth')
        torch.save(model.state_dict(), checkpoint_path)

    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    fixed_batch = images[0:16, :, :, :].to(device)
    log_images = True
    writer = SummaryWriter("logs")
    if log_images:
        
        I_train = utils.make_grid(inputs[0:16, :, :, :], nrow=4, normalize=True, scale_each=True)
        writer.add_image('train/image', I_train , global_step = epoch)
        

        I_val = utils.make_grid(fixed_batch, nrow=4, normalize=True, scale_each=True)
        writer.add_image('val/image', I_val,global_step = epoch)

    base_up_factor = 8

    if log_images:
        __, a1, a2 = model(inputs[0:16,:,:,:])
        if a1 is not None:
            attn1 = visualize_attn(I_train, a1, up_factor=base_up_factor, nrow=4)
            writer.add_image('train/attention_map_1', attn1, global_step = epoch)
        if a2 is not None:
            attn2 = visualize_attn(I_train, a2, up_factor=2*base_up_factor, nrow=4)
            writer.add_image('train/attention_map_2', attn2,global_step= epoch)
        # val data
        __, a1, a2 = model(fixed_batch)
        if a1 is not None:
            attn1 = visualize_attn(I_val, a1, up_factor=base_up_factor, nrow=4)
            writer.add_image('val/attention_map_1', attn1, global_step = epoch)
        if a2 is not None:
            attn2 = visualize_attn(I_val, a2, up_factor=2*base_up_factor, nrow=4)
            writer.add_image('val/attention_map_2', attn2, global_step = epoch) 
    if epoch == num_epochs - 1:
        checkpoint_path = '/home/rishab/alexnet_attention/last_epoch_model'
        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_path ,'last_model.pth')
        torch.save(model.state_dict(), checkpoint_path)

    if es(model,val_loss):
        print("Early Stopping")

    print('Training finished.')

if __name__ == "__main__":
    main()