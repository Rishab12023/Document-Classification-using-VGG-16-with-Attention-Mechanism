import os
import csv
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    
    root_dir = "/home/rishab/train"
    # Hyperparameters
    batch_size = 16
    learning_rate = 0.001

    save_path = "/home/rishab/models"
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

    model = AttnVGG(num_classes=16,normalize_attn=True)
    es = EarlyStopping(min_delta=0.01)
    
    print('\nstart training ...\n')
    step = 0
    EMA_accuracy = 0
    AUC_val = 0
    writer = SummaryWriter("logs")

    best_val_loss = float("inf")
    num_epochs = 20

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-4)
    lr_lambda = lambda epoch : np.power(0.1, epoch//10)
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_train_loss:.4f}')
        writer.add_scalar("Training Loss" , avg_train_loss,epoch)

        # Adjusting Learning Rate
        scheduler.step()

        model.eval()
        total = 0
        correct = 0
        val_loss = 0
        with torch.no_grad():
            with open("val_results.csv", "wt", newline=" ") as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=",")

                for images, labels in val_loader:
                    inputs = images.to(device)
                    labels = labels.to(device)
                    outputs,_,_ = model(inputs)
                    loss = criterion(outputs,labels)
                    val_loss += loss.item()*images.size(0)
                    _,predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                    responses = F.softmax(predicted, dim = 1).squeeze().cpu().numpy()
                    responses = [responses[i] for i in range(responses.shape[0])]
                    csv_writer.writerows(responses)
                val_loss /= len(val_loader.dataset)
                accuracy_val = 100*correct / total
                print(f"Validation Accuracy,{accuracy_val:.2f}%" )
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {val_loss:.4f}')
                writer.add_scalar("Validation Loss",val_loss,epoch)

            AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics('val_results.csv', 'val.csv')
            # save checkpoints
            print('\nsaving checkpoints ...\n')
            checkpoint = {
                'state_dict': model.module.state_dict(),
                'opt_state_dict': optimizer.state_dict(),
            }
            torch.save(checkpoint, os.path.join(save_path, 'checkpoint_latest.pth'))
            if AUC > AUC_val: # save optimal validation model
                torch.save(checkpoint, os.path.join(save_path,'checkpoint.pth'))
                AUC_val = AUC
            # log scalars
            writer.add_scalar('val/accuracy', correct/total, epoch)
            writer.add_scalar('val/mean_precision', precision_mean, epoch)
            writer.add_scalar('val/mean_recall', recall_mean, epoch)
            writer.add_scalar('val/precision_mel', precision_mel, epoch)
            writer.add_scalar('val/recall_mel', recall_mel, epoch)
            writer.add_scalar('val/AP', AP, epoch)
            writer.add_scalar('val/AUC', AUC, epoch)
            print("\n[epoch %d] val result: accuracy %.2f%%" % (epoch+1, 100*correct/total))
            print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
                    (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
            print("\nAP %.4f AUC %.4f optimal AUC: %.4f\n" % (AP, AUC, AUC_val))
            if es(model,val_loss):
                print("Early Stopping")
                break


    data_iter = iter(val_loader)
    images, labels = next(data_iter)
    fixed_batch = images[0:16, :, :, :].to(device)
    log_images = True
    if log_images:
        print('\nlog images ...\n')
        I_train = utils.make_grid(inputs[0:16, :, :, :], nrow=4, normalize=True, scale_each=True)
        writer.add_image('train/image', I_train, epoch)
        
    
        I_val = utils.make_grid(fixed_batch, nrow=4, normalize=True, scale_each=True)
        writer.add_image('val/image', I_val, epoch)

    base_up_factor = 8

    if log_images:
        __, a1, a2 = model(inputs[0:16,:,:,:])
        if a1 is not None:
            attn1 = visualize_attn(I_train, a1, up_factor=base_up_factor, nrow=4)
            writer.add_image('train/attention_map_1', attn1, epoch)
        if a2 is not None:
            attn2 = visualize_attn(I_train, a2, up_factor=2*base_up_factor, nrow=4)
            writer.add_image('train/attention_map_2', attn2, epoch)
        # val data
        __, a1, a2 = model(fixed_batch)
        if a1 is not None:
            attn1 = visualize_attn(I_val, a1, up_factor=base_up_factor, nrow=4)
            writer.add_image('val/attention_map_1', attn1, epoch)
        if a2 is not None:
            attn2 = visualize_attn(I_val, a2, up_factor=2*base_up_factor, nrow=4)
            writer.add_image('val/attention_map_2', attn2, epoch)

    print('Training finished.')

if __name__ == "__main__":
    main()