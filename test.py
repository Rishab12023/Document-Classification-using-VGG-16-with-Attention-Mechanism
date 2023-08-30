import os
import csv
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
import torchvision
import torchvision.utils as utils
import torchvision.transforms as transforms
from network import AttnVGG

from utilities import *
# from transforms import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0]
def main():

    root_dir = "/home/rishab/alexnet_attention/test"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = datasets.ImageFolder(root=root_dir,transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=8)

    num_classes = 16

    net = AttnVGG(num_classes=16,normalize_attn=True)
    checkpoint = torch.load("last_model.pth")
    net.load_state_dict(checkpoint)
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()

    #Testing
    writer = SummaryWriter("logs")
    all_preds = []
    all_labels = []
    with torch.no_grad():
        total = 0
        correct = 0
        for images,labels in testloader:
            inputs = images.to(device)
            labels = labels.to(device)
            pred_test,_,_ = model(inputs)
            _,predict = torch.max(pred_test, 1)
            total += labels.size(0)
            correct += (predict == labels).sum().item()
            all_preds.extend(predict.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
    # Calculate and print accuracy
    print("Correct_Predictions",correct)
    print("Total",total)
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

    # Confusuion Matrix
    CM = confusion_matrix(all_labels, all_preds, labels=list(range(16)))


    acc = np.sum(np.diag(CM)) / np.sum(CM)

    print('\nTestset Accuracy(mean): %f %%' % (100 * acc))
    print()
    print('Confusion Matirx : ')
    print(CM)

    class_sensitivity = []
    class_precision = []
    class_metrics = []

    csv_filename = "test_results.csv"
    with open(csv_filename, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerow(["Class","Sensitivity","Precision"])
        for class_idx in range(num_classes):
            tp = CM[class_idx, class_idx]
            fn = np.sum(CM[class_idx, :]) - tp
            fp = np.sum(CM[:, class_idx]) - tp
            tn = np.sum(CM) - tp - fn - fp
            
            sensitivity = tp / (tp + fn)
            precision = tp / (tp + fp)
            class_sensitivity.append(sensitivity)
            class_precision.append(precision)

            writer.writerow([f"Class {class_idx}"])
            writer.writerow(["Sensitivity",sensitivity])
            writer.writerow(["Precision", precision])
            class_metrics.append([sensitivity, precision])
            
            # Write mean metrics
            mean_sensitivity = np.mean(class_sensitivity)
            mean_precision = np.mean(class_precision)
            writer.writerow(["Mean Sensitivity", mean_sensitivity])
            writer.writerow(["Mean Precision", mean_precision])

            print(f'Class {class_idx}:')
            print('- Sensitivity:', sensitivity)
            print('- Precision:', precision)
            print()
        
        # Write confusion matrix
        writer.writerow([])
        writer.writerow(["Confusion Matrix"])
        writer.writerows(CM)
    
    print('Mean Sensitivity:', mean_sensitivity)
    print('Mean Precision:', mean_precision)

    
    if __name__ == "__main__":
        main()        