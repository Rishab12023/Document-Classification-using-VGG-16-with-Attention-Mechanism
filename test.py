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

    root_dir = "/test"

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    testset = datasets.ImageFolder(root=root_dir,transform=transform)

    testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False, num_workers=8)

    net = AttnVGG(num_classes=16,normalize_attn=True)
    checkpoint = torch.load("chechpoint.pth")
    net.load_state_dict(checkpoint["state_dict"])
    model = nn.DataParallel(net, device_ids=device_ids).to(device)
    model.eval()

    #Testing
    writer = SummaryWriter("logs")
    with torch.no_grad():
        with open("test_results.csv", "wt", newline="") as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=",")
            for images,labels in testloader:
                inputs = images.to(device)
                labels = labels.to(device)
                pred_test,_,_ = model(inputs)
                predict = torch.argmax(pred_test, 1)
                total += labels.size(0)
                correct += (predict == labels).sum().item()

                responses = F.softmax(pred_test, dim=1).squeeze().cpu().numpy()
                responses = [responses[i] for i in range(responses.shape[0])]
                csv_writer.writerows(responses)

    AP, AUC, precision_mean, precision_mel, recall_mean, recall_mel = compute_metrics('test_results.csv', 'test.csv')
    print("\ntest result: accuracy %.2f%%" % (100*correct/total))
    print("\nmean precision %.2f%% mean recall %.2f%% \nprecision for mel %.2f%% recall for mel %.2f%%" %
            (100*precision_mean, 100*recall_mean, 100*precision_mel, 100*recall_mel))
    print("\nAP %.4f AUC %.4f\n" % (AP, AUC))

    if __name__ == "__main__":
        main()        