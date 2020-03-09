
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from base import BaseModel
import math

class SpectralNet(nn.Module):
    """ SubSpectralNet architecture """
    def __init__(self):
        super().__init__()

        # init the layers
        self.conv1 = nn.ModuleList([nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=3) for _ in range(7)])
        self.conv1_bn = nn.ModuleList([nn.BatchNorm1d(32) for _ in range(7)])
        self.mp1 = nn.ModuleList([nn.MaxPool1d((3)) for _ in range(7)])
        self.drop1 = nn.ModuleList([nn.Dropout(0.1) for _ in range(7)])
        self.conv2 = nn.ModuleList([nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=3) for _ in range(7)])
        self.conv2_bn = nn.ModuleList([nn.BatchNorm1d(64) for _ in range(7)])
        self.mp2 = nn.ModuleList([nn.MaxPool1d(4) for _ in range(7)])
        self.drop2 = nn.ModuleList([nn.Dropout(0.1) for _ in range(7)])
        self.fc1 = nn.ModuleList([nn.Linear(21*64, 32) for _ in range(7)])
        self.drop3 = nn.ModuleList([nn.Dropout(0.3) for _ in range(7)])
        self.fc2 = nn.Linear(32*7, 2)




    def forward(self, x):
        # for every sub-spectrogram
        intermediate=[]
        input_=x 
        for i in range(input_.shape[1]):
            x= input_[:, i,:, :].squeeze(1)
            x = self.conv1[i](x.permute(0,2,1)) # batch_sz x n_chan x n_fft_bins
            x = self.conv1_bn[i](x)
            x = F.relu(x)
            x = self.mp1[i](x)
            x = self.drop1[i](x)
    
            x = self.conv2[i](x)
            x = self.conv2_bn[i](x)
            x = F.relu(x)
            x = self.mp2[i](x)
            x = self.drop2[i](x)
            x = x.view(-1, 21*64)
            #first dense layer , channel-wise
            x = self.fc1[i](x)
            x = self.drop3[i](x)
            # extracted intermediate layers
            intermediate.append(x)
            
     
        x = torch.cat((intermediate), 1)
        x= self.fc2(x)
        x = F.log_softmax(x, dim=1)
        return  x
 