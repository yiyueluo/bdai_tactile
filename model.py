import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class tacNet(nn.Module):
    def __init__(self, windowSize):
        super(tacNet, self).__init__() 
        if windowSize == 0:
            self.conv_0_left = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
            self.conv_0_right = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
        else:
            self.conv_0_left = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
            self.conv_0_right = nn.Sequential(
                nn.Conv2d(windowSize, 32, kernel_size=(3,3),padding=1),
                nn.LeakyReLU(),
                nn.BatchNorm2d(32))
                

        self.conv_1_left = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)) # 48 * 48

        self.conv_1_right = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(64)) # 48 * 48

        self.conv_2_left = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))

        self.conv_2_right = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(128))


        self.cls_0 = nn.Sequential(
            nn.Linear(800,400),
            nn.LeakyReLU())

        self.cls_1 = nn.Sequential(
            nn.Linear(400,200),
            nn.LeakyReLU())

        self.cls_2 = nn.Sequential(
            nn.Linear(200,100),
            nn.LeakyReLU())



    def forward(self, tac_left, tac_right):
        left = self.conv_0(tac_left)
        left = self.conv_1(left)
        left = self.conv_2(left)

        right = self.conv_0(tac_right)
        right = self.conv_1(right)
        right = self.conv_2(right)

        # print(output.shape)

        left = left.reshape(left.shape[0],-1)
        right = left.reshape(right.shape[0],-1)

        output = torch.cat((left, right), 1)

        output = self.cls_0(output)
        output = self.cls_1(output)
        output = self.cls_2(output)

        # print (output.shape, torch.sum(output))

        return output