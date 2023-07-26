import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class tacNet(nn.Module):
    def __init__(self, args):
        super(tacNet, self).__init__() 
        if args.window == 0: # 9x11
            self.conv_0_left = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.ReLU())
            self.conv_0_right = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(3,3),padding=1),
                nn.ReLU())
        else:
            self.conv_0_left = nn.Sequential(
                nn.Conv2d(args.window, 32, kernel_size=(3,3),padding=1),
                nn.ReLU())
            self.conv_0_right = nn.Sequential(
                nn.Conv2d(args.window, 32, kernel_size=(3,3),padding=1),
                nn.ReLU()) # 7x9
                

        self.conv_1_left = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.ReLU())
        self.conv_1_right = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3,3),padding=1),
            nn.ReLU()) # 5x7

        self.conv_11_left = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.ReLU())
        self.conv_11_right = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3,3),padding=1),
            nn.ReLU())

        self.conv_111_left = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3),padding=1),
            nn.ReLU())
        self.conv_111_right = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=(3,3),padding=1),
            nn.ReLU())
            

        self.conv_2_left = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3),padding=1),
            nn.ReLU())
        self.conv_2_right = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=(3,3),padding=1),
            nn.ReLU()) #3x5


        self.cls_0 = nn.Sequential(
            nn.Linear(6336, 480), #
            nn.ReLU())

        self.cls_1 = nn.Sequential(
            nn.Linear(480, 240),
            nn.ReLU())

        self.cls_2 = nn.Sequential(
            nn.Linear(240, args.cls))
            # nn.Sigmoid())

        self.softmax = nn.Softmax(dim=1)



    def forward(self, tac, eval):
        tac_left = tac[:, :, :, :11] # B X N X 9 X 11
        tac_right = tac[:, :, :, 11:]
        #9x11
        left = self.conv_0_left(tac_left) #7x9
        left = self.conv_1_left(left) #5x7
        # left = self.conv_11_left(left) #5x7
        # left = self.conv_111_left(left) #5x7
        left = self.conv_2_left(left) #3x5

        right = self.conv_0_right(tac_right)
        right = self.conv_1_right(right)
        # right = self.conv_11_right(right)
        # right = self.conv_111_right(right)
        right = self.conv_2_right(right)

        # print(output.shape)

        left = left.reshape(left.shape[0],-1)
        right = right.reshape(right.shape[0],-1)

        output = torch.cat((left, right), 1)

        if eval:
            feature = torch.clone(output)

        output = self.cls_0(output)
        output = self.cls_1(output)
        output = self.cls_2(output)

        # output = self.softmax(output)

        # print (output.shape, torch.sum(output))

        if eval:
            return output, feature
        else:
            return output

    