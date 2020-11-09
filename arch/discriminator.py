import torch
import torch.nn as nn


class Discriminator(nn.Module):   ##############  PatchGAN
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 4, 2, 1)
        self.conv2 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv3 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv4 = nn.Conv2d(256, 512, 4, 2, 1)
        self.conv5 = nn.Conv2d(512, 1024, 4, 2, 1)
        self.conv6 = nn.Conv2d(1024, 2048, 4, 2, 1)

        self.conv7 = nn.Conv2d(2048,1,3,1,1)
        self.conv8 = nn.Conv2d(2048,5,2,1,0)

        self.leakyrelu = nn.LeakyReLU(0.01,inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.leakyrelu(x)
        x = self.conv2(x)
        x = self.leakyrelu(x)
        x = self.conv3(x)
        x = self.leakyrelu(x)
        x = self.conv4(x)
        x = self.leakyrelu(x)
        x = self.conv5(x)
        x = self.leakyrelu(x)
        x = self.conv6(x)
        x = self.leakyrelu(x)
        return self.conv7(x),self.sigmoid(self.conv8(x).view(x.size(0),-1))

