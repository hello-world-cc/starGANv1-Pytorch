import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,3,stride=1,padding=1)
        self.IN = nn.InstanceNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel,out_channel,3,1,1)

    def forward(self, x):
        y = x
        x = self.conv1(x)
        x = self.IN(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.IN(x)
        return x+y


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(3+5,64,7,1,3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.block = BasicBlock(256,256)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 7, 1, 3),
            nn.Tanh()
        )

    def forward(self, x, c):
        c = c.view(c.size(0),-1,1,1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat((x,c),dim=1)
        x = self.downsample(x)
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        x = self.block(x)
        x = self.upsample(x)
        return x