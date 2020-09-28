import warnings
warnings.filterwarnings("ignore")

import torch, torchvision
import torch.nn as nn
from networks.gdn_class import GDN

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.conv2 =  nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):        
        identity = x   
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.lrelu(x)
        x += identity    
        return x

class SizeBlock(nn.Module):
    def __init__(self, inplanes, planes, device):
        super(SizeBlock, self).__init__()

        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=2, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.gdn = GDN(planes, device)

    def forward(self, x):        
        identity = x
        identity = self.conv3(identity)
        x = self.conv1(x)
        x = self.lrelu(x)
        x = self.conv2(x)
        x = self.gdn(x)
        x += identity    
        return x
        
class BasicBlockInvert(nn.Module):
    def __init__(self, inplanes, planes):
        super(BasicBlockInvert, self).__init__()

        self.T_conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=3, stride=1, padding=1)
        self.T_conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)

    def forward(self, x):        
        identity = x   
        x = self.T_conv1(x)
        x = self.lrelu(x)
        x = self.T_conv2(x)
        x = self.lrelu(x)
        x += identity    
        return x

class SizeBlockInvert(nn.Module):
    def __init__(self, inplanes, planes, device):
        super(SizeBlockInvert, self).__init__()

        self.T_conv1 = nn.ConvTranspose2d(inplanes, planes, kernel_size=4, stride=2, padding=1)
        self.T_conv2 = nn.ConvTranspose2d(planes, planes, kernel_size=3, stride=1, padding=1)
        self.T_conv3 = nn.ConvTranspose2d(inplanes, planes, kernel_size=2, stride=2, padding=0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.01, inplace=False)
        self.igdn = GDN(128, device, inverse=True)

    def forward(self, x):        
        identity = x
        identity = self.T_conv3(identity)
        x = self.T_conv1(x)
        x = self.lrelu(x)
        x = self.T_conv2(x)
        x = self.igdn(x)
        x += identity    
        return x

class ResNet(nn.Module):
    def __init__(self, device):
        super(ResNet, self).__init__()

        self.block_1 = SizeBlock(3, 128, device)
        self.block_2 = BasicBlock(128, 128)
        self.block_3 = SizeBlock(128, 128, device)
        self.block_4 = BasicBlock(128, 128)
        self.block_5 = SizeBlock(128, 128, device)
        self.block_6 = BasicBlock(128, 128)
        self.block_7 = SizeBlock(128, 128, device)
        self.block_8 = BasicBlock(128, 128)
        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()
        # MID
        self.T_block_1 = BasicBlockInvert(128, 128)
        self.T_block_2 = SizeBlockInvert(128, 128, device)
        self.T_block_3 = BasicBlockInvert(128, 128)
        self.T_block_4 = SizeBlockInvert(128, 128, device)
        self.T_block_5 = BasicBlockInvert(128, 128)
        self.T_block_6 = SizeBlockInvert(128, 128, device)
        self.T_block_7 = BasicBlockInvert(128, 128)
        self.T_block_8 = SizeBlockInvert(128, 128, device)
        self.T_block_9 = BasicBlockInvert(128, 128)
        self.T_conv = nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1)
        
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        # MID
        x = self.T_block_1(x)
        x = self.T_block_2(x)
        x = self.T_block_3(x)
        x = self.T_block_4(x)
        x = self.T_block_5(x)
        x = self.T_block_6(x)
        x = self.T_block_7(x)
        x = self.T_block_8(x)
        x = self.T_block_9(x)
        x = self.T_conv(x)
        return x

    def encode(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        x = self.block_6(x)
        x = self.block_7(x)
        x = self.block_8(x)
        x = self.conv(x)
        x = self.sigmoid(x)
        return x
        
    def decode(self, x):
        x = self.T_block_1(x)
        x = self.T_block_2(x)
        x = self.T_block_3(x)
        x = self.T_block_4(x)
        x = self.T_block_5(x)
        x = self.T_block_6(x)
        x = self.T_block_7(x)
        x = self.T_block_8(x)
        x = self.T_block_9(x)
        x = self.T_conv(x)
        return x