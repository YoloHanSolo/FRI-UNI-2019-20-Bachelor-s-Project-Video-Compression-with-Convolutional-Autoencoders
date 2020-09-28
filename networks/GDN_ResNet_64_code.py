import warnings
warnings.filterwarnings("ignore")

import torch, torchvision
import torch.nn as nn
from torchvision import transforms

from networks.gdn_class import GDN

class GDN_Net(nn.Module):

    def __init__(self, device):
        super(GDN_Net, self).__init__()
        self.device = device

        self.conv_1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)   
        self.conv_2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  
        self.conv_3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)   
        self.conv_4 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)  
        self.conv_5 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1) 
        # MID
        self.T_conv_5 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) 
        self.T_conv_4 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1) 
        self.T_conv_3 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)   
        self.T_conv_2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)  
        self.T_conv_1 = nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1)  

        self.gdn = GDN(64, device)
        self.igdn = GDN(64, device, inverse=True )

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform(m.weight)
                #nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.gdn(x)
        x = self.conv_2(x)
        x = self.gdn(x)    
        x = self.conv_3(x)
        x = self.gdn(x)
        x = self.conv_4(x)
        x = self.gdn(x)
        x = self.conv_5(x)
        x = self.sigmoid(x)
        # MID
        x = self.T_conv_5(x)
        x = self.igdn(x)
        x = self.T_conv_4(x)
        x = self.igdn(x)
        x = self.T_conv_3(x)
        x = self.igdn(x)    
        x = self.T_conv_2(x)
        x = self.igdn(x)
        x = self.T_conv_1(x)
        return x

    def encode(self, x):
        x = self.conv_1(x)
        x = self.gdn(x)
        x = self.conv_2(x)
        x = self.gdn(x)    
        x = self.conv_3(x)
        x = self.gdn(x)
        x = self.conv_4(x)
        x = self.gdn(x)
        x = self.conv_5(x)
        x = self.sigmoid(x)
        return x

    def decode(self, x):      
        x = self.T_conv_5(x)
        x = self.igdn(x)
        x = self.T_conv_4(x)
        x = self.igdn(x)
        x = self.T_conv_3(x)
        x = self.igdn(x)    
        x = self.T_conv_2(x)
        x = self.igdn(x)
        x = self.T_conv_1(x)
        return x