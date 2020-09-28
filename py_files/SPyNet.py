import torch
import math
import numpy as np
import cv2

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 13) # requires at least pytorch version 1.3.0
torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

arguments_strModel = 'sintel-final'
backwarp_tenGrid = {}

def backwarp(tenInput, tenFlow):
    if str(tenFlow.size()) not in backwarp_tenGrid:
        tenHorizontal = torch.linspace(-1.0, 1.0, tenFlow.shape[3]).view(1, 1, 1, tenFlow.shape[3]).expand(tenFlow.shape[0], -1, tenFlow.shape[2], -1)
        tenVertical = torch.linspace(-1.0, 1.0, tenFlow.shape[2]).view(1, 1, tenFlow.shape[2], 1).expand(tenFlow.shape[0], -1, -1, tenFlow.shape[3])
        backwarp_tenGrid[str(tenFlow.size())] = torch.cat([ tenHorizontal, tenVertical ], 1).cuda()

    tenFlow = torch.cat([ tenFlow[:, 0:1, :, :] / ((tenInput.shape[3] - 1.0) / 2.0), tenFlow[:, 1:2, :, :] / ((tenInput.shape[2] - 1.0) / 2.0) ], 1)
    return torch.nn.functional.grid_sample(input=tenInput, grid=(backwarp_tenGrid[str(tenFlow.size())] + tenFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border', align_corners=True)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        class Preprocess(torch.nn.Module):
            def __init__(self):
                super(Preprocess, self).__init__()

            def forward(self, tenInput):
                tenBlue = (tenInput[:, 0:1, :, :] - 0.406) / 0.225
                tenGreen = (tenInput[:, 1:2, :, :] - 0.456) / 0.224
                tenRed = (tenInput[:, 2:3, :, :] - 0.485) / 0.229

                return torch.cat([ tenRed, tenGreen, tenBlue ], 1)

        class Basic(torch.nn.Module):
            def __init__(self, intLevel):
                super(Basic, self).__init__()

                self.netBasic = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
                    torch.nn.ReLU(inplace=False),
                    torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
                )

            def forward(self, tenInput):
                return self.netBasic(tenInput)

        self.netPreprocess = Preprocess()
        self.netBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])
        self.load_state_dict({ strKey.replace('module', 'net'): tenWeight for strKey, tenWeight in torch.load("C:/Users/jnpel/OneDrive/Diploma/Framework/networks/network-sintel-final.pytorch").items() })

    def forward(self, tenFirst, tenSecond):
        tenFlow = []
        tenFirst = [ self.netPreprocess(tenFirst) ]
        tenSecond = [ self.netPreprocess(tenSecond) ]

        for intLevel in range(5):
            if tenFirst[0].shape[2] > 32 or tenFirst[0].shape[3] > 32:
                tenFirst.insert(0, torch.nn.functional.avg_pool2d(input=tenFirst[0], kernel_size=2, stride=2, count_include_pad=False))
                tenSecond.insert(0, torch.nn.functional.avg_pool2d(input=tenSecond[0], kernel_size=2, stride=2, count_include_pad=False))

        tenFlow = tenFirst[0].new_zeros([ tenFirst[0].shape[0], 2, int(math.floor(tenFirst[0].shape[2] / 2.0)), int(math.floor(tenFirst[0].shape[3] / 2.0)) ])

        for intLevel in range(len(tenFirst)):
            tenUpsampled = torch.nn.functional.interpolate(input=tenFlow, scale_factor=2, mode='bilinear', align_corners=True) * 2.0
            if tenUpsampled.shape[2] != tenFirst[intLevel].shape[2]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
            if tenUpsampled.shape[3] != tenFirst[intLevel].shape[3]: tenUpsampled = torch.nn.functional.pad(input=tenUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')
            tenFlow = self.netBasic[intLevel](torch.cat([ tenFirst[intLevel], backwarp(tenInput=tenSecond[intLevel], tenFlow=tenUpsampled), tenUpsampled ], 1)) + tenUpsampled
            
        return tenFlow

class Network_OF:

    def __init__(self, name, input_size, network_path):
        self.name = name
        self.device = torch.device("cuda:0")
        self.path = network_path
        self.network = Network().cuda().eval() 
        self.input_size = input_size
        
        self.frame_1 = None
        self.frame_2 = None
        
        self._angle = None
        self._magnitude = None     
        
        self._out = None        
                 
        print("Network {} loaded!\n".format(self.name))
        
    def Show(self, wait):
        show_frame = np.ndarray((self.input_size[0],self.input_size[1],3)) # CREATE EMPTY ARRAY
        #self._magnitude, self._angle = cv2.cartToPolar(self._magnitude, self._angle) 
        print(np.amax(self._magnitude))
        print(np.amin(self._magnitude))
        show_frame[...,1] = 255 # SATURATION TO 255
        show_frame[...,0] = self._angle * 180/(2*np.pi) # CONVERT ANGLE 0 -> 180 TO 0 -> 255
        show_frame[...,2] = self._magnitude#cv2.normalize(self._magnitude, None, 0, 255, cv2.NORM_MINMAX)
        show_frame = show_frame.astype(np.uint8)
        show_frame = cv2.cvtColor(show_frame, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame", show_frame)  
        cv2.waitKey(wait)

    def ProcessFrame(self, components, frame_name):  
        self.frame_1 = components[frame_name[0]]._out # PREVIOUS FRAME 
        self.frame_2 = components[frame_name[1]]._out # CURRENT FRAME
        
        if self.frame_1 is not None and self.frame_2 is not None:
        
            preprocessed_1 = torch.FloatTensor(self.frame_1[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)) # TO TENSOR + TRANSFORM
            preprocessed_2 = torch.FloatTensor(self.frame_2[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) * (1.0 / 255.0)) # TO TENSOR + TRANSFORM          
            preprocessed_1 = preprocessed_1.cuda().view(1, 3, self.input_size[1], self.input_size[0]) # TO GPU
            preprocessed_2 = preprocessed_2.cuda().view(1, 3, self.input_size[1], self.input_size[0]) # TO GPU            
            procesed = self.network(preprocessed_1, preprocessed_2) # PROCESS FRAME THROUGH NETWORK  
            procesed = procesed[0, :, :, :].cpu().numpy() # TO CPU + TO NUMPY
                    
            self._magnitude = procesed[0,:,:]
            self._angle = procesed[1,:,:]
            
            self._out = np.ndarray((self.input_size[0],self.input_size[1],2))
            self._out[:,:,0] = self._angle
            self._out[:,:,1] = self._magnitude          
        else:
            print("Input missing!\n")
   
