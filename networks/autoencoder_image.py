import torch, torchvision, cv2
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np
from collections import namedtuple
from PIL import Image

#from networks.ResNet16_code import ResNet, BasicBlock, BasicBlockInvert
from networks.ResNet_GDN_code import ResNet

class Network_AEC:      
    def __init__(self, name, tensor_shape, quantizator, network_path):
        self.name = name
        self.tensor_shape = tensor_shape
        self.size = tensor_shape[0] *  tensor_shape[1] *  tensor_shape[2]
        self.Q = quantizator
        self.path = network_path

        self.device = "cuda:0"
        self.network = ResNet(torch.device(self.device))
        self.network.load_state_dict(torch.load(self.path, map_location=self.device))
        self.network.cuda().eval()
        
        self._code = None
        self._out = None   

        self._mean = None
        self._std = None
        self._min = None
        self._max = None     
               
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5], inplace=True)
        ]) 
                    
        print(sum(p.numel() for p in self.network.parameters()))
        print(sum(p.numel() for p in self.network.parameters() if p.requires_grad))

        print("Init = Network {}\n\tPath = {}\n".format(self.name, self.path))
        
    def Show(self, wait):
        cv2.imshow("frame", cv2.cvtColor(self._out, cv2.COLOR_RGB2BGR))  
        cv2.waitKey(wait)

    # ENCODE + DECODE
    def ProcessFrame(self, component):
        data_in = component._out # LOAD FRAME  
        transformed = torch.unsqueeze(self.image_transform(data_in), 0).to(torch.device(self.device)) # TRANSFORM + ADD BATCH DIMENSION + TO GPU

        with torch.no_grad():
            encoded = self.network.encode(transformed)

            ##
            self._mean = torch.mean(encoded)
            self._std = torch.std(encoded, unbiased=False).item()
            self._max = torch.max(encoded).item()
            self._min = torch.min(encoded).item()
            ##

            quantized = self.Q.quantize_tensor(x=encoded, std=self._std, left=self._min, right=self._max)   
            self._code = np.ravel(quantized.cpu().numpy().astype(np.uint8)) 
            dequantized = self.Q.dequantize_tensor(quantized, std=self._std, left=self._min, right=self._max)    
            decoded = self.network.decode(dequantized)    

        decoded = ((decoded.data.cpu()).clamp(-1, 1)) * 0.5 + 0.5 # SEND TO CPU, LIMIT INTERVAL, DENORMALIZE
        self._out = np.transpose(np.squeeze(decoded.numpy()*255), (1, 2, 0)).astype(np.uint8) # 0,1 TO 0,255, REMOVE BATCH DIM, UINT8

    # ENCODE
    def EncodeFrame(self, component): 
        data_in = component._out # LOAD FRAME  
        transformed = torch.unsqueeze(self.image_transform(data_in), 0).to(torch.device(self.device)) # TRANSFORM + ADD BATCH DIMENSION + TO GPU

        with torch.no_grad():
            encoded = self.network.encode(transformed)
       
        ##
        self._mean = torch.mean(encoded)
        self._std = torch.std(encoded, unbiased=False).item()
        self._max = torch.max(encoded).item()
        self._min = torch.min(encoded).item()
        ##

        quantized = self.Q.quantize_tensor(x=encoded, std=self._std, left=self._min, right=self._max)   
        self._code = np.ravel(quantized.cpu().numpy().astype(np.uint8)) 

    # DECODE
    def DecodeFrame(self, component): 
        code = np.reshape(component._code, (1, self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2]))
        code = torch.from_numpy(code).to(self.device)

        dequantized = self.Q.dequantize_tensor(x=code, std=self._std, left=self._min, right=self._max)   

        with torch.no_grad():    
            decoded = self.network.decode(dequantized)    

        decoded = ((decoded.data.cpu()).clamp(-1, 1)) * 0.5 + 0.5 # SEND TO CPU, LIMIT INTERVAL, DENORMALIZE
        self._out = np.transpose(np.squeeze(decoded.numpy()*255), (1, 2, 0)).astype(np.uint8)
         # 0,1 TO 0,255, REMOVE BATCH DIM, UINT8

        