import torch, torchvision, cv2
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils as utils
import numpy as np

from networks.GDN_ResNet_64_code import GDN_Net

class ResidualCoder:      
    def __init__(self, name, tensor_shape, quantizator, network_path):
        self.name = name
        self.tensor_shape = tensor_shape
        self.path = network_path
        self.Q = quantizator

        self.device = "cuda:0"
        self.model = GDN_Net(torch.device(self.device))
        self.model.load_state_dict(torch.load(self.path, map_location=self.device))
        self.model.cuda().eval()

        self._code = None
        self._out = None  

        self._std = None
        self._mean = None 
        self._min = None
        self._max = None  

        print(sum(p.numel() for p in self.model.parameters()))
        print(sum(p.numel() for p in self.model.parameters() if p.requires_grad))
              
        print("Init = Network {}\n\tPath = {}\n".format(self.name, self.path))
        
    def Show(self, wait):
        cv2.imshow("res_frame", ((self._out+255)/2).astype(np.uint8))  
        cv2.waitKey(wait)

    def ProcessFrame(self, component):
        res_frame = component._out/255 # LOAD FRAME
        
        self._mean = np.mean(res_frame)  
        self._std = np.std(res_frame)
        res_frame = (res_frame - self._mean) / self._std   

        res_frame = torch.unsqueeze(torch.from_numpy(np.transpose(res_frame, (2,0,1))), 0).to(self.device)

        with torch.no_grad():   
            encoded = self.model.encode(res_frame)
            
            ##
            self._max = torch.max(encoded).item()
            self._min = torch.min(encoded).item()
            ##            

            quantized = self.Q.quantize_tensor(x=encoded, std=self._std, left=self._min, right=self._max)
            self._code = np.ravel(quantized.cpu().numpy().astype(np.uint8))
            dequantized = self.Q.dequantize_tensor(x=quantized, std=self._std, left=self._min, right=self._max)
            decoded = self.model.decode(dequantized) 
        
        self._out = np.transpose(np.squeeze(decoded.cpu().numpy()), (1,2,0)).astype(np.float32)
        self._out = ((self._out * self._std) + self._mean)*255

    def EncodeFrame(self, component):
        res_frame = component._out/255 # LOAD FRAME  
        self._mean = np.mean(res_frame)  
        self._std = np.std(res_frame)
        res_frame = (res_frame - self._mean) / self._std   
        res_frame = torch.unsqueeze(torch.from_numpy(np.transpose(res_frame, (2,0,1))), 0).to(self.device)
      

        with torch.no_grad():   
            encoded = self.model.encode(res_frame)

        ##
        self._max = torch.max(encoded).item()
        self._min = torch.min(encoded).item()
        ##  
        
        quantized = self.Q.quantize_tensor(x=encoded, std=self._std, left=self._min, right=self._max)
        self._code = np.ravel(quantized.cpu().numpy().astype(np.uint8))

    def DecodeFrame(self, component):
        code = np.reshape(component._code, (1, self.tensor_shape[0], self.tensor_shape[1], self.tensor_shape[2]))
        code = torch.from_numpy(code).to(self.device)
        dequantized = self.Q.dequantize_tensor(x=code, std=self._std, left=self._min, right=self._max)
        
        with torch.no_grad():   
            decoded = self.model.decode(dequantized)
        
        self._out = np.transpose(np.squeeze(decoded.cpu().numpy()), (1,2,0)).astype(np.float32)
        self._out = ((self._out * self._std) + self._mean)*255