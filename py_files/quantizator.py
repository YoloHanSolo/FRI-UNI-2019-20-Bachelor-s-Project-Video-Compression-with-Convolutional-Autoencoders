import torch, math
import numpy as np

class Quantizator:
    def __init__(self, bins, type):
        self.bins = bins
        self.type = type
        print("Init = Quantizator\nType = {}\nBins = {}\n".format(self.type, self.bins))

    def quantize_tensor(self, x, std=1, left=1, right=1):
        if self.type == 0: # UNIFORM
            return self.uniform_quantization(x, left, right)
        elif self.type == 1: # SIN
            return self.sin_quantization(x, left, right)
        elif self.type == 2: # NORMAL
            return self.normal_quantization(x, std)
        else: # LAPLACE
            return self.laplace_quantization(x, std)

    def dequantize_tensor(self, x, std=1, left=1, right=1):   
        if self.type == 0: # UNIFORM
            return self.uniform_dequantization(x, left, right)
        elif self.type == 1: # SIN
            return self.sin_dequantization(x, left, right)
        elif self.type == 2: # NORMAL
            return self.normal_dequantization(x, std)
        else: # LAPLACE
            return self.laplace_dequantization(x, std)


## NORMAL QUANTIZATION

    def normal_quantization(self, x, std):
        values = self.normal_quantization_borders(std)
        x = x.cpu()
        for i in range(1, self.bins+1):
            x = torch.where(x < torch.tensor(values[i-1]), torch.tensor(float(i)), x)
        x = (x-1).cuda()
        return x.byte()

    def normal_dequantization(self, x, std):
        values = self.normal_quantization_points(std)
        x = x.cpu().float()
        for i in range(0, self.bins):
            x = torch.where(x == torch.tensor(float(i)), torch.tensor(values[i]), x)
        return x.cuda()
    
    ## LAPLACE QUANTIZATION

    def laplace_quantization(self, x, std):
        values = self.laplace_quantization_borders(std)
        x = x.cpu()
        for i in range(1, self.bins+1):
            x = torch.where(x < torch.tensor(values[i-1]), torch.tensor(float(i)), x)
        x = (x-1).cuda()
        return x.byte()

    def laplace_dequantization(self, x, std):
        values = self.laplace_quantization_points(std)
        x = x.cpu().float()
        for i in range(0, self.bins):
            x = torch.where(x == torch.tensor(float(i)), torch.tensor(values[i]), x)
        return x.cuda()

    ## UNIFORM QUANTIZATION

    def uniform_quantization(self, x, left, right):
        q_scale = 1 / (self.bins-1)
        multiplier = 1/(right-left)
        x.clamp_(min=left, max=right)
        x = ((x-left) * multiplier) / q_scale
        x.clamp_(min=0, max=self.bins).round_().byte()
        return x

    def uniform_dequantization(self, x, left, right):
        q_scale = 1 / (self.bins+1)
        multiplier = 1/(right-left)
        x = (q_scale * (x+1).float())/multiplier + left
        return x

    ## SIN QUANTIZATION

    def sin_quantization(self, x, left, right):
        values = self.sin_quantization_borders()
        multiplier = 1/(right-left)
        x.clamp_(min=left, max=right)
        x = ((x-left) * multiplier)
        x = x.cpu()
        for i in range(1, self.bins+1):
            x = torch.where(x < torch.tensor(values[i-1]), torch.tensor(float(i)), x)
        x = (x-1).cuda()
        return x.byte()

    def sin_dequantization(self, x, left, right):
        values = self.sin_quantization_points()
        multiplier = 1/(right-left)
        x = x.cpu().float()
        for i in range(0, self.bins):
            x = torch.where(x == torch.tensor(float(i)), torch.tensor(values[i]), x)
        x = x/multiplier + left
        return x.cuda()

    ## OTHER FUNC

    def sin_quantization_borders(self):
        integral_sum, step = 0, 0.001      
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = (math.sin(math.pi*x) ** 2) * 2
            integral_sum += step * y
            if integral_sum >= sum_lim * (count+1):
                values[count] = x
                count += 1
        if count != num:
            values[count] = 1
        else:
            values[count-1] = 1
        return values

    def sin_quantization_points(self):
        integral_sum, step = 0, 0.001  
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = (math.sin(math.pi*x) ** 2) * 2
            integral_sum += step * y
            if integral_sum >= (sum_lim/2) * (2*count+1):
                values[count] = x
                count += 1
        return values

    ## UNIFORM

    def uniform_quantization(self, x, left, right):
        q_scale = 1 / (self.bins-1)
        multiplier = 1/(right-left)
        x.clamp_(min=left, max=right)
        x = ((x-left) * multiplier) / q_scale
        x.clamp_(min=0, max=self.bins).round_().byte()
        return x

    def uniform_dequantization(self, x, left, right):
        q_scale = 1 / (self.bins+1)
        multiplier = 1/(right-left)
        x = (q_scale * (x+1).float())/multiplier + left
        return x

    ## OTHER FUNC

    def normal_quantization_borders(self, std):
        integral_sum, step = 0, 0.001      
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = 1 / (std*math.sqrt(2*math.pi)) * (math.e ** ((-0.5) * (((x-0.5)/std) ** 2)))
            integral_sum += step * y
            if integral_sum >= sum_lim * (count+1):
                values[count] = x
                count += 1
        if count != num:
            values[count] = 1
        else:
            values[count-1] = 1
        return values

    def normal_quantization_points(self, std):
        integral_sum, step = 0, 0.001  
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = 1/(std*math.sqrt(2*math.pi)) * (math.e ** ((-0.5)* (((x-0.5)/std) ** 2)))
            integral_sum += step * y
            if integral_sum >= (sum_lim/2) * (2*count+1):
                values[count] = x
                count += 1
        return values

    def laplace_quantization_borders(self, std):
        integral_sum, step = 0, 0.001      
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = (1/(2*math.sqrt((std**2)/2))) * (math.e **(-(abs(x-0.5)/math.sqrt((std**2)/2)))) 
            integral_sum += step * y
            if integral_sum >= sum_lim * (count+1):
                values[count] = x
                count += 1
        if count != num:
            values[count] = 1
        else:
            values[count-1] = 1
        return values

    def laplace_quantization_points(self, std):
        integral_sum, step = 0, 0.001  
        num = self.bins
        sum_lim = 1/num
        values = np.ndarray((num))
        count = 0
        for x in np.arange(0.0, 1.0, step):
            y = (1/(2*math.sqrt((std**2)/2))) * (math.e **(-(abs(x-0.5)/math.sqrt((std**2)/2)))) 
            integral_sum += step * y
            if integral_sum >= (sum_lim/2) * (2*count+1):
                values[count] = x
                count += 1
        return values