import cv2, struct
import numpy as np

class Frame():
    def __init__(self):
        self._out = None
        self._code = None

    def Show(self, wait, frame_name):
        cv2.imshow(frame_name, cv2.cvtColor(self._out, cv2.COLOR_RGB2BGR))  
        cv2.waitKey(wait)
        
    def LoadFrame(self, component):
        self._out = component._out
        
    def LoadCode(self, component):
        self._code = component._code

    def GetStatsFrom(self, component):
        mean = format(struct.unpack('!I', struct.pack('!f', component._mean))[0], '032b')
        std = format(struct.unpack('!I', struct.pack('!f', component._std))[0], '032b')
        min = format(struct.unpack('!I', struct.pack('!f', component._min))[0], '032b')
        max = format(struct.unpack('!I', struct.pack('!f', component._max))[0], '032b')
        
        
        stats = np.ndarray((16), dtype=np.uint8)
        count = 0
        for i in range(0, 32, 8):
            stats[0 + count] = int(mean[i:i+8], 2)
            stats[4 + count] = int(std[i:i+8], 2)
            stats[8 + count] = int(min[i:i+8], 2)
            stats[12 + count] = int(max[i:i+8], 2)     
            count += 1
        self._code = np.concatenate((self._code, stats))

    def SendStatsTo(self, component):
        code = self._code

        size = component.tensor_shape
        size = size[0] * size[1] * size[2]

        self._code = code[0:size]
        stats = code[size:len(code)]

        mean, std, min, max = "", "", "", ""
        for i in range(4):
            mean += '{0:08b}'.format(stats[0 + i])
            std += '{0:08b}'.format(stats[4 + i])
            min += '{0:08b}'.format(stats[8 + i])
            max += '{0:08b}'.format(stats[12 + i])
            
        mean = struct.unpack('!f',struct.pack('!I', int(mean, 2)))[0]
        std = struct.unpack('!f',struct.pack('!I', int(std, 2)))[0]
        min = struct.unpack('!f',struct.pack('!I', int(min, 2)))[0]
        max = struct.unpack('!f',struct.pack('!I', int(max, 2)))[0]
        
        component._mean = mean
        component._std = std
        component._min = min
        component._max = max
        component._code = self._code

class Buffer():
    def __init__(self):
        self.buffer = []
        print("Init = Buffer\n")

    def Show(self):
        for element in self.buffer:
            cv2.imshow("buffer", cv2.cvtColor(element, cv2.COLOR_RGB2BGR))  
            cv2.waitKey(40)
        
    def LoadFrame(self, component):
        self.buffer.append(component._out)
        
    def LoadCode(self, component):
        self.buffer.append(component._code)

class Subtract():
    def __init__(self):
        self._out = None
        self._sum = 0

    def Show(self, wait):
        cv2.imshow("frame_subtract", ((self._out+255)/2).astype(np.uint8))
        cv2.waitKey(wait)
        
    def Calculate(self, component_0, component_1):
        frame_1 = component_0._out
        frame_2 = component_1._out
        if frame_1 is not None and frame_2 is not None:
            self._out = np.subtract(frame_1.astype(np.float32), frame_2.astype(np.float32))
            self._sum = np.sum(self._out)
        else:
            print("{}: Input missing!\n".format(__class__.__name__))         

class Add():
    def __init__(self):
        self._out = None

    def Show(self, wait):
        cv2.imshow("frame_add", cv2.cvtColor(self._out, cv2.COLOR_RGB2BGR))  
        cv2.waitKey(wait)
        
    def Calculate(self, component_0, component_1):
        frame_1 = component_0._out
        frame_2 = component_1._out
        if frame_1 is not None and frame_2 is not None:
            self._out = np.add(frame_1.astype(np.float32),frame_2)
            self._out = np.clip(self._out, 0.0, 255.0).astype(np.uint8)
        else:
            print("{}: Input missing!\n".format(__class__.__name__)) 
            
class OpticalFlow():
    def __init__(self):
        self._out = None
        self.std = 0
        self.mean = 0
        
    def Show(self, wait):
        show_frame = np.ndarray((self._out.shape[0], self._out.shape[1], 3)) # CREATE EMPTY ARRAY
        show_frame[...,1] = 255 # SATURATION TO 255
        show_frame[...,0] = self._out[...,1] * 180/(2*np.pi) # CONVERT ANGLE 0 -> 180 TO 0 -> 255
        show_frame[...,2] = cv2.normalize(self._out[...,0], None, 0, 255, cv2.NORM_MINMAX)
        show_frame = show_frame.astype(np.uint8)
        show_frame = cv2.cvtColor(show_frame, cv2.COLOR_HSV2BGR)
        cv2.imshow("frame_of", show_frame)  
        cv2.waitKey(wait)

    def Save(self, name, flow_index):
        np.save("C:/Users/jnpel/Desktop/diploma/flow_frames/flow_{}_{}".format(name, flow_index), self._out.astype(np.float16))
        
    def Calculate(self, component_0, component_1):   
        frame_prev = component_0._out        
        frame_next = component_1._out         
        self._out = cv2.calcOpticalFlowFarneback(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2GRAY), cv2.cvtColor(frame_next, cv2.COLOR_BGR2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)

class MotionCompensation():
    def __init__(self):
        self._out = None
        
    def Show(self, wait):
        cv2.imshow("frame_mc", cv2.cvtColor(self._out, cv2.COLOR_RGB2BGR))  
        cv2.waitKey(wait)
        
    def Calculate(self, component_0, component_1):   
        frame = component_0._out
        flow = component_1._out
        flow = np.float32(flow)
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        self._out = np.copy(frame)
        self._out = cv2.remap(frame, flow, None, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) #cv2.BORDER_CONSTANT BORDER_TRANSPARENT BORDER_REFLECT
         
