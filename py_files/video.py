import numpy as np
import cv2, torchvision, os
from PIL import Image
from torchvision import transforms
       
class Video:
    def __init__(self, path):
        self.path = path
        
        self.width = None
        self.height = None
        self.channel = None
        self.frames = None  
        self.width_resized = None
        self.height_resized = None
        
        self.images = []
        self.images_out = []
        
        self.resize_resolution = tuple()
        
    def LoadVideo(self):
        if os.path.isdir(self.path):  
            for filename in os.listdir(self.path):
                image = cv2.imread(os.path.join(self.path,filename))
                if image is not None:
                    self.images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            self.width = self.images[0].shape[1]
            self.height = self.images[0].shape[0]
            self.channel = self.images[0].shape[2]
            self.frames = len(self.images)
        else:
            video = cv2.VideoCapture(self.path)
            while(video.isOpened()):
                ret, frame = video.read()
                if ret == True:
                    self.images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    break
            video.release()
            self.width = self.images[0].shape[1]
            self.height = self.images[0].shape[0]
            self.channel = self.images[0].shape[2]
            self.frames = len(self.images)

        
    def SaveVideo(self, framerate=30):
        video = cv2.VideoWriter("./video_out.avi", cv2.VideoWriter_fourcc(*'XVID'), framerate, (self.width_resized, self.height_resized))
        for i in range(len(self.images_out)): 
            video.write(self.images_out[i])
        video.release()
        
    def ResizeVideo(self, resize_resolution):
        self.width_resized = resize_resolution[0]
        self.height_resized = resize_resolution[1]
        transform = transforms.Compose([
            transforms.Resize(resize_resolution, interpolation=Image.BILINEAR),
        ])
        for index, image in enumerate(self.images):
            self.images[index] = np.asarray(transform(Image.fromarray(image))) 
            # IN ORDER FOR SS-SSIM AND MS-SSIM METRIC TO WORK PROPERLY WE MUST RESIZE USING TORCH.TRANSFORM
            #self.images[index] = cv2.resize(image, resize_resolution, interpolation = cv2.INTER_LINEAR)    
    
    def CropVideo(self, crop_size): # crop_size <- (top,bot,left,right)
        for index, image in enumerate(self.images):           
            self.images[index] = image[crop_size[0]:self.height-crop_size[1], crop_size[2]:self.width-crop_size[3]]

    def RotateVideo(self, rotate): # 90deg couner-clockwise       
        for index, image in enumerate(self.images):
            self.images[index] = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            #self.images[index] = np.ndarray((self.width_resized, self.height_resized, 3))
            #self.images[index][:,:,0] = np.rot90(image[:,:,0], rotate)
            #self.images[index][:,:,1] = np.rot90(image[:,:,1], rotate)
            #self.images[index][:,:,2] = np.rot90(image[:,:,2], rotate)

    def GetFrame(self, frame_index):
        return self.images[frame_index]
        
    def SaveFrame(self, component):
        self.images_out.append(cv2.cvtColor(component._out, cv2.COLOR_RGB2BGR))

    def Type(self):
        print("Init = Video\n\tOriginal Width = {}\n\tOriginal Height = {}\n\tResized Width = {}\n\tResized Height = {}\n\tChannels = {}\n\tFrames = {}\n"
        .format(self.width, self.height, self.width_resized, self.height_resized, self.channel, self.frames))