import warnings, sys
warnings.filterwarnings("ignore")

from py_files.video import Video
from py_files.elements import Frame
from py_files.quantizator import Quantizator
from networks.autoencoder_image import Network_AEC
from coder.arithmetic_encoder import ArithmeticCoder

from skimage.measure import compare_ssim

# VIDEO PREPROCESSING
video_sample = Video("video_mp4/tara.mp4")
video_sample.LoadVideo()
video_sample.ResizeVideo((256,256))
#video_sample.RotateVideo(3)
video_sample.Type()
video_sample.frames = 100

"""
video_sample = Video("video_png/sample_video")
video_sample.LoadVideo()
video_sample.CropVideo((60,60,0,0))
video_sample.ResizeVideo((256,256))
video_sample.Type()
"""

if len(sys.argv) > 1:
    save = sys.argv[1]
else:
    save = False
    
ssim_sum = 0
ssim_score = 0

# DEFINE COMPONENTS
current_frame = Frame()
decoded_frame = Frame()

image_frame = Frame()

# 0 - UNIFORM
# 1 - SIN
# 2 - NORMAL
# 3 - LAPLACE

image_q = Quantizator(bins=2, type=0)

image_autoencoder = Network_AEC(name="AEC", tensor_shape=(128,8,8), quantizator=image_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_image_n2.pth")
image_arith_coder = ArithmeticCoder(tensor_shape=(128,8,8), output_name="image_compression/frame", adaptive=True, stats=16)

# PIPELINE
for frame_index in range(0,video_sample.frames):

    current_frame._out = video_sample.GetFrame(frame_index)

    image_autoencoder.EncodeFrame(current_frame)
    if save:
        image_frame.LoadCode(image_autoencoder)
        image_frame.GetStatsFrom(image_autoencoder)  
        image_arith_coder.Encode(frame_index, image_frame)
        image_frame.SendStatsTo(image_autoencoder)  
    image_autoencoder.DecodeFrame(image_autoencoder)
    
    decoded_frame.LoadFrame(image_autoencoder)

    current_frame.Show(1, "of")
    decoded_frame.Show(1, "df")

    ssim = compare_ssim(current_frame._out, decoded_frame._out, multichannel=True)
    ssim_sum += ssim

    print("Frame ", frame_index)

ssim_score = ssim_sum/video_sample.frames
print(ssim_score)

if save:
    image_arith_coder.AvgFileSize()