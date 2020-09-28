import warnings, sys
warnings.filterwarnings("ignore")

from py_files.video import Video
from py_files.elements import Frame, Subtract, Add, OpticalFlow, MotionCompensation
from py_files.quantizator import Quantizator
from networks.autoencoder_image     import Network_AEC
from networks.autoencoder_flow      import FlowCoder
from networks.autoencoder_residual  import ResidualCoder
from coder.arithmetic_encoder import ArithmeticCoder

from skimage.measure import compare_ssim

# VIDEO PREPROCESSING
video_sample = Video("video_mp4/tara.mp4")
video_sample.LoadVideo()
video_sample.ResizeVideo((256,256))
video_sample.RotateVideo(3)
video_sample.Type()
video_sample.frames = 400

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
flow_frame = Frame()
residual_frame = Frame()

# 0 - UNIFORM
# 1 - SIN
# 2 - NORMAL
# 3 - LAPLACE

image_q     = Quantizator(bins=6, type=1)
flow_q      = Quantizator(bins=2, type=1)
residual_q  = Quantizator(bins=6, type=1)

image_autoencoder       = Network_AEC(name="Image", tensor_shape=(128,8,8), quantizator=image_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_image_n2.pth")
flow_autoencoder        = FlowCoder(name= "Flow", tensor_shape=(64,8,8), quantizator=flow_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_flow_64_n.pth")
residual_autoencoder    = ResidualCoder(name="Residual", tensor_shape=(64,8,8), quantizator=residual_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_residual_64.pth")

image_arith_coder = ArithmeticCoder(tensor_shape=(128,8,8), output_name="full_compression/img_frame", adaptive=True, stats=16)
flow_arith_coder = ArithmeticCoder(tensor_shape=(64,8,8), output_name="full_compression/flow_frame", adaptive=True, stats=16)
residual_arith_coder = ArithmeticCoder(tensor_shape=(64,8,8), output_name="full_compression/res_frame", adaptive=True, stats=16)

optical_flow = OpticalFlow()
motion_compensation = MotionCompensation()    
subtract_operator = Subtract()
add_operator = Add()

# PIPELINE
for frame_index in range(0,250):#video_sample.frames):

    current_frame._out = video_sample.GetFrame(frame_index)

    if frame_index % 10 == 0:     
        image_autoencoder.EncodeFrame(current_frame)
        
        if save:
            image_frame.LoadCode(image_autoencoder)
            image_frame.GetStatsFrom(image_autoencoder)  
            image_arith_coder.Encode(frame_index, image_frame)
            image_frame.SendStatsTo(image_autoencoder)  
        image_autoencoder.DecodeFrame(image_autoencoder)
        
        decoded_frame.LoadFrame(image_autoencoder)
        
    else:
        optical_flow.Calculate(decoded_frame, current_frame)

        flow_autoencoder.EncodeFrame(optical_flow)
        if save:
            flow_frame.LoadCode(flow_autoencoder)
            flow_frame.GetStatsFrom(flow_autoencoder)  
            flow_arith_coder.Encode(frame_index, flow_frame)
            flow_frame.SendStatsTo(flow_autoencoder)  
        flow_autoencoder.DecodeFrame(flow_autoencoder)

        motion_compensation.Calculate(decoded_frame, flow_autoencoder)
        subtract_operator.Calculate(current_frame, motion_compensation)

        residual_autoencoder.EncodeFrame(subtract_operator)
        if save:
            residual_frame.LoadCode(residual_autoencoder)
            residual_frame.GetStatsFrom(residual_autoencoder)  
            residual_arith_coder.Encode(frame_index, residual_frame)
            residual_frame.SendStatsTo(residual_autoencoder)  
        residual_autoencoder.DecodeFrame(residual_autoencoder) 
        
        add_operator.Calculate(motion_compensation, residual_autoencoder)
        decoded_frame.LoadFrame(add_operator)  

    current_frame.Show(1, "of")
    decoded_frame.Show(1, "df")
    
    video_sample.SaveFrame(decoded_frame)
    
    ssim = compare_ssim(current_frame._out, decoded_frame._out, multichannel=True)
    ssim_sum += ssim

    print("Frame ", frame_index)

ssim_score = ssim_sum/video_sample.frames
print("\nAvg. SSIM = {:.3f}".format(ssim_score))

video_sample.SaveVideo()
if save:

    image_arith_coder.AvgFileSize()
    flow_arith_coder.AvgFileSize()
    residual_arith_coder.AvgFileSize()

    image_sum = image_arith_coder.bpp * image_arith_coder.files
    flow_sum = flow_arith_coder.bpp * flow_arith_coder.files
    residual_sum = residual_arith_coder.bpp * residual_arith_coder.files
    avg_bpp = (image_sum + flow_sum + residual_sum)/video_sample.frames
    print("Avg. bpp = {:.3f}".format(avg_bpp))