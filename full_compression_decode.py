import warnings
warnings.filterwarnings("ignore")

from py_files.video import Video
from py_files.elements import Frame, Add, Buffer, MotionCompensation
from py_files.quantizator import Quantizator
from networks.autoencoder_image     import Network_AEC
from networks.autoencoder_flow      import FlowCoder
from networks.autoencoder_residual  import ResidualCoder
from coder.arithmetic_encoder import ArithmeticCoder

frames = 100

# DEFINE COMPONENTS
decoded_frame = Frame()

image_frame = Frame()
flow_frame = Frame()
residual_frame = Frame()

buffer = Buffer()

# 0 - UNIFORM
# 1 - SIN
# 2 - NORMAL
# 3 - LAPLACE

image_q     = Quantizator(bins=10, type=1)
flow_q      = Quantizator(bins=2, type=1)
residual_q  = Quantizator(bins=8, type=1)

image_autoencoder       = Network_AEC(name="Image", tensor_shape=(128,8,8), quantizator=image_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_image_n2.pth")
flow_autoencoder        = FlowCoder(name= "Flow", tensor_shape=(64,8,8), quantizator=flow_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_flow_64_n.pth")
residual_autoencoder    = ResidualCoder(name="Residual", tensor_shape=(64,8,8), quantizator=residual_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_residual_64.pth")

image_arith_coder = ArithmeticCoder(tensor_shape=(128,8,8), output_name="full_compression/img_frame", adaptive=True, stats=16)
flow_arith_coder = ArithmeticCoder(tensor_shape=(64,8,8), output_name="full_compression/flow_frame", adaptive=True, stats=16)
residual_arith_coder = ArithmeticCoder(tensor_shape=(64,8,8), output_name="full_compression/res_frame", adaptive=True, stats=16)

motion_compensation = MotionCompensation()    
add_operator = Add()

# PIPELINE
for frame_index in range(0, frames):

    if frame_index % 10 == 0:     
        image_arith_coder.Decode(frame_index)
        image_frame.LoadCode(image_arith_coder)
        image_frame.SendStatsTo(image_autoencoder)
        image_autoencoder.DecodeFrame(image_autoencoder)
        
        decoded_frame.LoadFrame(image_autoencoder)

    else:
    
        flow_arith_coder.Decode(frame_index)
        flow_frame.LoadCode(flow_arith_coder)
        flow_frame.SendStatsTo(flow_autoencoder)
        flow_autoencoder.DecodeFrame(flow_autoencoder)
             
        motion_compensation.Calculate(decoded_frame, flow_autoencoder)
        
        residual_arith_coder.Decode(frame_index)
        residual_frame.LoadCode(residual_arith_coder)
        residual_frame.SendStatsTo(residual_autoencoder)
        residual_autoencoder.DecodeFrame(residual_autoencoder)

        add_operator.Calculate(motion_compensation, residual_autoencoder)
        decoded_frame.LoadFrame(add_operator)  

    decoded_frame.Show(1, "df")

    buffer.LoadFrame(decoded_frame)
    print("Frame ", frame_index)

buffer.Show()