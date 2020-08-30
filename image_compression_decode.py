import warnings
warnings.filterwarnings("ignore")

from py_files.elements import Frame, Buffer
from py_files.quantizator import Quantizator
from networks.autoencoder_image import Network_AEC
from coder.arithmetic_encoder import ArithmeticCoder

frames = 450

# DEFINE COMPONENTS
decoded_frame = Frame()
image_frame = Frame()

buffer = Buffer()

# 0 - UNIFORM
# 1 - SIN
# 2 - NORMAL
# 3 - LAPLACE

image_q = Quantizator(bins=3, type=1)

image_autoencoder = Network_AEC(name="AEC", tensor_shape=(128,8,8), quantizator=image_q, network_path="C:/Users/jnpel/OneDrive/Diploma/Framework/networks/autoencoder_image_n.pth")
image_arith_coder = ArithmeticCoder(tensor_shape=(128,8,8), output_name="image_compression/frame", adaptive=True, stats=16)

# PIPELINE
for frame_index in range(0, frames):

    image_arith_coder.Decode(frame_index)
    image_frame.LoadCode(image_arith_coder)
    image_frame.SendStatsTo(image_autoencoder)
    image_autoencoder.DecodeFrame(image_autoencoder)
    
    decoded_frame.LoadFrame(image_autoencoder)

    decoded_frame.Show(1, "df")

    buffer.LoadFrame(decoded_frame)
    print("Frame ", frame_index)

buffer.Show()