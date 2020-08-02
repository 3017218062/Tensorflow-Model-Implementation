import math
import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.activations import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.initializers import *
from tensorflow.keras.utils import get_custom_objects

epsilon = 1.001e-5
initializer = glorot_uniform(seed=2020)


def GhostModule(x, filters, ratio, convkernels, dwkernels, padding='same', strides=1, useBias=False,
                activation="relu"):
    moreFliters = math.ceil(filters * 1.0 / ratio)
    x = Conv2D(moreFliters, convkernels, strides=strides, padding=padding, activation=activation, use_bias=useBias)(x)
    if (ratio == 1.): return x
    dw = DepthwiseConv2D(dwkernels, strides, padding=padding, depth_multiplier=ratio - 1, activation=activation,
                         use_bias=useBias)(x)
    dw = dw[:, :, :, :int(filters - moreFliters)]
    # dw = Lambda(slices, arguments={'channel': int(outchannels - conv_out_channel)})(dw)
    x = concatenate([x, dw], axis=-1)
    return x
