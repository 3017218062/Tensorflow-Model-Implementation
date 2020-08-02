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


def convBlock(x, growthRate, activation="relu", name="conv"):
    _x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_0_bn")(x)
    _x = Activation(activation, name=name + "_0_" + activation)(_x)
    _x = Conv2D(4 * growthRate, 1, use_bias=False,
                kernel_initializer=initializer,
                name=name + "_1_conv")(_x)
    _x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_1_bn")(_x)
    _x = Activation(activation, name=name + "_1_" + activation)(_x)
    _x = Conv2D(growthRate, 3, padding="same", use_bias=False,
                kernel_initializer=initializer,
                name=name + "_2_conv")(_x)
    x = concatenate([x, _x], axis=3)
    return x


def denseBlock(x, blocks, activation="relu", name="dense"):
    for i in range(blocks):
        x = convBlock(x, 32, activation=activation, name=name + "_block" + str(i + 1))
    return x


def transitionBlock(x, reduction, activation="relu", name="transition"):
    x = BatchNormalization(axis=3, epsilon=1.001e-5, name=name + "_bn")(x)
    x = Activation(activation, name=name + "_" + activation)(x)
    x = Conv2D(int(x.shape[3] * reduction), 1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_conv")(x)
    x = AveragePooling2D(2, strides=2, name=name + "_pool")(x)
    return x


def DenseNet(x, blocks, activation="relu"):
    x = ZeroPadding2D(padding=3)(x)
    x = Conv2D(64, 7, strides=2, use_bias=False,
               kernel_initializer=initializer,
               name="conv1/conv")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name="conv1/bn")(x)
    x = Activation(activation, name="conv1/" + activation)(x)
    x = ZeroPadding2D(padding=1)(x)
    x = MaxPooling2D(3, strides=2, name="pool1")(x)

    x = denseBlock(x, blocks[0], activation=activation, name="conv2")
    x = transitionBlock(x, 0.5, activation=activation, name="pool2")
    x = denseBlock(x, blocks[1], activation=activation, name="conv3")
    x = transitionBlock(x, 0.5, activation=activation, name="pool3")
    x = denseBlock(x, blocks[2], activation=activation, name="conv4")
    x = transitionBlock(x, 0.5, activation=activation, name="pool4")
    x = denseBlock(x, blocks[3], activation=activation, name="conv5")

    x = BatchNormalization(axis=3, epsilon=epsilon, name="bn")(x)
    x = Activation(activation, name=activation)(x)
    return x


def DenseNet121(x, activation="relu"):
    return DenseNet(x, blocks=[6, 12, 24, 16], activation=activation)


def DenseNet169(x, activation="relu"):
    return DenseNet(x, blocks=[6, 12, 32, 32], activation=activation)


def DenseNet201(x, activation="relu"):
    return DenseNet(x, blocks=[6, 12, 48, 32], activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = DenseNet121(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
