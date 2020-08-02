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


def block(x, filters, kernelSize=3, stride=1, groups=32, shortcut=True, activation="relu", name="block"):
    if shortcut:
        _x = Conv2D((64 // groups) * filters, 1, strides=stride, use_bias=False,
                    kernel_initializer=initializer,
                    name=name + "_0_conv")(x)
        _x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_0_bn")(_x)
    else:
        _x = x

    x = Conv2D(filters, 1, strides=1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_1_conv")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_1_bn")(x)
    x = Activation(activation, name=name + "_1_" + activation)(x)

    c = filters // groups
    x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
    x = DepthwiseConv2D(kernelSize, strides=stride, depth_multiplier=c, use_bias=False,
                        kernel_initializer=initializer,
                        name=name + "_2_conv")(x)
    xShape = x.shape[1:-1]
    x = Reshape(xShape + (groups, c, c))(x)
    x = Lambda(lambda x: sum([x[:, :, :, :, i] for i in range(c)]), output_shape=None, name=name + "_2_reduce")(x)
    x = Reshape(xShape + (filters,))(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_2_bn")(x)
    x = Activation(activation, name=name + "_2_" + activation)(x)

    x = Conv2D((64 // groups) * filters, 1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_3_conv")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_3_bn")(x)

    x = Add(name=name + "_add")([_x, x])
    x = Activation(activation, name=name + "_out")(x)
    return x


def stack(x, filters, blocks, stride=2, groups=32, activation="relu", name="stack"):
    x = block(x, filters, stride=stride, groups=groups, activation=activation, name=name + "_block1")
    for i in range(2, blocks + 1):
        x = block(x, filters, groups=groups, shortcut=False, activation=activation, name=name + "_block" + str(i))
    return x


def ResNet(x, stackFunc, preact, useBias, activation="relu"):
    x = ZeroPadding2D(padding=3, name="conv1_pad")(x)
    x = Conv2D(64, 7, strides=2, use_bias=useBias,
               kernel_initializer=initializer,
               name="conv1_conv")(x)

    if not preact:
        x = BatchNormalization(axis=3, epsilon=epsilon, name="conv1_bn")(x)
        x = Activation(activation, name="conv1_" + activation)(x)

    x = ZeroPadding2D(padding=1, name="pool1_pad")(x)
    x = MaxPooling2D(3, strides=2, name="pool1_pool")(x)

    x = stackFunc(x)

    if preact:
        x = BatchNormalization(axis=3, epsilon=epsilon, name="post_bn")(x)
        x = Activation(activation, name="post_" + activation)(x)
    return x


def ResNeXt50(x, activation="relu"):
    def stackFunc(x):
        x = stack(x, 64, 3, stride=1, activation=activation, name="conv2")
        x = stack(x, 128, 4, activation=activation, name="conv3")
        x = stack(x, 256, 6, activation=activation, name="conv4")
        x = stack(x, 512, 3, activation=activation, name="conv5")
        return x

    return ResNet(x, stackFunc, preact=False, useBias=False, activation=activation)


def ResNeXt101(x, activation="relu"):
    def stackFunc(x):
        x = stack(x, 64, 3, stride=1, activation=activation, name="conv2")
        x = stack(x, 128, 4, activation=activation, name="conv3")
        x = stack(x, 256, 23, activation=activation, name="conv4")
        x = stack(x, 512, 3, activation=activation, name="conv5")
        return x

    return ResNet(x, stackFunc, preact=False, useBias=False, activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = ResNeXt50(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
