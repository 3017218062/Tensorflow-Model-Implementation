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


def block(x, filters, kernelSize=3, stride=1, shortcut=False, activation="relu", name="block"):
    preact = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_preact_bn")(x)
    preact = Activation(activation, name=name + "_preact_" + activation)(preact)

    if shortcut:
        _x = Conv2D(4 * filters, 1, strides=stride,
                    kernel_initializer=initializer,
                    name=name + "_0_conv")(preact)
    else:
        _x = MaxPooling2D(1, strides=stride)(x) if stride > 1 else x

    x = Conv2D(filters, 1, strides=1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_1_conv")(preact)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_1_bn")(x)
    x = Activation(activation, name=name + "_1_" + activation)(x)

    x = ZeroPadding2D(padding=1, name=name + "_2_pad")(x)
    x = Conv2D(filters, kernelSize, strides=stride, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_2_conv")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_2_bn")(x)
    x = Activation(activation, name=name + "_2_" + activation)(x)

    x = Conv2D(4 * filters, 1,
               kernel_initializer=initializer,
               name=name + "_3_conv")(x)
    x = Add(name=name + "_out")([_x, x])
    return x


def stack(x, filters, blocks, stride=2, activation="relu", name="stack"):
    x = block(x, filters, shortcut=True, activation=activation, name=name + "_block1")
    for i in range(2, blocks):
        x = block(x, filters, activation=activation, name=name + "_block" + str(i))
    x = block(x, filters, stride=stride, activation=activation, name=name + "_block" + str(blocks))
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


def ResNet50(x, activation="relu"):
    def stackFunc(x):
        x = stack(x, 64, 3, stride=1, activation=activation, name="conv2")
        x = stack(x, 128, 4, activation=activation, name="conv3")
        x = stack(x, 256, 6, activation=activation, name="conv4")
        x = stack(x, 512, 3, activation=activation, name="conv5")
        return x

    return ResNet(x, stackFunc, preact=True, useBias=True, activation=activation)


def ResNet101(x, activation="relu"):
    def stackFunc(x):
        x = stack(x, 64, 3, stride=1, activation=activation, name="conv2")
        x = stack(x, 128, 4, activation=activation, name="conv3")
        x = stack(x, 256, 23, activation=activation, name="conv4")
        x = stack(x, 512, 3, activation=activation, name="conv5")
        return x

    return ResNet(x, stackFunc, preact=True, useBias=True, activation=activation)


def ResNet152(x, activation="relu"):
    def stackFunc(x):
        x = stack(x, 64, 3, stride=1, activation=activation, name="conv2")
        x = stack(x, 128, 8, activation=activation, name="conv3")
        x = stack(x, 256, 36, activation=activation, name="conv4")
        x = stack(x, 512, 3, activation=activation, name="conv5")
        return x

    return ResNet(x, stackFunc, preact=True, useBias=True, activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = ResNet50(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
    print(getFlops(model))
