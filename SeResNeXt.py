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


def seBlock(x, reduction=16, name="se"):
    channels = x.shape[-1]
    _x = GlobalAveragePooling2D(name=name + "_pooling")(x)
    _x = _x[:, None, None, :]
    _x = Conv2D(channels // reduction, 1,
                kernel_initializer=initializer,
                name=name + "_conv1")(_x)
    _x = Activation("relu")(_x)
    _x = Conv2D(channels, 1,
                kernel_initializer=initializer,
                name=name + "_conv2")(_x)
    _x = Activation("sigmoid")(_x)
    x = Multiply()([_x, x])
    return x


def groupConv(x, filters, kernelSize, strides=1, groups=32, useBias=True, activation="linear", padding="valid",
              name="groupconv"):
    cSplits = [filters // groups for _ in range(groups)]
    cSplits[0] += (filters - sum(cSplits))

    xSplits = [x.shape[-1] // groups for _ in range(groups)]
    xSplits[0] += (x.shape[-1] - sum(xSplits))
    xSplits = tf.split(x, xSplits, -1)

    xs = []
    for i in range(groups):
        xs.append(Conv2D(cSplits[i], kernelSize, strides=strides, padding=padding, use_bias=useBias,
                         kernel_initializer=initializer, bias_initializer=Zeros(),
                         activation=activation, name=name + "_layer%d" % i)(xSplits[i]))
    x = concatenate(xs, axis=3)
    return x


def bottleneck(x, filters, reduction=16, strides=1, groups=32, width=4, activation="relu", name="bottleneck"):
    _x = x
    w = (filters // 4) * width * groups // 64

    x = Conv2D(w, 1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_conv1")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_bn1")(x)
    x = Activation(activation)(x)

    x = ZeroPadding2D(1)(x)
    x = groupConv(x, w, 3, strides=strides, groups=groups, useBias=False, name=name + "_conv2")
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_bn2")(x)
    x = Activation(activation)(x)

    x = Conv2D(filters, 1, use_bias=False,
               kernel_initializer=initializer,
               name=name + "_conv3")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_bn3")(x)

    if strides != 1 or x.shape[-1] != _x.shape[-1]:
        _x = Conv2D(x.shape[-1], 1, strides=strides, use_bias=False,
                    kernel_initializer=initializer,
                    name=name + "_conv4")(_x)
        _x = BatchNormalization(axis=3, epsilon=epsilon, name=name + "_bn4")(_x)

    x = seBlock(x, reduction=reduction, name=name + "_se")

    x = Add()([x, _x])
    x = Activation(activation)(x)
    return x


def SENet(x, repetitions, filters=64, groups=1, reduction=16, activation="relu"):
    x = ZeroPadding2D(3)(x)
    x = Conv2D(filters, 7, strides=2, use_bias=False,
               kernel_initializer=initializer,
               name="conv1")(x)
    x = BatchNormalization(axis=3, epsilon=epsilon)(x)
    x = Activation(activation)(x)

    x = ZeroPadding2D(1)(x)
    x = MaxPooling2D(3, strides=2)(x)

    filters *= 2
    for i, stage in enumerate(repetitions):
        filters *= 2
        for j in range(stage):
            if i != 0 and j == 0:
                x = bottleneck(x, filters, reduction=reduction, strides=2, groups=groups,
                               name="bottleneck" + str(i) + "_" + str(j))
            else:
                x = bottleneck(x, filters, reduction=reduction, strides=1, groups=groups,
                               name="bottleneck" + str(i) + "_" + str(j))
    return x


def SEResNeXt50(x, activation="relu"):
    return SENet(x, repetitions=(3, 4, 6, 3), groups=32, reduction=16, activation=activation)


def SEResNeXt101(x, activation="relu"):
    return SENet(x, repetitions=(3, 4, 23, 3), groups=32, reduction=16, activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = SEResNeXt50(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
