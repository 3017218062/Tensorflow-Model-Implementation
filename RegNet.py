import math
import numpy as np
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
RegnetYTypesConfig = {
    "RegNetY_200MF": {
        "stemWidth": 32,
        "w_a": 36.44,
        "w_0": 24,
        "w_m": 2.49,
        "groups": 8,
        "depth": 13,
        "hasSE": True,
    },
    "RegNetY_400MF": {
        "stemWidth": 32,
        "w_a": 27.89,
        "w_0": 48,
        "w_m": 2.09,
        "groups": 8,
        "depth": 16,
        "hasSE": True,
    },
    "RegNetY_600MF": {
        "stemWidth": 32,
        "w_a": 32.54,
        "w_0": 48,
        "w_m": 2.32,
        "groups": 16,
        "depth": 15,
        "hasSE": True,
    },
    "RegNetY_800MF": {
        "stemWidth": 32,
        "w_a": 38.84,
        "w_0": 56,
        "w_m": 2.4,
        "groups": 16,
        "depth": 14,
        "hasSE": True,
    },
    "RegNetY_1_6GF": {
        "stemWidth": 32,
        "w_a": 20.71,
        "w_0": 48,
        "w_m": 2.65,
        "groups": 24,
        "depth": 27,
        "hasSE": True,
    },
    "RegNetY_3_2GF": {
        "stemWidth": 32,
        "w_a": 42.63,
        "w_0": 80,
        "w_m": 2.66,
        "groups": 24,
        "depth": 21,
        "hasSE": True,
    },
    "RegNetY_4_0GF": {
        "stemWidth": 32,
        "w_a": 31.41,
        "w_0": 96,
        "w_m": 2.24,
        "groups": 64,
        "depth": 22,
        "hasSE": True,
    },
    "RegNetY_6_4GF": {
        "stemWidth": 32,
        "w_a": 33.22,
        "w_0": 112,
        "w_m": 2.27,
        "groups": 72,
        "depth": 25,
        "hasSE": True,
    },
    "RegNetY_8_0GF": {
        "stemWidth": 32,
        "w_a": 76.82,
        "w_0": 192,
        "w_m": 2.19,
        "groups": 56,
        "depth": 17,
        "hasSE": True,
    },
    "RegNetY_12GF": {
        "stemWidth": 32,
        "w_a": 73.36,
        "w_0": 168,
        "w_m": 2.37,
        "groups": 112,
        "depth": 19,
        "hasSE": True,
    },
    "RegNetY_16GF": {
        "stemWidth": 32,
        "w_a": 106.23,
        "w_0": 200,
        "w_m": 2.48,
        "groups": 112,
        "depth": 18,
        "hasSE": True,
    },
    "RegNetY_32GF": {
        "stemWidth": 32,
        "w_a": 115.89,
        "w_0": 232,
        "w_m": 2.53,
        "groups": 232,
        "depth": 20,
        "hasSE": True,
    }
}


def getRegNetConfig(regnetType, q=8):
    stemWidth, hasSE = regnetType["stemWidth"], regnetType["hasSE"]
    w_a, w_0, w_m, depth, groups = regnetType["w_a"], regnetType["w_0"], regnetType["w_m"], regnetType["depth"], \
                                   regnetType["groups"]

    ks = np.round(np.log((np.arange(depth) * w_a + w_0) / w_0) / np.log(w_m))
    widthPerStage = w_0 * np.power(w_m, ks)
    widthPerStage = (np.round(np.divide(widthPerStage, q)) * q).astype(np.int).tolist()

    ts_temp = zip(widthPerStage + [0], [0] + widthPerStage, widthPerStage + [0], [0] + widthPerStage)
    ts = [w != wp or r != rp for w, wp, r, rp in ts_temp]
    depthPerStage = np.diff([d for d, t in zip(range(len(ts)), ts) if t]).tolist()

    per_stage_width = np.unique(widthPerStage).tolist()

    groupsPerStage = [groups for _ in range(len(widthPerStage))]
    groupsPerStage = [min(per_g, per_w) for per_g, per_w in zip(groupsPerStage, per_stage_width)]
    widthPerStage = [int(round(per_w / per_g) * per_g) for per_w, per_g in zip(per_stage_width, groupsPerStage)]

    return stemWidth, hasSE, widthPerStage, depthPerStage, groupsPerStage


def seBlock(x, reduction=16):
    channels = x.shape[-1]
    _x = tf.identity(x)
    x = GlobalAveragePooling2D()(x)
    x = Conv2D(channels // reduction, 1, use_bias=False, activation="relu",
               kernel_initializer=initializer)(x[:, None, None, :])
    x = Conv2D(channels, 1, use_bias=False, activation="sigmoid",
               kernel_initializer=initializer)(x)
    x = Multiply()([_x, x])
    return x


def groupConv(x, filters, kernelSize, groups=32, strides=1, padding="valid", useBias=True, activation="linear"):
    cSplits = [filters // groups for _ in range(groups)]
    cSplits[0] += (filters - sum(cSplits))

    xSplits = [x.shape[-1] // groups for _ in range(groups)]
    xSplits[0] += (x.shape[-1] - sum(xSplits))
    xSplits = tf.split(x, xSplits, -1)

    xs = []
    for i in range(groups):
        xs.append(Conv2D(cSplits[i], kernelSize, strides=strides, padding=padding, use_bias=useBias,
                         kernel_initializer=initializer, activation=activation)(xSplits[i]))
    x = concatenate(xs, axis=-1)
    return x


def convbnBlock(x, filters=128, kernelSize=3, strides=1, padding="same", useBias=False, activation="relu"):
    x = Conv2D(filters, kernel_size=kernelSize, strides=strides, use_bias=useBias, padding=padding,
               kernel_initializer=initializer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x


def basicBlock(x, filters, strides=1, groups=1, neckRatio=1., hasSE=True, downSample=False, activation="relu"):
    _x = tf.identity(x)
    if downSample:
        _x = convbnBlock(_x, filters=filters, kernelSize=1, strides=2, padding="valid", useBias=True,
                         activation=activation)

    x = convbnBlock(x, filters=filters, kernelSize=1, strides=1, padding="valid", useBias=True, activation=activation)
    x = groupConv(x, int(filters * neckRatio), 3, groups=groups, strides=strides, padding="same", useBias=True)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    x = convbnBlock(x, filters=filters, kernelSize=1, strides=1, padding="valid", useBias=True, activation=activation)

    if hasSE:
        x = seBlock(x, reduction=4)
    x = Add()([_x, x])
    x = Activation(activation)(x)
    return x


def stageBlock(x, filters, strides=2, groups=8, blocks=2, hasSE=True, activation="relu"):
    for i in range(blocks):
        if i == 0:
            x = basicBlock(x, filters, strides=strides, groups=groups, hasSE=hasSE, downSample=True,
                           activation=activation)
        else:
            x = basicBlock(x, filters, strides=1, groups=groups, hasSE=hasSE, downSample=False, activation=activation)
    return x


def RegNet(x, stemWidth, widthPerStage, depthPerStage, groupsPerStage, hasSE=True, activation="relu"):
    x = convbnBlock(x, filters=stemWidth, kernelSize=3, strides=2, padding="same", useBias=True, activation=activation)
    x = stageBlock(x, widthPerStage[0], strides=2, blocks=depthPerStage[0], groups=groupsPerStage[0], hasSE=hasSE)
    x = stageBlock(x, widthPerStage[1], strides=2, blocks=depthPerStage[1], groups=groupsPerStage[1], hasSE=hasSE)
    x = stageBlock(x, widthPerStage[2], strides=2, blocks=depthPerStage[2], groups=groupsPerStage[2], hasSE=hasSE)
    x = stageBlock(x, widthPerStage[3], strides=2, blocks=depthPerStage[3], groups=groupsPerStage[3], hasSE=hasSE)
    return x


def RegNetY_200MF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_200MF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_400MF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_400MF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_600MF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_600MF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_800MF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_800MF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_1_6GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_1_6GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_3_2GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_3_2GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_4_0GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_4_0GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_6_4GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_6_4GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_8_0GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_8_0GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_12GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_12GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_16GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_16GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


def RegNetY_32GF(x, activation="relu"):
    config = getRegNetConfig(RegnetYTypesConfig["RegNetY_32GF"], q=8)
    return RegNet(x, config[0], config[2], config[3], config[4], hasSE=config[1], activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = RegNetY_200MF(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
