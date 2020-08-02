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

ConvInitializer = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 2.0,
        "mode": "fan_out",
        "distribution": "normal"
    }
}

DenseInitializer = {
    "class_name": "VarianceScaling",
    "config": {
        "scale": 1. / 3.,
        "mode": "fan_out",
        "distribution": "uniform"
    }
}

ARGS = [
    {
        "kernelSize": 3, "repeats": 1, "ifilters": 32, "ofilters": 16,
        "expandRatio": 1, "idSkip": True, "strides": [1, 1], "seRatio": 0.25
    }, {
        "kernelSize": 3, "repeats": 2, "ifilters": 16, "ofilters": 24,
        "expandRatio": 6, "idSkip": True, "strides": [2, 2], "seRatio": 0.25
    }, {
        "kernelSize": 5, "repeats": 2, "ifilters": 24, "ofilters": 40,
        "expandRatio": 6, "idSkip": True, "strides": [2, 2], "seRatio": 0.25
    }, {
        "kernelSize": 3, "repeats": 3, "ifilters": 40, "ofilters": 80,
        "expandRatio": 6, "idSkip": True, "strides": [2, 2], "seRatio": 0.25
    }, {
        "kernelSize": 5, "repeats": 3, "ifilters": 80, "ofilters": 112,
        "expandRatio": 6, "idSkip": True, "strides": [1, 1], "seRatio": 0.25
    }, {
        "kernelSize": 5, "repeats": 4, "ifilters": 112, "ofilters": 192,
        "expandRatio": 6, "idSkip": True, "strides": [2, 2], "seRatio": 0.25
    }, {
        "kernelSize": 3, "repeats": 1, "ifilters": 192, "ofilters": 320,
        "expandRatio": 6, "idSkip": True, "strides": [1, 1], "seRatio": 0.25
    }
]


def roundFilters(filters, widthCoefficient, depthDivisor):
    filters *= widthCoefficient
    newFilters = int(filters + depthDivisor / 2) // depthDivisor * depthDivisor
    newFilters = max(depthDivisor, newFilters)
    if newFilters < 0.9 * filters:
        newFilters += depthDivisor
    return int(newFilters)


def roundRepeats(repeats, depthCoefficient):
    return int(math.ceil(depthCoefficient * repeats))


def seBlock(x, reducedFilters=32, filters=128, activation="relu"):
    _x = tf.identity(x)
    x = GlobalAveragePooling2D()(x)
    x = Conv2D(reducedFilters, 1, padding="same", use_bias=True, activation=activation,
               kernel_initializer=ConvInitializer)(x[:, None, None, :])
    x = Conv2D(filters, 1, padding="same", use_bias=True, activation="sigmoid",
               kernel_initializer=ConvInitializer)(x)
    x = Multiply()([_x, x])
    return x


def mbConvBlock(x, args, dropRate=None, activation="relu"):
    hasSE = (args["seRatio"] is not None) and (0 < args["seRatio"] <= 1)
    filters = int(args["ifilters"] * args["expandRatio"])
    _x = tf.identity(x)
    if args["expandRatio"] != 1:
        x = Conv2D(filters, 1, padding="same", use_bias=False,
                   kernel_initializer=ConvInitializer)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation(activation)(x)

    x = DepthwiseConv2D(args["kernelSize"], strides=args["strides"], padding="same", use_bias=False,
                        depthwise_initializer=ConvInitializer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    if hasSE:
        reducedFilters = max(1, int(args["ifilters"] * args["seRatio"]))
        x = seBlock(x, reducedFilters, filters, activation=activation)

    x = Conv2D(args["ofilters"], 1, padding="same", use_bias=False, kernel_initializer=ConvInitializer)(x)
    x = BatchNormalization(axis=3)(x)

    if args["idSkip"] and \
            all(s == 1 for s in args["strides"]) and \
            args["ifilters"] == args["ofilters"]:
        if dropRate and dropRate > 0:
            x = Dropout(dropRate, noise_shape=(None, 1, 1, 1))(x)
        x = Add()([_x, x])
    return x


def EfficientNet(x, widthCoefficient, depthCoefficient, dropConnectRate=0.2, depthDivisor=8,
                 blocksArgs=ARGS, activation="relu"):
    x = Conv2D(roundFilters(32, widthCoefficient, depthDivisor), 3, strides=(2, 2), padding="same", use_bias=False,
               kernel_initializer=ConvInitializer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)

    blocksTotal = sum(blockArgs["repeats"] for blockArgs in blocksArgs)
    blockNum = 0
    for idx, blockArgs in enumerate(blocksArgs):
        blockArgs["ifilters"] = roundFilters(blockArgs["ifilters"], widthCoefficient, depthDivisor)
        blockArgs["ofilters"] = roundFilters(blockArgs["ofilters"], widthCoefficient, depthDivisor)
        blockArgs["repeats"] = roundRepeats(blockArgs["repeats"], depthCoefficient)

        dropRate = dropConnectRate * float(blockNum) / blocksTotal
        x = mbConvBlock(x, blockArgs, dropRate=dropRate, activation=activation)
        blockNum += 1
        if blockArgs["repeats"] > 1:
            blockArgs["ifilters"] = blockArgs["ofilters"]
            blockArgs["strides"] = [1, 1]
            for bidx in range(blockArgs["repeats"] - 1):
                dropRate = dropConnectRate * float(blockNum) / blocksTotal
                x = mbConvBlock(x, blockArgs, dropRate=dropRate, activation=activation)
                blockNum += 1

    x = Conv2D(roundFilters(1280, widthCoefficient, depthDivisor), 1, padding="same", use_bias=False,
               kernel_initializer=ConvInitializer)(x)
    x = BatchNormalization(axis=3)(x)
    x = Activation(activation)(x)
    return x


def EfficientNetB0(x, activation="relu"):
    return EfficientNet(x, 1.0, 1.0, activation=activation)


def EfficientNetB1(x, activation="relu"):
    return EfficientNet(x, 1.0, 1.1, activation=activation)


def EfficientNetB2(x, activation="relu"):
    return EfficientNet(x, 1.1, 1.2, activation=activation)


def EfficientNetB3(x, activation="relu"):
    return EfficientNet(x, 1.2, 1.4, activation=activation)


def EfficientNetB4(x, activation="relu"):
    return EfficientNet(x, 1.4, 1.8, activation=activation)


def EfficientNetB5(x, activation="relu"):
    return EfficientNet(x, 1.6, 2.2, activation=activation)


def EfficientNetB6(x, activation="relu"):
    return EfficientNet(x, 1.8, 2.6, activation=activation)


def EfficientNetB7(x, activation="relu"):
    return EfficientNet(x, 2.0, 3.1, activation=activation)


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = EfficientNetB0(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
