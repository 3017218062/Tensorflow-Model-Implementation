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


def convbnBlock(x, filters, rows, cols, padding="same", strides=1, activation="relu", name=None):
    if name is not None:
        bnName = name + "_bn"
        convName = name + "_conv"
    else:
        bnName = None
        convName = None
    x = Conv2D(filters, (rows, cols), strides=strides, padding=padding, use_bias=False,
               kernel_initializer=initializer,
               name=convName)(x)
    x = BatchNormalization(axis=3, scale=False, name=bnName)(x)
    x = Activation(activation, name=name)(x)
    return x


def InceptionV3(x, activation="relu"):
    x = convbnBlock(x, 32, 3, 3, strides=2, padding="valid", activation=activation)
    x = convbnBlock(x, 32, 3, 3, padding="valid", activation=activation)
    x = convbnBlock(x, 64, 3, 3, activation=activation)
    x = MaxPooling2D(3, strides=2)(x)

    x = convbnBlock(x, 80, 1, 1, padding="valid", activation=activation)
    x = convbnBlock(x, 192, 3, 3, padding="valid", activation=activation)
    x = MaxPooling2D(3, strides=2)(x)

    # mixed 0: 35 x 35 x 256
    branch1x1 = convbnBlock(x, 64, 1, 1, activation=activation)

    branch5x5 = convbnBlock(x, 48, 1, 1, activation=activation)
    branch5x5 = convbnBlock(branch5x5, 64, 5, 5, activation=activation)

    branch3x3dbl = convbnBlock(x, 64, 1, 1, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)

    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = convbnBlock(branch_pool, 32, 1, 1, activation=activation)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed0")

    # mixed 1: 35 x 35 x 288
    branch1x1 = convbnBlock(x, 64, 1, 1, activation=activation)

    branch5x5 = convbnBlock(x, 48, 1, 1, activation=activation)
    branch5x5 = convbnBlock(branch5x5, 64, 5, 5, activation=activation)

    branch3x3dbl = convbnBlock(x, 64, 1, 1, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)

    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = convbnBlock(branch_pool, 64, 1, 1, activation=activation)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed1")

    # mixed 2: 35 x 35 x 288
    branch1x1 = convbnBlock(x, 64, 1, 1, activation=activation)

    branch5x5 = convbnBlock(x, 48, 1, 1, activation=activation)
    branch5x5 = convbnBlock(branch5x5, 64, 5, 5, activation=activation)

    branch3x3dbl = convbnBlock(x, 64, 1, 1, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)

    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = convbnBlock(branch_pool, 64, 1, 1, activation=activation)
    x = concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool], axis=3, name="mixed2")

    # mixed 3: 17 x 17 x 768
    branch3x3 = convbnBlock(x, 384, 3, 3, strides=2, padding="valid", activation=activation)

    branch3x3dbl = convbnBlock(x, 64, 1, 1, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, activation=activation)
    branch3x3dbl = convbnBlock(branch3x3dbl, 96, 3, 3, strides=2, padding="valid", activation=activation)

    branch_pool = MaxPooling2D(3, strides=2)(x)
    x = concatenate([branch3x3, branch3x3dbl, branch_pool], axis=3, name="mixed3")

    # mixed 4: 17 x 17 x 768
    branch1x1 = convbnBlock(x, 192, 1, 1, activation=activation)

    branch7x7 = convbnBlock(x, 128, 1, 1, activation=activation)
    branch7x7 = convbnBlock(branch7x7, 128, 1, 7, activation=activation)
    branch7x7 = convbnBlock(branch7x7, 192, 7, 1, activation=activation)

    branch7x7dbl = convbnBlock(x, 128, 1, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 128, 7, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 128, 1, 7, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 128, 7, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 192, 1, 7, activation=activation)

    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = convbnBlock(branch_pool, 192, 1, 1, activation=activation)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed4")

    # mixed 5, 6: 17 x 17 x 768
    for i in range(2):
        branch1x1 = convbnBlock(x, 192, 1, 1, activation=activation)

        branch7x7 = convbnBlock(x, 160, 1, 1, activation=activation)
        branch7x7 = convbnBlock(branch7x7, 160, 1, 7, activation=activation)
        branch7x7 = convbnBlock(branch7x7, 192, 7, 1, activation=activation)

        branch7x7dbl = convbnBlock(x, 160, 1, 1, activation=activation)
        branch7x7dbl = convbnBlock(branch7x7dbl, 160, 7, 1, activation=activation)
        branch7x7dbl = convbnBlock(branch7x7dbl, 160, 1, 7, activation=activation)
        branch7x7dbl = convbnBlock(branch7x7dbl, 160, 7, 1, activation=activation)
        branch7x7dbl = convbnBlock(branch7x7dbl, 192, 1, 7, activation=activation)

        branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
        branch_pool = convbnBlock(branch_pool, 192, 1, 1, activation=activation)
        x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed" + str(5 + i))

    # mixed 7: 17 x 17 x 768
    branch1x1 = convbnBlock(x, 192, 1, 1, activation=activation)

    branch7x7 = convbnBlock(x, 192, 1, 1, activation=activation)
    branch7x7 = convbnBlock(branch7x7, 192, 1, 7, activation=activation)
    branch7x7 = convbnBlock(branch7x7, 192, 7, 1, activation=activation)

    branch7x7dbl = convbnBlock(x, 192, 1, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 192, 7, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 192, 1, 7, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 192, 7, 1, activation=activation)
    branch7x7dbl = convbnBlock(branch7x7dbl, 192, 1, 7, activation=activation)

    branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
    branch_pool = convbnBlock(branch_pool, 192, 1, 1, activation=activation)
    x = concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool], axis=3, name="mixed7")

    # mixed 8: 8 x 8 x 1280
    branch3x3 = convbnBlock(x, 192, 1, 1, activation=activation)
    branch3x3 = convbnBlock(branch3x3, 320, 3, 3, strides=2, padding="valid", activation=activation)

    branch7x7x3 = convbnBlock(x, 192, 1, 1, activation=activation)
    branch7x7x3 = convbnBlock(branch7x7x3, 192, 1, 7, activation=activation)
    branch7x7x3 = convbnBlock(branch7x7x3, 192, 7, 1, activation=activation)
    branch7x7x3 = convbnBlock(branch7x7x3, 192, 3, 3, strides=2, padding="valid", activation=activation)

    branch_pool = MaxPooling2D(3, strides=2)(x)
    x = concatenate([branch3x3, branch7x7x3, branch_pool], axis=3, name="mixed8")

    # mixed 9: 8 x 8 x 2048
    for i in range(2):
        branch1x1 = convbnBlock(x, 320, 1, 1, activation=activation)

        branch3x3 = convbnBlock(x, 384, 1, 1, activation=activation)
        branch3x3_1 = convbnBlock(branch3x3, 384, 1, 3, activation=activation)
        branch3x3_2 = convbnBlock(branch3x3, 384, 3, 1, activation=activation)
        branch3x3 = concatenate([branch3x3_1, branch3x3_2], axis=3, name="mixed9_" + str(i))

        branch3x3dbl = convbnBlock(x, 448, 1, 1, activation=activation)
        branch3x3dbl = convbnBlock(branch3x3dbl, 384, 3, 3, activation=activation)
        branch3x3dbl_1 = convbnBlock(branch3x3dbl, 384, 1, 3, activation=activation)
        branch3x3dbl_2 = convbnBlock(branch3x3dbl, 384, 3, 1, activation=activation)
        branch3x3dbl = concatenate([branch3x3dbl_1, branch3x3dbl_2], axis=3)

        branch_pool = AveragePooling2D(3, strides=1, padding="same")(x)
        branch_pool = convbnBlock(branch_pool, 192, 1, 1, activation=activation)
        x = concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool], axis=3, name="mixed" + str(9 + i))

    return x


if __name__ == "__main__":
    i = Input((224, 224, 3))
    x = InceptionV3(i, activation="relu")
    x = GlobalAveragePooling2D()(x)
    o = Dense(10, activation="softmax")(x)

    model = Model(i, o)
    model.summary()
