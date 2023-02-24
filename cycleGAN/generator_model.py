
import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Add, ReLU
from tensorflow_addons.layers import InstanceNormalization



def Conv2D_LBlock(x, filters, ksize=3, strides=1, apply_act=True):
    y = Conv2D(filters=filters,
                kernel_size=ksize,
                strides=strides,
                padding="same")(x)
    y = InstanceNormalization(axis=-1, epsilon=K.epsilon())(y)
    if apply_act: y = ReLU()(y)
    return y


def DeConv2D_LBlock(x, filters, ksize=3, strides=1, apply_act=True):
    y = Conv2DTranspose(filters=filters,
                kernel_size=ksize,
                strides=strides,
                padding="same")(x)
    y = InstanceNormalization(axis=-1, epsilon=K.epsilon())(y)
    if apply_act: y = ReLU()(y)
    return y    


def Residual1_LBlock(x, filters, ksize=3):
    y = Conv2D_LBlock(x, filters, ksize)
    y = Conv2D_LBlock(y, filters, ksize, apply_act=False)
    return Add()([x, y])



        

def CreateGenerator(input_shape, filters=32, residual_blocks=12, ksize=3):
    input_t = Input(input_shape)
    # base
    x = Conv2D_LBlock(input_t, filters, ksize=round(2.3*ksize))
    # 'downsamling' blocks
    x = Conv2D_LBlock(x, 2*filters, ksize=ksize, strides=2)
    x = Conv2D_LBlock(x, 4*filters, ksize=ksize, strides=2)
    # residual blocks
    for i in range(residual_blocks):
        x = Residual1_LBlock(x, 4*filters, ksize=ksize)
    # 'upsampling' blocks
    x = DeConv2D_LBlock(x, 3*filters, ksize=ksize, strides=2)
    x = DeConv2D_LBlock(x, 1*filters, ksize=ksize, strides=2)
    x = DeConv2D_LBlock(x, 1 if len(input_shape)<3 else input_shape[-1], ksize=ksize, strides=1, apply_act=False)
    x = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(x)

    return tf.keras.Model(inputs=input_t, outputs=x, name='Generator-' + hex(id(input_t)))






def _check():
    input_shape = (64,64, 3)
    g = CreateGenerator(input_shape, filters=1, residual_blocks=5)
    g.compile()

    sample_input = np.random.sample((10, *input_shape))
    sample_out = g.predict(sample_input)

    print("Input shape:", sample_input.shape)
    print("Out shape:", sample_out.shape)
    g.summary()

if __name__ == "__main__":
    _check()