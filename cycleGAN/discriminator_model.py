import numpy as np
import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, LeakyReLU 
from tensorflow_addons.layers import InstanceNormalization

import config

'''
Conv2D block
x -> Conv(N, kxk) -> InstanceNorm -> LeakyReLU -> y
'''
def Conv2D_LBlock(x, filters, strides, ksize):
    y = Conv2D(filters=filters,
                kernel_size=ksize,
                strides=strides,
                padding="valid")(x)
    y = InstanceNormalization(axis=-1, epsilon=K.epsilon())(y)
    y = LeakyReLU(alpha=config.LEAKY_RELU_COEF)(y)
    return y




def CreateDiscriminator(input_shape, filters=[32,64,128,192], striding=[1,1,1,1], ksize=3):
    # input layer
    input_t = Input(input_shape)
    # first conv
    layers =  LeakyReLU(alpha=config.LEAKY_RELU_COEF)(
      Conv2D(filters=filters[0],
                kernel_size=ksize,
                strides=2,
                padding="valid")(input_t)
    )
    # Conv blocks
    for stride, filt in zip(striding, filters):
        layers = Conv2D_LBlock(layers, filt, strides=stride, ksize=ksize)
    # last conv with sigmoid
    layers = Conv2D(filters=1, kernel_size=ksize, padding='valid', activation='sigmoid')(layers)

    return tf.keras.Model(inputs=input_t, outputs=layers, name='Discriminator-' + hex(id(input_t)))

def _check():
    input_shape = (32,32, 3)
    d = CreateDiscriminator(input_shape, filters=[32,64,128], striding=[1,1,1])
    d.compile()

    sample_input = np.random.sample((10, *input_shape))
    sample_out = d.predict(sample_input)

    print("Input shape:", sample_input.shape)
    print("Out shape:", sample_out.shape)
    d.summary()

if __name__ == "__main__":
    _check()



