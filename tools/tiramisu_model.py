"""
Implementation of the One Hundred Layers Tiramisu as described in
The One Hundred Layers Tiramisu: Fully Convolutional DenseNets for Semantic Segmentation.

Tiramisu is a Fully Convolutional Networks (FCN) network based on DenseNet architecture.

Papers:
Tiramisu: https://arxiv.org/pdf/1611.09326.pdf
FCN: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf
DenseNet: https://arxiv.org/abs/1608.06993
"""

from keras.models import Model
from keras.layers import Reshape
from keras.regularizers import l2
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dropout

def relu_bn(x):
    x = Activation('relu')(x)
    #x = LeakyReLU()(x)
    return BatchNormalization()(x)

def conv(x, num_filters, size, wd, dr, stride=1):
    x = Conv2D(num_filters, (size, size), kernel_initializer='he_uniform', padding='same',
                      strides=(stride,stride), kernel_regularizer=l2(wd))(x)
    if dr:
        return Dropout(dr)(x)
    else:
        return x

def conv_relu_bn(x, num_filters, size=3, wd=0, dr=0, stride=1):
    return conv(relu_bn(x), num_filters, size, wd=wd, dr=dr, stride=stride)

def dense_block(n, x, growth_rate, dr, wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, dr=dr, wd=wd)
        x = concatenate([x, b])
        added.append(b)
    return x, added

def transition_dn(x, dr, wd):
    # Original idea from the paper has MaxPooling2D(strides=(2, 2)) after Conv2D without stride
    #x = conv_relu_bn(x, x.get_shape().as_list()[-1], size=1, dr=dr, wd=wd)
    #return MaxPooling2D(strides=(2, 2))(x)

    # Try stride 2 1x1 convolution instead
    return conv_relu_bn(x, x.get_shape().as_list()[-1], size=1, dr=dr, wd=wd, stride=2)

def down_path(x, num_layers, growth_rate, dr, wd):
    skips = []
    for i, n in enumerate(num_layers):
        x, added = dense_block(n, x, growth_rate, dr, wd)
        skips.append(x)
        x = transition_dn(x, dr=dr, wd=wd)
    return skips, added

def transition_up(added, wd):
    x = concatenate(added)
    return Conv2DTranspose(x.get_shape().as_list()[-1], (3,3), kernel_initializer='he_uniform',
               padding='same', strides=(2,2), kernel_regularizer=l2(wd))(x)

def up_path(added, skips, num_layers, growth_rate, dr, wd):
    for i, n in enumerate(num_layers):
        x = transition_up(added, wd)
        x = concatenate([x,skips[i]])
        x, added = dense_block(n, x, growth_rate, dr, wd)
    return x

"""
init_num_filter: initial number of filters
num_layers_per_block: list of number of layers in each dense block
growth_rate: number of filters to add per dense block
dr: dropout rate
wd: weight decay
"""
def get_tiramisu_model(input_tensor, num_classes,
    init_num_filter=48, num_layers_per_block=[4,5,7,10,12,15], growth_rate=16, dr=0.2, wd=1e-4):

    x = conv(input_tensor, init_num_filter, 3, wd, 0)
    skips, added = down_path(x, num_layers_per_block, growth_rate, dr, wd)
    x = up_path(added, list(reversed(skips[:-1])), list(reversed(num_layers_per_block[:-1])), growth_rate, dr, wd)

    x = conv(x, num_classes, 1, wd, 0)
    outputs = Activation('softmax')(x)

    model = Model(inputs=[input_tensor], outputs=[outputs])
    return model
