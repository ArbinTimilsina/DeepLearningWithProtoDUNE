# Inspired from https://github.com/roebius/deeplearning_keras2/blob/master/nbs2/tiramisu-keras.ipynb

from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers import BatchNormalization, Activation, Dense, Dropout
from keras.layers.convolutional import Conv2D, SeparableConv2D, UpSampling2D, Conv2DTranspose
from keras.regularizers import l2
from keras.layers import Reshape


def relu(x): return Activation('relu')(x)
def dropout(x, p): return Dropout(p)(x) if p else x
def bn(x): return BatchNormalization()(x)
def relu_bn(x): return relu(bn(x))
def concat(xs): return concatenate(xs)
def reverse(a): return list(reversed(a))

def conv(x, nf, sz, wd, p, stride=1):
    x = Conv2D(nf, (sz, sz), kernel_initializer='he_uniform', padding='same',   # Keras 2
                      strides=(stride,stride), kernel_regularizer=l2(wd))(x)
    return dropout(x, p)

def conv_relu_bn(x, nf, sz=3, wd=0, p=0, stride=1):
    return conv(relu_bn(x), nf, sz, wd=wd, p=p, stride=stride)


def dense_block(n,x,growth_rate,p,wd):
    added = []
    for i in range(n):
        b = conv_relu_bn(x, growth_rate, p=p, wd=wd)
        x = concat([x, b])
        added.append(b)
    return x,added

def transition_dn(x, p, wd):
#     x = conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd)  # - original idea from the paper
#     return MaxPooling2D(strides=(2, 2))(x)
    return conv_relu_bn(x, x.get_shape().as_list()[-1], sz=1, p=p, wd=wd, stride=2)  # - seems to work better

def down_path(x, nb_layers, growth_rate, p, wd):
    skips = []
    for i,n in enumerate(nb_layers):
        x,added = dense_block(n,x,growth_rate,p,wd)
        skips.append(x)
        x = transition_dn(x, p=p, wd=wd)
    return skips, added

def transition_up(added, wd=0):
    x = concat(added)
    _,r,c,ch = x.get_shape().as_list()
#     return Deconvolution2D(ch, 3, 3, (None,r*2,c*2,ch), init='he_uniform',
#                border_mode='same', subsample=(2,2), W_regularizer=l2(wd))(x)
    return Conv2DTranspose(ch, (3,3), kernel_initializer='he_uniform',    # Keras 2
               padding='same', strides=(2,2), kernel_regularizer=l2(wd))(x)
#     x = UpSampling2D()(x)
#     return conv(x, ch, 2, wd, 0)


def up_path(added, skips, nb_layers, growth_rate, p, wd):
    for i,n in enumerate(nb_layers):
        x = transition_up(added, wd)
        x = concat([x,skips[i]])
        x,added = dense_block(n,x,growth_rate,p,wd)
    return x

def create_tiramisu(nb_classes, img_input, nb_dense_block=6,
    growth_rate=16, nb_filter=48, nb_layers_per_block=5, p=None, wd=0):

    if type(nb_layers_per_block) is list or type(nb_layers_per_block) is tuple:
        nb_layers = list(nb_layers_per_block)
    else: nb_layers = [nb_layers_per_block] * nb_dense_block

    x = conv(img_input, nb_filter, 3, wd, 0)
    skips,added = down_path(x, nb_layers, growth_rate, p, wd)
    x = up_path(added, reverse(skips[:-1]), reverse(nb_layers[:-1]), growth_rate, p, wd)

    x = conv(x, nb_classes, 1, wd, 0)
    #_,r,c,f = x.get_shape().as_list()
    #x = Reshape((-1, nb_classes))(x)
    return Activation('softmax')(x)
