from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization
from keras.layers.convolutional import Conv2D, SeparableConv2D, UpSampling2D

def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2D(filters=filters,kernel_size=3, strides=strides,
                                   padding='same', activation='relu')(input_layer)
    output_layer = BatchNormalization()(output_layer)
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                          padding='same', activation='relu')(input_layer)
    output_layer = BatchNormalization()(output_layer)
    return output_layer

def upsample_2d(input_layer, size=(2,2)):
    output_layer = UpSampling2D(size)(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters, upsampling_size=(2,2)):
    upsampled_layer = upsample_2d(small_ip_layer, upsampling_size)

    concatenated_layer = concatenate([upsampled_layer, large_ip_layer])

    output1_layer = separable_conv2d_batchnorm(concatenated_layer, filters)
    output2_layer = separable_conv2d_batchnorm(output1_layer, filters)

    return output2_layer

def get_fcn_model(input_tensor, num_classes, num_filters=64):
    # With each encoder layer, the depth of FCN model (the number of filters) increases.
    encoder1_layer = encoder_block(input_tensor, 1*num_filters, strides=2)
    encoder2_layer = encoder_block(encoder1_layer, 2*num_filters, strides=2)
    encoder3_layer = encoder_block(encoder2_layer, 4*num_filters, strides=2)

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder3_layer, 8*num_filters, kernel_size=1, strides=1)

    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1_layer = decoder_block(conv_layer, encoder2_layer, 4*num_filters)
    decoder2_layer = decoder_block(decoder1_layer, encoder1_layer, 2*num_filters)
    x = decoder_block(decoder2_layer, input_tensor, 1*num_filters)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(x)

    model = Model(inputs=[input_tensor], outputs=[outputs])
    return model

"""
from keras.applications.densenet import DenseNet121
input_tensor = Input((IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH))
base_model = DenseNet121(weights='imagenet', include_top=False, input_tensor=input_tensor)

DenseNet121: block_1: 1-6; block_2: 7-52; block_3: 53-140; block_4: 141-312; block_5: 313-426

# To freeze all base model's convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# To re-train some layers of base model
layer_no = 313
for layer in base_model.layers[:layer_no]:
    layer.trainable = False
for layer in base_model.layers[layer_no:]:
    layer.trainable = True
"""
def get_densenet121_fcn_model(base_model, num_classes):
    num_filters = 128
    input_tensor = base_model.input

    # Start pool1 of DenseNet121 as first encoder layer
    encoder1_layer = base_model.get_layer('pool1').output # 56x56x64
    encoder2_layer = base_model.get_layer('pool3_pool').output # 14x14x256
    encoder3_layer = base_model.get_layer('relu').output #7x7x1024

    # Make the decoder layers
    decoder1_layer = decoder_block(encoder3_layer, encoder2_layer, 4*num_filters) #14x14x512
    decoder2_layer = decoder_block(decoder1_layer, encoder1_layer, 2*num_filters, upsampling_size=(4,4)) #56x56x256
    decoder3_layer = decoder_block(decoder2_layer, input_tensor, 1*num_filters, upsampling_size=(4,4)) #224x224x128

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(decoder3_layer)

    model = Model(inputs=[input_tensor], outputs=[outputs])
    return model
