from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dense, Dropout

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

def get_vgg16_fcn_model(base_model, num_classes):
    num_filters = 64
    input_tensor = base_model.input

    # Start block2 of VGG16 as first encoder layer
    encoder1_layer = base_model.get_layer('block2_pool').output
    encoder2_layer = base_model.get_layer('block3_pool').output
    encoder3_layer = base_model.get_layer('block4_pool').output
    encoder4_layer = base_model.get_layer('block5_pool').output

    # Make the decoder layers
    decoder1_layer = decoder_block(encoder4_layer, encoder2_layer, 4*num_filters, upsampling_size=(4,4))
    decoder2_layer = decoder_block(decoder1_layer, encoder1_layer, 2*num_filters)
    decoder3_layer = decoder_block(decoder2_layer, input_tensor, 1*num_filters, upsampling_size=(4,4))

    outputs = Conv2D(num_classes, (1, 1), activation='softmax', padding='same')(decoder3_layer)

    model = Model(inputs=[input_tensor], outputs=[outputs])
    return model

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

def get_fcn_with_vgg16_on_top_model(base_model, num_classes, num_filters=64):
    base_model_output = base_model.output
    # Output of VGG16 is 7x7x512; so upsample and transpose to get 224x224x3
    # Conv2DTranspose: new_rows = ((rows - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
    base_model_output = UpSampling2D(size=(2,2))(base_model_output)
    base_model_output = Conv2DTranspose(filters=300, kernel_size=(3,3), padding='same')(base_model_output)
    base_model_output = UpSampling2D(size=(2,2))(base_model_output) #28x28x300
    base_model_output = Conv2DTranspose(filters=150, kernel_size=(3, 3), padding='same')(base_model_output)
    base_model_output = UpSampling2D(size=(2,2))(base_model_output) #56x56x150
    base_model_output = Conv2DTranspose(filters=15, kernel_size=(3, 3), padding='same')(base_model_output)
    base_model_output = UpSampling2D(size=(2,2))(base_model_output) #112x112x15
    base_model_output = Conv2DTranspose(filters=3, kernel_size=(3, 3), padding='same')(base_model_output)
    input_tensor = UpSampling2D(size=(2,2))(base_model_output) #224x224x3

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

    model = Model(inputs=[base_model.input], outputs=[outputs])
    return model

def make_conv2d_block(input_tensor, num_filters, kernel_size=3, batchnorm=True):
    # First layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)

    # Second layer
    x = Conv2D(filters=num_filters, kernel_size=(kernel_size, kernel_size), kernel_initializer="he_normal",
               padding="same")(x)
    if batchnorm:
        x = BatchNormalization()(x)
    x = Activation("relu")(x)
    return x

def get_unet_model(input_tensor, num_classes, num_filters=16, dropout=0.25, batchnorm=True):
    # Vontracting path
    c1 = make_conv2d_block(input_tensor, num_filters=num_filters*1, kernel_size=3, batchnorm=batchnorm)
    p1 = MaxPooling2D((2, 2)) (c1)
    p1 = Dropout(dropout)(p1)

    c2 = make_conv2d_block(p1, num_filters=num_filters*2, kernel_size=3, batchnorm=batchnorm)
    p2 = MaxPooling2D((2, 2)) (c2)
    p2 = Dropout(dropout)(p2)

    c3 = make_conv2d_block(p2, num_filters=num_filters*4, kernel_size=3, batchnorm=batchnorm)
    p3 = MaxPooling2D((2, 2)) (c3)
    p3 = Dropout(dropout)(p3)

    c4 = make_conv2d_block(p3, num_filters=num_filters*8, kernel_size=3, batchnorm=batchnorm)
    p4 = MaxPooling2D(pool_size=(2, 2)) (c4)
    p4 = Dropout(dropout)(p4)

    c5 = make_conv2d_block(p4, num_filters=num_filters*16, kernel_size=3, batchnorm=batchnorm)

    # Expansive path
    u6 = Conv2DTranspose(num_filters*8, (3, 3), strides=(2, 2), padding='same') (c5)
    u6 = concatenate([u6, c4])
    u6 = Dropout(dropout)(u6)
    c6 = make_conv2d_block(u6, num_filters=num_filters*8, kernel_size=3, batchnorm=batchnorm)

    u7 = Conv2DTranspose(num_filters*4, (3, 3), strides=(2, 2), padding='same') (c6)
    u7 = concatenate([u7, c3])
    u7 = Dropout(dropout)(u7)
    c7 = make_conv2d_block(u7, num_filters=num_filters*4, kernel_size=3, batchnorm=batchnorm)

    u8 = Conv2DTranspose(num_filters*2, (3, 3), strides=(2, 2), padding='same') (c7)
    u8 = concatenate([u8, c2])
    u8 = Dropout(dropout)(u8)
    c8 = make_conv2d_block(u8, num_filters=num_filters*2, kernel_size=3, batchnorm=batchnorm)

    u9 = Conv2DTranspose(num_filters*1, (3, 3), strides=(2, 2), padding='same') (c8)
    u9 = concatenate([u9, c1], axis=3)
    u9 = Dropout(dropout)(u9)
    c9 = make_conv2d_block(u9, num_filters=num_filters*1, kernel_size=3, batchnorm=batchnorm)

    outputs = Conv2D(num_classes, (1, 1), activation='softmax') (c9)

    model = Model(inputs=[input_tensor], outputs=[outputs])
    return model
