from keras.models import Model
from keras.layers.merge import concatenate
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D, SeparableConv2D, UpSampling2D, Conv2DTranspose
from keras.layers import BatchNormalization, Activation, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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

def upsample_2d(input_layer):
    output_layer = UpSampling2D((2,2))(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    return output_layer

def decoder_block(small_ip_layer, large_ip_layer, filters):
    upsampled_layer = upsample_2d(small_ip_layer)

    concatenated_layer = concatenate([upsampled_layer, large_ip_layer])

    output1_layer = separable_conv2d_batchnorm(concatenated_layer, filters)
    output2_layer = separable_conv2d_batchnorm(output1_layer,      filters)

    return output2_layer

def get_fcn_model(input_tensor, num_classes, num_filters=16):
    # With each encoder layer, the depth of FCN model (the number of filters) increases.
    encoder1_layer = encoder_block(input_tensor, 1*num_filters, strides=2)
    encoder2_layer = encoder_block(encoder1_layer, 2*num_filters, strides=2)
    encoder3_layer = encoder_block(encoder2_layer, 3*num_filters, strides=2)

    # Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encoder3_layer, 4*num_filters, kernel_size=1, strides=1)

    # Add the same number of Decoder Blocks as the number of Encoder Blocks
    decoder1_layer = decoder_block(conv_layer, encoder2_layer, 3*num_filters)
    decoder2_layer = decoder_block(decoder1_layer, encoder1_layer, 2*num_filters)
    x = decoder_block(decoder2_layer, input_tensor, 1*num_filters)

    outputs = Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    model = Model(inputs=[input_tensor], outputs=[outputs])
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

def train_model(model, X, y, num_training, num_validation, model_path, num_epochs=1, batch_size=1):
    # Stop training when a monitored quantity has stopped improving after certain epochs
    early_stop = EarlyStopping(monitor='val_loss', mode='min', patience=25, verbose=1)

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.25, patience=3, cooldown=0, verbose=1)

    # Save the best model after every epoch
    check_point = ModelCheckpoint(filepath=model_path, verbose=1, save_best_only=True, monitor='val_loss', mode='min')

    history = model.fit_generator(X,
                                  steps_per_epoch = num_training//batch_size,
                                  epochs=num_epochs,
                                  validation_data=y,
                                  validation_steps= num_validation//batch_size,
                                  verbose=2,
                                  callbacks=[check_point, early_stop, reduce_lr],
                                  shuffle=False,
                                  use_multiprocessing=False,
                                  workers=1)

    return history
