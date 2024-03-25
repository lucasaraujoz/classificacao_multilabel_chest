import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply

def cbam_block(input_tensor, reduction_ratio=8):
    # Canal de atenção
    channel_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
    channel_max_pooling = tf.keras.layers.GlobalMaxPooling2D()(input_tensor)
    channel_concat = tf.keras.layers.Concatenate(axis=1)([channel_avg_pooling, channel_max_pooling])
    channel_dense_1 = tf.keras.layers.Dense(units=tf.keras.backend.int_shape(input_tensor)[-1] // reduction_ratio,
                                            activation='relu')(channel_concat)
    channel_dense_2 = tf.keras.layers.Dense(units=tf.keras.backend.int_shape(input_tensor)[-1])(channel_dense_1)
    channel_sigmoid = tf.keras.layers.Activation('sigmoid')(channel_dense_2)
    channel_attention = tf.keras.layers.Multiply()([input_tensor, tf.expand_dims(channel_sigmoid, axis=1)])

    # Atenção espacial
    spatial_avg_pooling = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=-1, keepdims=True))(channel_attention)
    spatial_max_pooling = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=-1, keepdims=True))(channel_attention)
    spatial_concat = tf.keras.layers.Concatenate(axis=-1)([spatial_avg_pooling, spatial_max_pooling])
    spatial_conv = tf.keras.layers.Conv2D(filters=1, kernel_size=(7, 7), padding='same', activation='sigmoid')(spatial_concat)
    spatial_attention = tf.keras.layers.Multiply()([channel_attention, spatial_conv])

    return spatial_attention

def se_block(input_feature, ratio=8):
	"""Contains the implementation of Squeeze-and-Excitation(SE) block.
	As described in https://arxiv.org/abs/1709.01507.
	"""
	
	channel_axis = 1 if K.image_data_format() == "channels_first" else -1
	channel = input_feature._keras_shape[channel_axis]

	se_feature = GlobalAveragePooling2D()(input_feature)
	se_feature = Reshape((1, 1, channel))(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	se_feature = Dense(channel // ratio,
					   activation='relu',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel//ratio)
	se_feature = Dense(channel,
					   activation='sigmoid',
					   kernel_initializer='he_normal',
					   use_bias=True,
					   bias_initializer='zeros')(se_feature)
	assert se_feature._keras_shape[1:] == (1,1,channel)
	if K.image_data_format() == 'channels_first':
		se_feature = Permute((3, 1, 2))(se_feature)

	se_feature = multiply([input_feature, se_feature])
	return se_feature

def squeeze_excite_block(tensor, ratio=4): #original=16
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu',
               kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid',
               kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def classification_model_v5():
    img_input = Input(shape=(256, 256, 3))
    crop = tf.keras.layers.RandomCrop(width=224, height=224)(img_input)
    resnet = tf.keras.applications.ResNet101(
        include_top=False, weights='imagenet', input_tensor=crop)
    resnet_out = resnet.output
    densenet = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=crop)

    for layer in densenet.layers:
        layer._name = layer._name + str("_2")

    densenet_out = densenet.output
    concat = Concatenate()([resnet_out, densenet_out])
    se_block_out = squeeze_excite_block(concat)
    x_F = tf.keras.layers.GlobalAveragePooling2D()(se_block_out)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model

def classification_model_v5_1():
    img_input = Input(shape=(256, 256, 3))
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input)
    resnet_out = resnet.output
    densenet = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=img_input)
    
    for layer in densenet.layers:
        layer._name = layer._name + str("_2")
    densenet_out = densenet.output
    resnet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(resnet_out)
    densenet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(densenet_out)
    concat = Concatenate()([resnet_branch_out, densenet_branch_out])
    x_F = tf.keras.layers.GlobalAveragePooling2D()(concat)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model

def classification_model_v5_2():
    img_input = Input(shape=(256, 256, 3))
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input)
    resnet_out = resnet.output
    densenet = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=img_input)
    
    for layer in densenet.layers:
        layer._name = layer._name + str("_2")
    densenet_out = densenet.output
    resnet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(resnet_out)
    densenet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(densenet_out)
    concat = Concatenate()([resnet_branch_out, densenet_branch_out])
    se_block_out = squeeze_excite_block(concat)
    x_F = tf.keras.layers.GlobalAveragePooling2D()(se_block_out)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model

def classification_model_v5_3():
    img_input = Input(shape=(256, 256, 3))
    resnet = tf.keras.applications.ResNet50V2(
        include_top=False, weights='imagenet', input_tensor=img_input)
    resnet_out = resnet.output
    densenet = tf.keras.applications.DenseNet121(
        include_top=False, weights='imagenet', input_tensor=img_input)
    
    for layer in densenet.layers:
        layer._name = layer._name + str("_2")
    densenet_out = densenet.output
    resnet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(resnet_out)
    densenet_branch_out = tf.keras.layers.Conv2D(1024,1,strides=1,padding='same')(densenet_out)
    concat = Concatenate()([resnet_branch_out, densenet_branch_out])
    cbam_output = cbam_block(concat)
    x_F = tf.keras.layers.GlobalAveragePooling2D()(cbam_output)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model