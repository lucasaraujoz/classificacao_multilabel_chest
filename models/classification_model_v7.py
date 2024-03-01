import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply

def squeeze_excite_block(tensor, ratio=16):
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

def classification_model_v7():
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
    x = tf.keras.layers.Dense(units=128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x_F)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model