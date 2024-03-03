import tensorflow as tf
import tensorflow.keras.backend as K
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply


def classification_model_v8():
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
    x_F = tf.keras.layers.GlobalAveragePooling2D()(concat)
    classification_fusion = tf.keras.layers.Dense(
        14, activation="sigmoid", name='fusion_global')(x_F)
    model = tf.keras.models.Model(
        inputs=img_input, outputs= classification_fusion)
    return model