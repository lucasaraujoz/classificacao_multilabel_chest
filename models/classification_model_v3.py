import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score
import tensorflow as tf
import utils
import uuid
import classificacao_multilabel_chest.data as data
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K
from keras import layers
from keras import optimizers
from math import ceil
import config

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

def classification_model_v3():
    img_input = Input(shape=(256, 256, 3))
    # crop = tf.keras.layers.RandomCrop(width=224, height=224)(img_input)
    backbone = tf.keras.applications.EfficientNetV2S(
        include_top=False, weights=None, input_tensor=img_input)
    x =  backbone.output
    # se_block_out = squeeze_excite_block(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x) #se_block_out
    predictions_global = tf.keras.layers.Dense(
        len(data.LABELS), activation="sigmoid")(x)
    model_global = tf.keras.models.Model(
        inputs=img_input, outputs=predictions_global)
    return model_global
