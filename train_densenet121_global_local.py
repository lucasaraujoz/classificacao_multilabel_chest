# VARS
BATCH_SIZE = 64
# TODO colocar variaveis: loss, optimizador, lr, batch,
# IMPORTS
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
import data
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K
from keras import layers
from keras import optimizers
from math import ceil


# TF CONFIGURATION
print("Número de GPUs disponíveis: ", len(
    tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


# LOAD DATASET AND SPLIT
df_train, df_test, df_val = data.split_dataset()

# generators global
train_generator_global = data.get_generator(
    df=df_train, x_col="path", shuffle=True, batch_size=BATCH_SIZE)
val_generator_global = data.get_generator(
    df=df_val, x_col="path", shuffle=False)
test_generator_global = data.get_generator(
    df=df_test, x_col="path", shuffle=False)

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

# BUILD MODEL
def global_branch(input_shape):
    densenet121 = tf.keras.applications.DenseNet121(
        input_shape=input_shape, include_top=False, weights="imagenet")
    densenet121._name = 'densenet121_global_branch'
    return densenet121

def create_model_global():
    input_shape = (224, 224, 3)
    g_model = global_branch(input_shape)
    # ____global model____
    x = g_model.output
    se_block_out = squeeze_excite_block(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(se_block_out) #se_block_out
    # #MLP
    # x = tf.keras.layers.Dense(128, activation='relu', name="mlp_01")(x)
    # x = tf.keras.layers.Dropout(0.2)(x)
    predictions_global = tf.keras.layers.Dense(
        len(data.LABELS), activation="sigmoid")(x)
    model_global = tf.keras.models.Model(
        inputs=g_model.input, outputs=predictions_global)
    return model_global

# callbacks setup
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=7,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor=.1,
    patience=5,
    min_lr=0.000001,
    min_delta=0.001
)
# TRIAL RUN
def train_global():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")

    model_global = create_model_global()
    model_global.compile(optimizer=tf.keras.optimizers.Adam(),
                         loss=tf.keras.losses.BinaryFocalCrossentropy(
                             apply_class_balancing=True),
                         metrics=[tf.keras.metrics.AUC(multi_label=True)])
    # #* warm up?
    # for layer in model_global.layers:
    #     layer.trainable = False
    #     if "mlp" in layer.name:
    #         layer.trainable = True

    #* Shallow Fine Tunning
    # for layer in model_global.layers:
    #     layer.trainable=False
    #     if "conv5_block16" in layer.name:
    #         layer.trainable = True
    #     if "conv5_block15" in layer.name:
    #         layer.trainable = True
    #     if "conv5_block14" in layer.name:
    #         layer.trainable = True
    #     if "bn" in layer.name:
    #         layer.trainable=True

    # H_G = model_global.fit(train_generator_global,
    #                        validation_data=val_generator_global,
    #                        epochs=5,
    #                        callbacks=[
    #                            # checkpoint,
    #                            lr_scheduler,
    #                            early_stopping
    #                        ],
    #                        )

    #* deep fine tuning
    for layer in model_global.layers:
        layer.trainable = True

    H_G = model_global.fit(train_generator_global,
                           validation_data=val_generator_global,
                           epochs=20,
                           callbacks=[
                               # checkpoint,
                               lr_scheduler,
                               early_stopping
                           ],
                           )

    utils.save_history(H_G.history, CHECKPOINT_PATH, branch="global")
    print("Predictions: ")
    predictions_global = model_global.predict(test_generator_global, verbose=1)

    results_global = utils.evaluate_classification_model(
        test_generator_global.labels, predictions_global, data.LABELS)

    prefix = model_name[:8]
    auc_macro_formatted = "{:.3f}".format(results_global['auc_macro'])

    filename = f"{prefix}_{auc_macro_formatted}.hdf5"

    filepath = os.path.join(CHECKPOINT_PATH, "checkpoint", filename)
    if os.path.exists(filepath):
        os.remove(filepath)

    utils.store_test_metrics(results_global, path=CHECKPOINT_PATH,
                             filename=f"metrics_global", name=model_name, json=True)
    if results_global['auc_macro'] > 0.75:
        model_global.save(filepath=filepath)
    # else:
    #     shutil.rmtree(CHECKPOINT_PAT

    # salvar modelo// restore best weights...


if __name__ == "__main__":
    train_global()
