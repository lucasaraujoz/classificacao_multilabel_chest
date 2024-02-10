#VARS
#TODO colocar variaveis: loss, optimizador, lr, batch,
#IMPORTS
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


#LOAD DATASET AND SPLIT
df_train, df_test, df_val = data.split_dataset()

#generators global
train_generator_global = data.get_generator(df = df_train, x_col="path", shuffle=True)
val_generator_global = data.get_generator(df = df_val, x_col="path", shuffle=False)
test_generator_global = data.get_generator(df = df_test, x_col="path", shuffle=False)

#generators local 
train_generator_local = data.get_generator(df = df_train, x_col="path_crop", shuffle=True)
val_generator_local = data.get_generator(df = df_val, x_col="path_crop", shuffle=False)
test_generator_local = data.get_generator(df = df_test, x_col="path_crop", shuffle=False)

#BUILD MODEL
def global_branch(input_shape):
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet")
  densenet121._name= 'densenet121_global_branch'
  return densenet121

def local_branch(input_shape):
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet")
  densenet121._name= 'densenet121_local_branch'
  return densenet121

def create_model_global():
    input_shape = (224,224,3)
    g_model = global_branch(input_shape)
    #____global model____
    x = g_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_global = tf.keras.layers.Dense(len(data.LABELS), activation="sigmoid")(x)
    model_global = tf.keras.models.Model(inputs=g_model.input, outputs=predictions_global)
    return model_global

def create_model_local():
    input_shape = (224,224,3)
    l_model = local_branch(input_shape)
    #____local model____
    x = l_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_local = tf.keras.layers.Dense(len(data.LABELS), activation="sigmoid")(x)
    model_local = tf.keras.models.Model(inputs=l_model.input, outputs=predictions_local)
    return model_local

#callbacks setup
early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=7,
        restore_best_weights=True
    )

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor      =   .1,
    patience    =   5,
    min_lr      =   0.000001,
    min_delta   =   0.001
)
#TRIAL RUN
def train_global():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")

    model_global = create_model_global()
    model_global.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),  
                            metrics=[tf.keras.metrics.AUC(multi_label=True)])
    # warm up?


    # Shallow Fine Tunning
    for layer in model_global.layers:
        layer.trainable=False
        if "conv5_block16" in layer.name:
            layer.trainable = True
        if "conv5_block15" in layer.name:
            layer.trainable = True
        if "conv5_block14" in layer.name:
            layer.trainable = True
        if "bn" in layer.name:
            layer.trainable=True
            
    H_G = model_global.fit(train_generator_global, 
        validation_data = val_generator_global,
        epochs = 15,
        callbacks=[
            # checkpoint,
            lr_scheduler,
            early_stopping
            ],
        )
    #salvar modelo// restore best weights...
    model_global.save(filepath=f"{CHECKPOINT_PATH}/checkpoint/model.hdf5")

    utils.save_history(H_G.history, CHECKPOINT_PATH, branch="global")
    print("Predictions: ")
    predictions_global = model_global.predict(test_generator_global, verbose=1)
    results_global = utils.evaluate_classification_model(test_generator_global.labels, predictions_global, data.LABELS)
    utils.store_test_metrics(results_global, path=CHECKPOINT_PATH, filename=f"metrics_global", name=model_name, json=True)

def train_local():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")


    model_local = create_model_local()
    model_local.compile(optimizer=tf.keras.optimizers.Adam(),
                            loss=tf.keras.losses.BinaryFocalCrossentropy(apply_class_balancing=True),  
                            metrics=[tf.keras.metrics.AUC(multi_label=True)])
    # warm up?

    # Shallow Fine Tunning
    for layer in model_local.layers:
        layer.trainable=False
        if "conv5_block16" in layer.name:
            layer.trainable = True
        if "conv5_block15" in layer.name:
            layer.trainable = True
        if "conv5_block14" in layer.name:
            layer.trainable = True
        if "bn" in layer.name:
            layer.trainable=True
            
    H_L = model_local.fit(train_generator_local, 
        validation_data = val_generator_local,
        epochs = 15,
        callbacks=[
            # checkpoint,
            lr_scheduler,
            early_stopping
            ],
        )
    #salvar modelo// restore best weights...
    model_local.save(filepath=f"{CHECKPOINT_PATH}/checkpoint/model.hdf5")

    utils.save_history(H_L.history, CHECKPOINT_PATH, branch="local")
    print("Predictions: ")
    predictions_local = model_local.predict(test_generator_local, verbose=1)
    results_local = utils.evaluate_classification_model(test_generator_local.labels, predictions_local, data.LABELS)
    utils.store_test_metrics(results_local, path=CHECKPOINT_PATH, filename=f"metrics_local", name=model_name, json=True)
    
if __name__ == "__main__":
    #  train_global()
     train_local()
