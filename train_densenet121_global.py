#VARS
BATCH_SIZE=8

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
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K
from keras import layers
from keras import optimizers
from math import ceil

labels = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    # "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"] 

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


def split_dataset():
    df = pd.read_csv('df_ori_mask_crop.csv')
    df = df.drop(df.loc[df['Finding Labels'] == 'No Finding'].index) # Removendo No Finding
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    groups = df['Patient ID'].values

    train_idxs, test_idxs = next(split.split(df, groups=groups))

    df_train = df.iloc[train_idxs]
    df_test = df.iloc[test_idxs]
    #split train/val -- 70/20/10
    split = GroupShuffleSplit(n_splits=1, test_size=0.125, random_state=42)
    groups = df_train['Patient ID'].values

    train_idxs, val_idxs = next(split.split(df_train, groups=groups))

    df_train_atualizado = df_train.iloc[train_idxs]
    df_val = df_train.iloc[val_idxs]
    return df_train_atualizado, df_test, df_val

def get_generator(df, x_col, batch_size=BATCH_SIZE, shuffle=False):
    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory = None,
        x_col=x_col,
        y_col= labels,
        class_mode= "raw",
        target_size=(224,224),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
    )
    return generator

def generator_two_img(gen1, gen2):
    while True:
        X1i = gen1.next()
        X2i = gen2.next()

        yield [X1i[0], X2i[0]], X1i[1]


df_train, df_test, df_val = split_dataset()



#generators global
train_generator_global = get_generator(df = df_train, x_col="path", shuffle=True)
val_generator_global = get_generator(df = df_val, x_col="path", shuffle=False)
test_generator_global = get_generator(df = df_test, x_col="path", shuffle=False)

#generators local 
train_generator_local = get_generator(df = df_train, x_col="path_crop", shuffle=True)
val_generator_local = get_generator(df = df_val, x_col="path_crop", shuffle=False)
test_generator_local = get_generator(df = df_test, x_col="path_crop", shuffle=False)


def compute_class_freqs(labels):
    """
    Compute positive and negative frequences for each class.

    Args:
        labels (np.array): matrix of labels, size (num_examples, num_classes)
    Returns:
        positive_frequencies (np.array): array of positive frequences for each
                                         class, size (num_classes)
        negative_frequencies (np.array): array of negative frequences for each
                                         class, size (num_classes)
    """

    # total number of patients (rows)
    N = labels.shape[0]

    positive_frequencies = np.sum(labels, axis=0) / N
    negative_frequencies = 1 - positive_frequencies

    return positive_frequencies, negative_frequencies


freq_pos, freq_neg = compute_class_freqs(train_generator_global.labels) #usando o generator global pra calcular a frequencia 
pos_weights = freq_neg
neg_weights = freq_pos
pos_contribution = freq_pos * pos_weights
neg_contribution = freq_neg * neg_weights

def get_weighted_loss(pos_weights, neg_weights, epsilon=1e-7):
    """
    Return weighted loss function given negative weights and positive weights.

    Args:
      pos_weights (np.array): array of positive weights for each class, size (num_classes)
      neg_weights (np.array): array of negative weights for each class, size (num_classes)

    Returns:
      weighted_loss (function): weighted loss function
    """
    def weighted_loss(y_true, y_pred):
        """
        Return weighted loss value.

        Args:
            y_true (Tensor): Tensor of true labels, size is (num_examples, num_classes)
            y_pred (Tensor): Tensor of predicted labels, size is (num_examples, num_classes)
        Returns:
            loss (Float): overall scalar loss summed across all classes
        """
        # initialize loss to zero
        loss = 0.0
        y_true = tf.cast(y_true, tf.float32)

        for i in range(len(pos_weights)):

          loss += K.mean(-(pos_weights[i] *y_true[:,i] * K.log(y_pred[:,i] + epsilon)
          + neg_weights[i]* (1 - y_true[:,i]) * K.log( 1 - y_pred[:,i] + epsilon))) #complete this line
        return loss


    return weighted_loss

def global_branch(input_shape):
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet")
  densenet121._name= 'densenet121_global_branch'
  return densenet121

def local_branch(input_shape):
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False, weights="imagenet")
  densenet121._name= 'densenet121_local_branch'
  return densenet121

def train_global():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{CHECKPOINT_PATH}/weights.ckpt",
        monitor = 'val_loss',
        save_weights_only = True,
        save_best_only=True,
        mode='auto',
        verbose=1
    )

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
        cooldown    =   5,
        min_lr      =   0.000001,
        min_delta   =   0.001
    )

    input_shape = (224,224,3)
    g_model = global_branch(input_shape)
    #____global model____
    x = g_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_global = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model_global = tf.keras.models.Model(inputs=g_model.input, outputs=predictions_global)
    model_global.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=9e-1),
                          loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])

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
            checkpoint,
            lr_scheduler,
            early_stopping
            ],
        )
    
    utils.save_history(H_G.history, CHECKPOINT_PATH, branch="global")
    print("Predictions: ")
    predictions_global = model_global.predict(test_generator_global, verbose=1)
    results_global = utils.evaluate_classification_model(test_generator_global.labels, predictions_global, labels)
    utils.store_test_metrics(results_global, path=CHECKPOINT_PATH, filename=f"metrics_global", name=model_name, json=True)

def train_local():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{CHECKPOINT_PATH}/weights.ckpt",
        monitor = 'val_loss',
        save_weights_only = True,
        save_best_only=True,
        mode='auto',
        verbose=1
    )

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
        cooldown    =   5,
        min_lr      =   0.000001,
        min_delta   =   0.001
    )

    #_____LOCAL_____
    input_shape = (224,224,3)
    l_model = local_branch(input_shape)
    x = l_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_local = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model_local = tf.keras.models.Model(inputs=l_model.input, outputs=predictions_local)
    model_local.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=9e-1),
                          loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])

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
            
    H_l = model_local.fit(train_generator_local, 
        validation_data = val_generator_local,
        epochs = 15,
        callbacks=[
            checkpoint,
            lr_scheduler,
            early_stopping
            ],
        )
    
    utils.save_history(H_l.history, CHECKPOINT_PATH, branch="local")
    print("Predictions: ")
    predictions_local = model_local.predict(test_generator_local, verbose=1)
    results_local = utils.evaluate_classification_model(test_generator_local.labels, predictions_local, labels)
    utils.store_test_metrics(results_local, path=CHECKPOINT_PATH, filename=f"metrics_local", name=model_name, json=True)
    
if __name__ == "__main__":
     train_global()
     train_local()
