# VARS
import os
from math import ceil
from keras import layers
import tensorflow.keras.backend as K
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
import uuid
import utils
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from glob import glob
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
BATCH_SIZE = 16


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
    # Removendo No Finding
    df = df.drop(df.loc[df['Finding Labels'] == 'No Finding'].index)
    split = GroupShuffleSplit(n_splits=1, test_size=0.2)
    groups = df['Patient ID'].values

    train_idxs, test_idxs = next(split.split(df, groups=groups))

    df_train = df.iloc[train_idxs]
    df_test = df.iloc[test_idxs]
    # split train/val -- 70/20/10
    split = GroupShuffleSplit(n_splits=1, test_size=0.125)
    groups = df_train['Patient ID'].values

    train_idxs, val_idxs = next(split.split(df_train, groups=groups))

    df_train_atualizado = df_train.iloc[train_idxs]
    df_val = df_train.iloc[val_idxs]
    return df_train_atualizado, df_test, df_val


def get_generator(df, x_col, batch_size=BATCH_SIZE, shuffle=False):
    datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization=True)

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory=None,
        x_col=x_col,
        y_col=labels,
        class_mode="raw",
        target_size=(224, 224),
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


# generators global
train_generator_global = get_generator(
    df=df_train, x_col="path", shuffle=False)
val_generator_global = get_generator(df=df_val, x_col="path", shuffle=False)
test_generator_global = get_generator(df=df_test, x_col="path", shuffle=False)

# generators local
train_generator_local = get_generator(
    df=df_train, x_col="path_crop", shuffle=False)
val_generator_local = get_generator(
    df=df_val, x_col="path_crop", shuffle=False)
test_generator_local = get_generator(
    df=df_test, x_col="path_crop", shuffle=False)


def global_branch(input_shape):
    densenet121 = tf.keras.applications.DenseNet169(
        input_shape=input_shape, include_top=False, weights="imagenet")
    densenet121._name = 'densenet121_global_branch'
    return densenet121


def local_branch(input_shape):
    densenet121 = tf.keras.applications.DenseNet169(
        input_shape=input_shape, include_top=False, weights="imagenet")
    densenet121._name = 'densenet121_local_branch'
    return densenet121


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


def model_fusion(input_shape, local_encoder, global_encoder):
    local_input = Input(name='local', shape=input_shape)
    globals_input = Input(name='global', shape=input_shape)

    local_features = local_encoder(local_input)
    global_features = global_encoder(globals_input)
    print(global_features)

    concatenated_volume = Concatenate(
        axis=-1)([local_features, global_features])
    se_block_out = squeeze_excite_block(concatenated_volume)

    # Classificador Global
    x_G = GlobalAveragePooling2D()(global_features)
    classification_global = Dense(
        len(labels), activation="sigmoid", name='sigmoid_global')(x_G)

    # Classificador Local
    x_L = GlobalAveragePooling2D()(local_features)
    classification_local = Dense(
        len(labels), activation="sigmoid", name="sigmoid_local")(x_L)

    # Clasification fusion
    x_F = GlobalAveragePooling2D()(se_block_out)
    classification_fusion = Dense(
        len(labels), activation="sigmoid", name="sigmoid_fusion")(x_F)

    fusion_model = tf.keras.models.Model(
        inputs=[globals_input, local_input], outputs=[
            classification_global, classification_local, classification_fusion]
    )

    return fusion_model


def train():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    print(f"Modelo - {model_name}")
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"{CHECKPOINT_PATH}/weights.ckpt",
                                                    monitor='val_loss',
                                                    save_weights_only=True,
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
        factor=.1,
        patience=3,
        # cooldown    =   5,
        min_lr=0.000001,
        min_delta=0.001
    )

    input_shape = (224, 224, 3)
    g_model = global_branch(input_shape)
    l_model = local_branch(input_shape)
    f_model = model_fusion(input_shape, l_model, g_model)

    train_two = generator_two_img(
        train_generator_global, train_generator_local)
    val_two = generator_two_img(val_generator_global, val_generator_local)
    test_two = generator_two_img(test_generator_global, test_generator_local)

    losses = {
        "sigmoid_local": tf.keras.losses.BinaryFocalCrossentropy(),
        "sigmoid_global": tf.keras.losses.BinaryFocalCrossentropy(),
        "sigmoid_fusion": tf.keras.losses.BinaryFocalCrossentropy(),

    }
    lossWeights = {"sigmoid_local": 0.25,
                   "sigmoid_global": 0.25, "sigmoid_fusion": 1.0}

    print("[INFO] compiling model...")
    f_model.compile(optimizer=tf.keras.optimizers.Adam(), loss=losses, loss_weights=lossWeights,
                    metrics=[tf.keras.metrics.AUC(multi_label=True)])


    # local
    for layer in f_model.layers[3].layers:  # local
        layer.trainable = False
        if "conv5_block16" in layer.name:
            layer.trainable = True
        if "conv5_block15" in layer.name:
            layer.trainable = True
        if "conv5_block14" in layer.name:
            layer.trainable = True
        if "bn" in layer.name:
            layer.trainable = True

    # global
    for x in f_model.layers[2].layers:  # global
        layer.trainable = False
        if "conv5_block16" in layer.name:
            layer.trainable = True
        if "conv5_block15" in layer.name:
            layer.trainable = True
        if "conv5_block14" in layer.name:
            layer.trainable = True
        if "bn" in layer.name:
            layer.trainable = True

    H_F = f_model.fit(train_two,
                      validation_data=val_two,
                      epochs=10,
                      steps_per_epoch=ceil(len(train_generator_global.labels)/BATCH_SIZE),
                      validation_steps= ceil(len(val_generator_global.labels)/BATCH_SIZE),
                      callbacks=[
                          checkpoint,
                          lr_scheduler,
                          early_stopping]
                      )

    utils.save_history(H_F.history, CHECKPOINT_PATH, branch="all")

    print("Predictions: ")
    predictions_global, predictions_local, predictions_fusion = f_model.predict(test_two,
                                                                                verbose=1,
                                                                                steps=ceil(len(test_generator_global.labels)/BATCH_SIZE))

    results_fusion = utils.evaluate_classification_model(
        test_generator_global.labels, predictions_fusion, labels)

    utils.store_test_metrics(results_fusion, path=CHECKPOINT_PATH,
                             filename="metrics_fusion", name=model_name, json=True)


if __name__ == "__main__":
    model = train()
