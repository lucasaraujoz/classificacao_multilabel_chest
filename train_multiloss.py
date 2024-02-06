#VARS
BATCH_SIZE=16

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
from math import ceil


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
    "No Finding",
    "Nodule",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"] 

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
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False)
  densenet121._name= 'densenet121_global_branch'
  return densenet121

def local_branch(input_shape):
  densenet121 = tf.keras.applications.DenseNet121(input_shape= input_shape, include_top= False)
  densenet121._name= 'densenet121_local_branch'
  return densenet121

def squeeze_excite_block(tensor, ratio=16):
    init = tensor
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    if K.image_data_format() == 'channels_first':
        se = Permute((3, 1, 2))(se)

    x = multiply([init, se])
    return x

def model_fusion(input_shape, local_encoder, global_encoder):
  local_input = Input(name='local', shape= input_shape)
  globals_input = Input(name='global', shape= input_shape)

  local_features = local_encoder(local_input)
  global_features = global_encoder(globals_input)
  print(global_features)
  
  concatenated_volume = Concatenate(axis=-1)([local_features, global_features])

  # Classificador Global
  x_G = GlobalAveragePooling2D()(global_features)
  classification_global = Dense(15, activation="sigmoid", name='sigmoid_global')(x_G)

  # Classificador Local
  x_L = GlobalAveragePooling2D()(local_features)
  classification_local = Dense(15, activation="sigmoid", name="sigmoid_local")(x_L)

  # Clasification fusion
  x_F = GlobalAveragePooling2D()(concatenated_volume)
  classification_fusion = Dense(15, activation="sigmoid", name="sigmoid_fusion")(x_F)

  fusion_model = tf.keras.models.Model(
      inputs=[local_input, globals_input], outputs= [classification_global, classification_local, classification_fusion]
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
        monitor = 'val_loss',
        save_weights_only = True,
        save_best_only=True,
        mode='auto',
        verbose=1
    )

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5,
        restore_best_weights=True
    )

    input_shape = (150,150,3)
    g_model = global_branch(input_shape)
    l_model = local_branch(input_shape)
    f_model = model_fusion(input_shape, l_model, g_model)
    
    # classificador fusão
    train_generator_global_2 = get_generator(df = df_train, x_col="path", shuffle=False) # recriando os generator com shuffe=False pra manter consistência do batch
    train_generator_local_2 = get_generator(df = df_train, x_col="path_crop", shuffle=False)
    
    train_two = generator_two_img(train_generator_global_2, train_generator_local_2)
    val_two = generator_two_img(val_generator_global, val_generator_local)
    test_two = generator_two_img(test_generator_global, test_generator_local)
    
    losses = {
        "sigmoid_local": "binary_crossentropy",
        "sigmoid_global": "binary_crossentropy",
        "sigmoid_fusion": "binary_crossentropy",
    
    }
    lossWeights = {"sigmoid_local": 0.25, "sigmoid_global": 0.25, "sigmoid_fusion":1.0}
    print("[INFO] compiling model...")
    f_model.compile(optimizer='adam', loss=losses, loss_weights=lossWeights,
        metrics=[tf.keras.metrics.AUC(multi_label=True)])
    

    # f_model = model_fusion(local_encoder, global_encoder)
    # f_model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])
    
    #congelar backbones
        #local
    for x in f_model.layers[3].layers:
        x.trainable = False
                
    H_G = f_model.fit(train_two, 
        validation_data = val_two,
        epochs = 10,
        steps_per_epoch =  ceil(len(train_generator_global_2.labels)/BATCH_SIZE),
        validation_steps = ceil(len(val_generator_global.labels)/BATCH_SIZE),
        callbacks=[
            checkpoint,
            early_stopping]
    )

    utils.save_history(H_F.history, CHECKPOINT_PATH, branch="global")
    #descongela local
    for x in f_model.layers[3].layers:
        x.trainable = True
    #congela global
    for x in f_model.layers[2].layers:
        x.trainable = False

    H_L = f_model.fit(train_two, 
        validation_data = val_two,
        epochs = 10,
        steps_per_epoch =  ceil(len(train_generator_global_2.labels)/BATCH_SIZE),
        validation_steps = ceil(len(val_generator_global.labels)/BATCH_SIZE),
        callbacks=[
            checkpoint,
            early_stopping]
    )

    utils.save_history(H_F.history, CHECKPOINT_PATH, branch="local")

    #descongela tudo:
    for x in f_model.layers[3].layers:
        x.trainable = True
    #congela global
    for x in f_model.layers[2].layers:
        x.trainable = True

    H_F= f_model.fit(train_two, 
        validation_data = val_two,
        epochs = 10,
        steps_per_epoch =  ceil(len(train_generator_global_2.labels)/BATCH_SIZE),
        validation_steps = ceil(len(val_generator_global.labels)/BATCH_SIZE),
        callbacks=[
            checkpoint,
            early_stopping]
    )

    utils.save_history(H_F.history, CHECKPOINT_PATH, branch="all")

    predictions = f_model.predict(test_two, verbose=1, steps= ceil(len(test_generator_global.labels)/BATCH_SIZE)) ## usando o generator como tamanho de steps
    auc_scores = roc_auc_score(test_generator_global.labels, predictions, average=None)
    auc_score_macro = roc_auc_score(test_generator_global.labels, predictions, average='macro')
    auc_score_micro = roc_auc_score(test_generator_global.labels, predictions, average='micro')
    auc_score_weighted = roc_auc_score(test_generator_global.labels, predictions, average='weighted')
    
    results = {
        "groun_truth" : test_generator_global.labels,
        "predictions" : predictions,
        "auc_scores" : auc_scores,
        "labels" : labels,
        "auc_macro" : auc_score_macro,
        "auc_micro" : auc_score_micro,
        "auc_weighted" : auc_score_weighted,
    }
    utils.store_test_metrics(results, path=CHECKPOINT_PATH, filename=f"test_metrics_auc_{auc_score_macro}")

if __name__ == "__main__":
     model = train()
