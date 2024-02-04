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
# import tensorflow.python.keras.backend as K
import utils
import uuid
from keras.layers import Dense, Flatten, Dropout, BatchNormalization, Input, Concatenate
from tensorflow.keras.layers import GlobalAveragePooling2D, Reshape, Dense, Permute, multiply
import tensorflow.keras.backend as K
from keras import layers


def split_dataset():
    df = pd.read_csv('/home/lucas/dataset_chest/df_ori_mask_crop.csv')
    # df = df.loc[:,['Image Index','Patient ID', 'Finding Labels']]
    # img_paths={os.path.basename(x): x for x in glob(os.path.join('.', '/home/lucas_araujo/pibic-2024/dataset', 'images*','images','*.png'))} 
    # df['path']=df['Image Index'].map(img_paths.get) #mapping image ids to all image paths
    # labels = df['Finding Labels'].str.get_dummies('|')
    # df = pd.concat([df, labels], axis=1)
    #split train/test 80/20
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

datagen = ImageDataGenerator(
        samplewise_center=True,
        samplewise_std_normalization= True)

train_generator = datagen.flow_from_dataframe(
    dataframe=df_train,
    directory = None,
    x_col='path',
    y_col= labels,
    class_mode= "raw",
    target_size=(224,224),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

val_generator = datagen.flow_from_dataframe(
    dataframe=df_val,
    directory = None,
    x_col='path',
    y_col= labels,
    class_mode= "raw",
    target_size=(224,224),
    batch_size=1,
    shuffle=False,
)


test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    directory = None,
    x_col='path',
    y_col= labels,
    class_mode= "raw",
    target_size=(224,224),
    batch_size=1,
    shuffle=False,
)



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


freq_pos, freq_neg = compute_class_freqs(train_generator.labels)
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


def model_fusion(local_encoder, global_encoder):
  for layer in local_encoder.layers:
      layer._name = layer.name + str("_2")

  local_features = local_encoder.output
  global_features = global_encoder.output
  concatenated_volume = Concatenate(axis=-1)([local_features, global_features])

  se_block_out = squeeze_excite_block(concatenated_volume)
  # Clasification fusion
  x_F = layers.Flatten()(se_block_out)
  classification_fusion =  layers.Dense(15, activation="sigmoid")(x_F)

  fusion_model = tf.keras.models.Model(
      inputs=[local_encoder.input, global_encoder.input], outputs= [classification_fusion]
  )

  return fusion_model

def train():
    # callbacks setup
    MODEL_PATH = "records"
    model_name = f"{uuid.uuid4()}"
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    
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
        patience=10,
        restore_best_weights=True
    )


    img_shape= (224,224,3)
    #classificador global
    global_encoder = global_branch(img_shape)
    x = global_encoder.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_global = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model_global = tf.keras.models.Model(inputs=global_encoder.input, outputs=predictions_global)
    model_global.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])

    # H_G = model_global.fit(train_generator, 
    #     validation_data = val_generator,
    #     epochs = 1,
    #     callbacks=[checkpoint,
    #             early_stopping]
    #     )
   
    #classificador local
    local_encoder = local_branch(img_shape)
    x = local_encoder.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    predictions_local = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
    model_local = tf.keras.models.Model(inputs=local_encoder.input, outputs=predictions_local)

    model_local.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])

    # H_L = model_local.fit(train_generator, 
    #     validation_data = val_generator,
    #     epochs = 1,
    #     callbacks=[checkpoint,
    #             early_stopping]
    #     )
    

    # classificador fusão

    f_model = model_fusion(local_encoder, global_encoder)
    f_model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])
    H_F = f_model.fit(train_generator, 
    validation_data = val_generator,
    epochs = 1,
    callbacks=[checkpoint,
            early_stopping]
    )




    # utils.save_history(H.history, CHECKPOINT_PATH)


    # predictions = model.predict(test_generator, verbose=1)
    # auc_scores = roc_auc_score(test_generator.labels, predictions, average=None)
    # auc_score_macro = roc_auc_score(test_generator.labels, predictions, average='macro')
    # auc_scores_micro = roc_auc_score(test_generator.labels, predictions, average='micro')
    # auc_scores_weighted = roc_auc_score(test_generator.labels, predictions, average='weighted')
    
    # results = {
    #     "groun_truth" : test_generator.labels,
    #     "predictions" : predictions,
    #     "auc_scores" : auc_scores,
    #     "labels" : labels,
    #     "auc_macro" : auc_score_macro,
    #     "auc_micro" : auc_score_micro,
    #     "auc_weighted" : auc_scores_weighted,
    # }
    # utils.store_test_metrics(results, path=CHECKPOINT_PATH) 

if __name__ == "__main__":
     model = train()
