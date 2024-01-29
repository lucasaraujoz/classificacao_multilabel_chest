BATCH_SIZE=8

import pandas as pd
import numpy as np
import os                                                                                                           
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import GroupShuffleSplit
import tensorflow as tf
import tensorflow.python.keras.backend as K

def split_dataset():
    df = pd.read_csv('/home/lucas_araujo/pibic-2024/dataset/Data_Entry_2017.csv')
    df = df.loc[:,['Image Index','Patient ID', 'Finding Labels']]
    img_paths={os.path.basename(x): x for x in glob(os.path.join('.', '/home/lucas_araujo/pibic-2024/dataset', 'images*','images','*.png'))} 
    df['path']=df['Image Index'].map(img_paths.get) #mapping image ids to all image paths
    labels = df['Finding Labels'].str.get_dummies('|')
    df = pd.concat([df, labels], axis=1)
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



base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False)

x = base_model.output

# add a global spatial average pooling layer
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# and a logistic layer
predictions = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)

model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss=get_weighted_loss(pos_weights, neg_weights),  metrics=[tf.keras.metrics.AUC(multi_label=True)])

H = model.fit(train_generator, epochs = 1)


print('fim')

