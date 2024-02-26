
import pandas as pd
import numpy as np
import os                      
from keras.preprocessing.image import ImageDataGenerator
from glob import glob
from sklearn.model_selection import GroupShuffleSplit

LABELS = [
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

#DATALOADER FUNCTION
def get_generator(df, x_col, batch_size=16, shuffle=False, size=(256,256), imageDataGenerator=None):
    datagen = imageDataGenerator
    if imageDataGenerator==None:
        datagen = ImageDataGenerator(
            horizontal_flip = True,
            samplewise_center=True,
            samplewise_std_normalization= True,
            )

    generator = datagen.flow_from_dataframe(
        dataframe=df,
        directory = None,
        x_col=x_col,
        y_col= LABELS,
        class_mode= "raw",
        target_size=size,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return generator