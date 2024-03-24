import config
import config_gpu
import os
import uuid
import utils
import tensorflow as tf
import data
import pandas as pd
from models.classification_model_v5 import *
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
df_train = pd.read_csv("df_train.csv")
df_val = pd.read_csv("df_val.csv")
df_test = pd.read_csv("df_test.csv")

datagen_train = tf.keras.preprocessing.image.ImageDataGenerator(
    horizontal_flip = True,
    rescale=1/255.0,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rotation_range=10,
    shear_range=0.1,
    zoom_range=0.1,
)

datagen_val_test = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255.
)

# GENERATORS GLOBAL
train_generator_global = data.get_generator(
    df=df_train, x_col="path", shuffle=True, batch_size=16, imageDataGenerator=datagen_train)
val_generator_global = data.get_generator(
    df=df_val, x_col="path", shuffle=False, imageDataGenerator=datagen_val_test)
test_generator_global = data.get_generator(
    df=df_test, x_col="path", shuffle=False, imageDataGenerator=datagen_val_test)

early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_auc',
        mode='max',
        patience=15,
        restore_best_weights=True
)
lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode='min',
    factor      =   .1,
    patience    =   4,
    min_lr      =   0.00000000001,
    min_delta   =   0.001
)

def train_global():
    MODEL_PATH = "records"
    VERSION = "v5_withseblock"
    model_name = "model_{}_{}".format(VERSION,str(uuid.uuid4())[:8])
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    model_global = classification_model_v5_3()

    opt = tf.keras.optimizers.Adamax(learning_rate = 1e-3)

    model_global.compile(optimizer=opt,
                         loss=tf.keras.losses.BinaryFocalCrossentropy(
                             apply_class_balancing=True, gamma=4, alpha=0.8),metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])
    
    print(f"Modelo - {model_name}, lr = {1e-3}, batch = {16}, opt={opt}")
    H_G = model_global.fit(train_generator_global,
                           validation_data=val_generator_global,
                           epochs=20,
                           callbacks=[
                               early_stopping,
                               lr_scheduler
                           ],
                           )

    # SALVAR HISTÓRICO
    utils.save_history(H_G.history, CHECKPOINT_PATH, branch="all")
    print("Predictions: ")
    predictions_fusion = model_global.predict(test_generator_global, verbose=1)
    # AVALIAR MODELO
    results_global = utils.evaluate_classification_model(
        test_generator_global.labels, predictions_fusion, data.LABELS)

    auc_macro_formatted = "{:.3f}".format(results_global['auc_macro'])
    utils.store_test_metrics(results_global, path=CHECKPOINT_PATH,
                             filename=f"metrics_fusion", name=model_name, json=True)
    if results_global['auc_macro'] > 0.790:
        model_global.save(f"{CHECKPOINT_PATH}/model_{auc_macro_formatted}")
    # else:
    #     shutil.rmtree(CHECKPOINT_PATH)


if __name__ == "__main__":
    train_global()
