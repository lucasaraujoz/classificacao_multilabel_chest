import config
import config_gpu
import os
import uuid
import utils
import tensorflow as tf
import data
import pandas as pd
from models.classification_model_v8 import *
# TF CONFIGURATION
print("Número de GPUs disponíveis: ", len(
    tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[config_gpu.INDEX_GPU], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[config_gpu.INDEX_GPU], True)
    except RuntimeError as e:
        print(e)



LABELS = [
    "Infiltration",
    "Pleural_Thickening",
    "Pneumonia",
    "Pneumothorax"]

df_train = pd.read_csv("df_train.csv")
df_val = pd.read_csv("df_val.csv")
df_test = pd.read_csv("df_test.csv")

datagen_val_test = tf.keras.preprocessing.image.ImageDataGenerator(
    samplewise_center=True,
    samplewise_std_normalization= True,
)

# GENERATORS GLOBAL
train_generator_global = data.get_generator(
    df=df_train, x_col="path", shuffle=True, batch_size=config.BATCH_SIZE, names=LABELS)
val_generator_global = data.get_generator(
    df=df_val, x_col="path", shuffle=False, imageDataGenerator=datagen_val_test, names=LABELS)
test_generator_global = data.get_generator(
    df=df_test, x_col="path", shuffle=False, imageDataGenerator=datagen_val_test, names=LABELS)

# CALLBACKS SETUP
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

def densenet_pneumonia(n_classes):
    return tf.keras.Sequential([
        tf.keras.applications.DenseNet121(input_shape=(224,224,3), include_top=False, weights='imagenet'),
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(n_classes, activation="sigmoid")]
    )
def train():
    MODEL_PATH = "records"
    VERSION = "v9"
    model_name = "model_{}_{}".format(VERSION,str(uuid.uuid4())[:8])
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"

    model = densenet_pneumonia(n_classes=4)
    print("instanciou modelo")
    opt = tf.keras.optimizers.Adamax(learning_rate = config.LR)

    model.compile(optimizer=opt,
                         loss=tf.keras.losses.BinaryFocalCrossentropy(
                             apply_class_balancing=True),metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])
    
    print(f"Modelo - {model_name}, lr = {config.LR}, batch = {config.BATCH_SIZE}, opt={opt}")
    H = model.fit(train_generator_global,
                           validation_data=val_generator_global,
                           epochs=15,
                           callbacks=[
                               early_stopping,
                               lr_scheduler
                           ],
                           )

    print("Predictions: ")
    predictions_fusion = model.predict(test_generator_global, verbose=1)
    # AVALIAR MODELO
    results_global = utils.evaluate_classification_model(
        test_generator_global.labels, predictions_fusion, data.LABELS)

    auc_macro_formatted = "{:.3f}".format(results_global['auc_macro'])
    if results_global['auc_macro'] > 0.790:
        os.makedirs(CHECKPOINT_PATH, exist_ok=True)
        model.save(f"{CHECKPOINT_PATH}/model_{auc_macro_formatted}")
        # SALVAR HISTÓRICO
        utils.save_history(H.history, CHECKPOINT_PATH, branch="all")
        utils.store_test_metrics(results_global, path=CHECKPOINT_PATH,
                             filename=f"metrics_fusion", name=model_name, json=True)

if __name__ == "__main__":
    train()
