import config
import config_gpu
import os
import uuid
import utils
import tensorflow as tf
import data
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


# LOAD DATASET AND SPLIT
df_train, df_test, df_val = data.split_dataset()

# GENERATORS GLOBAL
train_generator_global = data.get_generator(
    df=df_train, x_col="path", shuffle=True, batch_size=config.BATCH_SIZE)
val_generator_global = data.get_generator(
    df=df_val, x_col="path", shuffle=False)
test_generator_global = data.get_generator(
    df=df_test, x_col="path", shuffle=False)

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

def train_global():
    MODEL_PATH = "records"
    VERSION = "v5"
    model_name = "model_{}_{}".format(VERSION,str(uuid.uuid4())[:8])
    CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

    model_global = classification_model_v8()
    optimizer = 'adamax'
    opt=None
    if optimizer== 'sgd':
        opt = tf.keras.optimizers.SGD(learning_rate = config.LR)
    elif optimizer == "adam":
        opt = tf.keras.optimizers.Adam(learning_rate = config.LR)
    elif optimizer == "adamax":
        opt = tf.keras.optimizers.Adamax(learning_rate = config.LR)

    model_global.compile(optimizer=opt,
                         loss=tf.keras.losses.BinaryFocalCrossentropy(
                             apply_class_balancing=True),metrics=[tf.keras.metrics.AUC(multi_label=True, curve='ROC')])
    
    print(f"Modelo - {model_name}, lr = {config.LR}, batch = {config.BATCH_SIZE}, opt={optimizer}")
    H_G = model_global.fit(train_generator_global,
                           validation_data=val_generator_global,
                           epochs=config.EPOCHS,
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
        #salvar csv
        df_train.to_csv(f"{CHECKPOINT_PATH}/df_train.csv")
        df_val.to_csv(f"{CHECKPOINT_PATH}/df_val.csv")
        df_test.to_csv(f"{CHECKPOINT_PATH}/df_test.csv")
    # else:
    #     shutil.rmtree(CHECKPOINT_PATH)


if __name__ == "__main__":
    train_global()
