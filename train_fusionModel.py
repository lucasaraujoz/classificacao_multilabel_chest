from math import ceil
import uuid
import numpy as np
import utils
import data
import os
import tensorflow as tf


BATCH_SIZE = 16

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

model_global = tf.keras.models.load_model(
    "/home/lucas/dataset_chest/classificacao_multilabel_chest/records/6bb412f6-517f-433d-9aa0-72735d38a053/checkpoint/model.hdf5")
model_local = tf.keras.models.load_model(
    "/home/lucas/dataset_chest/classificacao_multilabel_chest/records/792d7dc5-07ae-45f8-9503-6168b0555833/checkpoint/model.hdf5")

level_1 = model_global.layers[-3].output
level_2 = model_local.layers[-3].output

# trocar nome layers
for layer in model_local.layers:
    layer._name = layer.name + str("_2")

level_1 = model_global.layers[-3].output
level_2 = model_local.layers[-3].output

# x = tf.keras.layers.Concatenate(axis=-1)([level_1, level_2])  # duas entradas
x = tf.keras.layers.Add()([level_1, level_2])
x = tf.keras.layers.Flatten()(x)
# #MLP
x = tf.keras.layers.Dense(256, activation='relu', name="mlp_01")(x)
x = tf.keras.layers.Dropout(0.2)(x)
output_tensor = tf.keras.layers.Dense(
    len(data.LABELS), activation='sigmoid', name="mlp_02")(x)
# #Instanciar e compilar modelo
model = tf.keras.models.Model(
    inputs=[model_global.input, model_local.input], outputs=output_tensor)

# generator dual
df_train, df_test, df_val = data.split_dataset()

# generators global
train_generator_global = data.get_generator(
    df=df_train, x_col="path", shuffle=False)
val_generator_global = data.get_generator(
    df=df_val, x_col="path", shuffle=False)
test_generator_global = data.get_generator(
    df=df_test, x_col="path", shuffle=False)

# generators local
train_generator_local = data.get_generator(
    df=df_train, x_col="path_crop", shuffle=False)
val_generator_local = data.get_generator(
    df=df_val, x_col="path_crop", shuffle=False)
test_generator_local = data.get_generator(
    df=df_test, x_col="path_crop", shuffle=False)


def generator_two_img(gen1, gen2):
    while True:
        X1i = gen1.next()
        X2i = gen2.next()

        yield [X1i[0], X2i[0]], X1i[1]


# train shuffle false; #TODO shuffle no dataframe, já que está false
train_two = generator_two_img(train_generator_global, train_generator_local)
val_two = generator_two_img(val_generator_global, val_generator_local)
test_two = generator_two_img(test_generator_global, test_generator_local)

# callbacks setup
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


MODEL_PATH = "records"
model_name = f"{uuid.uuid4()}"
CHECKPOINT_PATH = f"{MODEL_PATH}/{model_name}"
os.makedirs(CHECKPOINT_PATH, exist_ok=True)
print(f"Modelo - {model_name}")

#*COMPILANDO MODELO
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryFocalCrossentropy(
                  apply_class_balancing=True),
              metrics=[tf.keras.metrics.AUC(multi_label=True)])

#*WARM UP
for layer in model.layers:
    layer.trainable=False
    if "mlp" in layer.name:
        layer.trainable=True

#*FIT
H_F = model.fit(train_two,
                validation_data=val_two,
                epochs=3,
                steps_per_epoch=ceil(
                    len(train_generator_global.labels)/BATCH_SIZE),
                validation_steps=ceil(
                    len(val_generator_global.labels)/BATCH_SIZE),
                callbacks=[
                    # checkpoint,
                    lr_scheduler,
                    early_stopping]
                )

#* SHALLOW FINE TUNING
# for layer in model.layers:
#     layer.trainable=False
#     if "mlp" in layer.name:
#         layer.trainable=True
#     if "conv5_block16" in layer.name:
#         layer.trainable=True


#*DEEP FINE TUNING
for layer in model.layers:
    layer.trainable=True    

H_F = model.fit(train_two,
                validation_data=val_two,
                epochs=10,
                shuffle=True,
                steps_per_epoch=ceil(
                    len(train_generator_global.labels)/BATCH_SIZE),
                validation_steps=ceil(
                    len(val_generator_global.labels)/BATCH_SIZE),
                callbacks=[
                    # checkpoint,
                    lr_scheduler,
                    early_stopping]
                )
utils.save_history(H_F.history, CHECKPOINT_PATH, branch="all")

print("Predictions: ")
predictions_fusion = model.predict(test_two,
                                   verbose=1,
                                   steps=ceil(len(test_generator_global.labels)/BATCH_SIZE))

results_fusion = utils.evaluate_classification_model(
    test_generator_global.labels, predictions_fusion, data.LABELS)

utils.store_test_metrics(results_fusion, path=CHECKPOINT_PATH,
                         filename="metrics_fusion", name=model_name, json=True)


#! salvar modelo acima de threshold 
if(results_fusion['auc_macro']>0.7):
    model.save(filepath=f"{CHECKPOINT_PATH}/checkpoint/model.hdf5")
else:
    pass #TODO remover pasta inteira
