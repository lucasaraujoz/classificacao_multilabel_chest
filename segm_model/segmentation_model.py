from glob import glob
import os, pandas as pd
import cv2, numpy as np
import matplotlib.pyplot as plt
from segmentation_models import *
from segmentation_models.metrics import iou_score
import shutil
import pickle

df = pd.read_csv('/home/lucas_araujo/pibic-2024/dataset/Data_Entry_2017.csv')
df = df.loc[:,['Image Index','Patient ID', 'Finding Labels']]
img_paths={os.path.basename(x): x for x in glob(os.path.join('.', '/home/lucas_araujo/pibic-2024/dataset', 'images*','images','*.png'))} 
df['path']=df['Image Index'].map(img_paths.get) #mapping image ids to all image paths


def save_masks(pred, names, output):
    """
        Salva as máscaras geradas

        Parameters
        ----------
        pred: array-like, float32
            Array contendo todas as máscaras preditas
        names: list, 'string'
            Lista de nomes dos arquivos das imagens
        output: string
            Nome do diretório onde as imagens serão salvas
            
    """
    for i in range(pred.shape[0]):
        m = pred[i, :, :, 0]
        cv2.imwrite(output + names[i] + '.png', 255 * ((m - m.min())/(m.max() - m.min())))

batch = []
names = []
i = 1
BACKBONE = 'densenet169'
name = 'Unet'
# model = Unet(BACKBONE, input_shape=(512, 512, 3), classes=1)
# model.load_weights('/home/lucas_araujo/pibic-2024/dataset_segm/best_weightsUnet_densenet169_best_weights.hdf5')
pre_processing = get_preprocessing(BACKBONE)

os.makedirs("./mask_chest", exist_ok=True)

output_dir_base = 'mask_chest/images_'  # Diretório base para salvar as imagens
output_dir_counter = 0

for (path, name) in zip(df['path'], df['Image Index']):
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    batch.append(img)
    names.append(name)
    print(f"{i}/1000")
    if i % 1000 == 0:
        output_dir = f'{output_dir_base}{output_dir_counter}/'
        os.makedirs(output_dir, exist_ok=True)
        batch = np.array(batch)
        model = Unet(BACKBONE, input_shape=(512, 512, 3), classes=1)
        model.load_weights('/home/lucas_araujo/pibic-2024/dataset_segm/best_weightsUnet_densenet169_best_weights.hdf5')
        X = pre_processing(batch)
        pred = model.predict(X, verbose=1)

        # Salvar masks dentro da pasta específica
        save_masks(pred, names, output_dir)

        # Limpar os batches e nomes
        batch = []
        names.clear()

        # Incrementar o contador de diretório
        output_dir_counter += 1

    i += 1

# Processar os dados restantes
if batch:
    output_dir = f'{output_dir_base}{output_dir_counter}/'
    os.makedirs(output_dir, exist_ok=True)

    batch = np.array(batch)
    X = pre_processing(batch)
    pred = model.predict(X, verbose=1)

    # Salvar masks dentro da pasta específica
    save_masks(pred, names, output_dir)