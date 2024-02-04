import cv2
import numpy as np
import os
from keras.callbacks import ModelCheckpoint
from segmentation_models import *
from segmentation_models.metrics import iou_score
import shutil
import pickle
import tensorflow as tf 
from keras import backend as K


def load_data(train, train_gt, test, test_gt):
    X_train = np.load(train, allow_pickle = True)
    y_train = np.expand_dims(np.load(train_gt, allow_pickle = True), axis = -1)//255
    # print(y_train.shape)
    # y_train = to_categorical(y_train, 1)
    X_test = np.load(test, allow_pickle = True)
    y_test = np.expand_dims(np.load(test_gt, allow_pickle = True), axis = -1)//255
    # print(y_test.shape)
    # y_test = to_categorical((y_test, 1), dtype = 'boolean')
    return X_train, y_train, X_test, y_test

def resize_data(old_set, shape):
    (H, W) = shape
    old_shape = old_set.shape
    new_set = np.empty(shape = (old_shape[0], H, W, old_shape[3]))
    for i in range(old_set.shape[0]):
        new_set[i] = np.expand_dims(cv2.resize(old_set[i, :, :, 0], (W, H)), axis = -1)
    
    return new_set

def resize_all(sets, shape):
    for i in range(len(sets)):
        sets[i] = resize_data(sets[i], shape)

    return sets



def dice_calc(im_1, im_2, empty_score=6.0):
    im1 = im_1 > .3#!= 0#.astype(np.bool)
    im2 = im_2 > .3#!= 0#.astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        # print("empty")
        return empty_score

    # Compute dice_val coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

def sensi_speci_accu(truth, mask):
	# (TP + TN)/(TP + TN + FP + FN)
	# TPR = TP / (TP + FN)
	# SPC = TN / (FP + TN)

    thresh = .3
    C = (((mask > thresh)*2 + (truth > thresh)).reshape(-1, 1) == range(4)).sum(0)

    # sensitivity = C[3]/C[1::2].sum()
    # specificity = C[0]/C[::2].sum()

    sensitivity = C[3]/np.sum(C[1] + C[3])
    specificity = C[0]/np.sum(C[0] + C[2])
    accuracy = (C[0] + C[3])/np.sum(C)

    return sensitivity, specificity, accuracy



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
    

def generate_log(y_true, y_pred, names, output):
    """
        Gera um log com os valores Dice, Sensibilidade, Especificidade e Acurácia para todas as imagens do conjunto de teste

        Parameters
        ----------
        y_true: array-like, float32
            Mascara anotada manualmente (Ground-truth).
        y_pred: array-like, float32
            Mascara predita pelo modelo.
        names: list, string
            Lista com os nomes dos arquivos das imagens.
        output: string
            Caminho para o arquivo de saída.
        
        Returns
        -------
        Dice: float32
            Valor médio do IoU no intervalo [0, 1]
        Sensitivity: float32
            Valor médio da sensibilidade no intervalo [0, 1]
        Especificity: float32
            Valor médio da especificidade no intervalo [0, 1]
        Accuracy: float32
            Valor médio da acurácia no intervalo [0, 1]
    """
    
    with open(output.split('.csv')[0] + '.csv', 'w') as f:
        f.write('NOME,IOU,SENSIBILIDADE,ESPECIFICIDADE,ACURACIA\n')
        dice, sensibilidade, especificidade, acuracia = 0, 0, 0, 0

        for i in range(y_pred.shape[0]):
            iou = dice_calc(y_true[i], y_pred[i])
            s, e, a = sensi_speci_accu(y_true[i], y_pred[i])
            
            dice += iou
            sensibilidade += s
            especificidade += e
            acuracia += a

            f.write('{},{},{},{},{}\n'.format(names[i], iou, s, e, a))
        
        return dice/y_pred.shape[0], sensibilidade/y_pred.shape[0], especificidade/y_pred.shape[0], acuracia/y_pred.shape[0]

def save_statistics(filename, model_name, pre_processing_method, statistics):
	(d, s, e, a) = statistics

	with open(filename, 'a') as f:
		f.write('{},{},{},{},{},{}\n'.format(model_name, pre_processing_method, d, s, e, a))


def create_folder(path):
	if not os.path.exists(path):
		# shutil.rmtree(path)
	    os.makedirs(path)
 
def prepare_folders(root, model_names, preprocessing_names):
    for model_name in model_names:
        current = root + '/' + model_name

        create_folder(current)
        create_folder(current + '/logs')
        create_folder(current + '/masks')
        create_folder(current + '/weights')

        for preprocessing_name in preprocessing_names:
            current_sub = current + '/masks/' + preprocessing_name

            create_folder(current_sub)

'''
'Linknet': Linknet,
'FPN': FPN,
'PSPNet': PSPNet
'''    
a = {
    'Unet': Unet 
}

'''
'vgg16', 'vgg19', 
'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 
'seresnet18', 'seresnet34', 'seresnet50', 'seresnet101', 'seresnet152', 
'seresnext50', 'seresnext101',
'resnext50', 'resnext101',
'senet154',
'densenet121', 'densenet169', 'densenet201',
'inceptionv3', 'inceptionresnetv2',
'mobilenet', 'mobilenetv2',
'efficientnetb0', 'efficientnetb1', 'efficientnetb2', 'efficientnetb3', 'efficientnetb4', 'efficientnetb5', 'efficientnetb6', 'efficientnetb7'
'''

backbones = ['densenet169']



import segmentation_models as sm

sm.set_framework('tf.keras')

sm.framework()

main_folder =  '/home/lucas_araujo/pibic-2024/dataset_segm/bases/' #'/content/drive/MyDrive/chest/segment_gabriel/bases/' 
main_models_folder =  './segm_models/'#'/content/drive/MyDrive/chest/segment_gabriel/segm_models'
X_train, y_train, X_val, y_val = load_data(main_folder + 'train_ALL_BASES_hm.pickle', main_folder + 'train_gt_ALL_BASES_hm.pickle',
                                           main_folder + 'test_ALL_BASES_hm.pickle', main_folder + 'test_gt_ALL_BASES_hm.pickle')

X_test, y_test, X_test, y_test = load_data(main_folder + 'test_Chest_hm.pickle', main_folder + 'test_gt_Chest_hm.pickle',
                                           main_folder + 'test_Chest_hm.pickle', main_folder + 'test_gt_Chest_hm.pickle')

with open('./evaluated.txt', 'r') as f: #/content/drive/MyDrive/chest/segment_gabriel/evaluated.txt'
    evaluated_models = names = [x.rstrip('\n') for x in f.readlines()]

with open('./names.txt', 'r') as f: #/content/drive/MyDrive/chest/segment_gabriel/names.txt
    names = [x.rstrip('\n') for x in f.readlines()]

prepare_folders(main_models_folder, [x[0] for x in a.items()], backbones)

for (name, Model) in a.items():
    for BACKBONE in backbones:
        if not (name + '_' + BACKBONE) in evaluated_models:
            print('TRAINING WITH MODEL ' + name + ' AND PREPROCESSING ' + BACKBONE)

            pre_processing = get_preprocessing(BACKBONE)

            if name == 'PSPNet':
                [X_train, y_train, X_val, y_val] = resize_all([ X_train, y_train, X_val, y_val], (480, 480))
                [X_test, y_test] = resize_all([ X_test, y_test], (480, 480))

            model = Model(BACKBONE, input_shape = (X_train.shape[1], X_train.shape[2], 3), classes = 1)

            xt = pre_processing(X_train)
            xv = pre_processing(X_val)
            xtest = pre_processing(X_test)
            
                
            model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy', iou_score])

            checkpoint = ModelCheckpoint('{}/{}/weights/{}_best_weights.hdf5'.format(main_models_folder, name, BACKBONE), monitor='val_iou_score', verbose=1, save_best_only=True,save_weights_only=True, mode='max')
            
            history = model.fit(
                x = xt,
                y = y_train,
                batch_size = 4,
                epochs = 60,
                callbacks = [checkpoint],
                validation_data = (xv, y_val),
            )

            #TODO FAZER PREDICT E AVALIAÇÃO SEPARADO ... 
            
            # pred = model.predict(xtest)

            # with open('{}/{}/pred_{}.pickle'.format(main_models_folder, name, BACKBONE), 'wb') as f:
            #     pickle.dump(pred, f)

            # statistics = generate_log(y_test, pred, names, '{}/{}/logs/{}.csv'.format(main_models_folder, name, BACKBONE))
            # save_masks(pred, names, '{}/{}/masks/{}/'.format(main_models_folder, name, BACKBONE))
            # save_statistics('{}/resultados'.format(main_models_folder), name, BACKBONE, statistics)

            # with open('evaluated.txt', 'a') as f:
            #     f.write(name + '_' + BACKBONE + '\n')
