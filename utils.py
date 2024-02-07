import pickle
import json
import os
from sklearn.metrics import roc_auc_score

def save_history(history, model_path, branch="geral"):
    with open(f"{model_path}/history_{branch}", "wb") as f:
            pickle.dump(history, f)

def store_test_metrics(var, path, filename="test_metrics", name="", json=False):
    with open(f"{path}/{filename}", "wb") as f:
     pickle.dump(var, f) #salva arquivo
    if(json==True):
        auc_macro = var['auc_macro']
        adicionar_dados(path="results.json", uuid=name, result=auc_macro)
        pass

def adicionar_dados(path, uuid, result):
    if not os.path.exists(path):
        with open(path, 'w') as f:
            json.dump({}, f)
    
    with open(path, 'r+') as f:
        dados = json.load(f)
        dados[uuid] = result
        f.seek(0)
        json.dump(dados, f)
        f.truncate()

def evaluate_classification_model(y_true, predictions, labels):
    auc_scores = roc_auc_score(y_true, predictions, average=None)
    auc_score_macro = roc_auc_score(y_true, predictions, average='macro')
    auc_score_micro = roc_auc_score(y_true, predictions, average='micro')
    auc_score_weighted = roc_auc_score(y_true, predictions, average='weighted')
    results = {
    "groun_truth" : y_true,
    "predictions" : predictions,
    "labels" : labels,
    "auc_scores" : auc_scores,
    "auc_macro" : auc_score_macro,
    "auc_micro" : auc_score_micro,
    "auc_weighted" : auc_score_weighted,
}
    return results