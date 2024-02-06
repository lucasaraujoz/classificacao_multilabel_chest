import pickle

def save_history(history, model_path, branch="geral"):
    with open(f"{model_path}/history_{branch}", "wb") as f:
            pickle.dump(history, f)

def store_test_metrics(var, path, filename="test_metrics"):
    with open(f"{path}/{filename}", "wb") as f:
     pickle.dump(var, f)