import xxhash
import dotenv
import os
import pickle
from pathlib import Path

dotenv.load_dotenv()

def get_env_folder(folder_key):
    path = os.getenv(folder_key)
    return Path(path) if path and os.path.isabs(path) else Path.cwd() / path

CACHE_FOLDER = get_env_folder("CACHE_PATH")
MODEL_FOLDER = get_env_folder("MODEL_PATH")
RESULTS_FOLDER = get_env_folder("RESULTS_PATH")

import os
import pickle

def load_from_cache(data_id, supplier, save_cache=True):
    if not data_id.endswith(".pkl"):
        data_id += ".pkl"  # todo for npy too
    
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    cache_file = CACHE_FOLDER / data_id
    if cache_file.exists():
        with open(cache_file, "rb") as file:
            data = pickle.load(file)
        print(f"Données chargées depuis '{cache_file}'.")
        return data
    else:
        print(f"Le fichier cache '{cache_file}' n'existe pas. Génération des données...")
        data = supplier()
        if save_cache:
            with open(cache_file, "wb") as file:
                pickle.dump(data, file)
            print(f"Données sauvegardées vers '{cache_file}'.")
        return data

def model_hash(model):
    weights = model.get_weights()
    weights_bytes = b"".join(w.tobytes() for w in weights)
    hex_hash = xxhash.xxh3_64_hexdigest(weights_bytes)
    return hex_hash

def data_hash(data):
    return xxhash.xxh3_64_hexdigest(pickle.dumps(data))