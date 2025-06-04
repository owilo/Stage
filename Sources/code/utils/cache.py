import xxhash
import dotenv
import os
import sys
import pickle
from pathlib import Path
import numpy as np

dotenv.load_dotenv()

def get_env_folder(folder_key):
    path = os.getenv(folder_key)
    return Path(path) if path and os.path.isabs(path) else Path.cwd() / path

CACHE_FOLDER = get_env_folder("CACHE_PATH")
MODEL_FOLDER = get_env_folder("MODEL_PATH")
RESULTS_FOLDER = get_env_folder("RESULTS_PATH")
IMAGES_FOLDER = get_env_folder("IMAGES_PATH")
DATASETS_FOLDER = get_env_folder("DATASETS_PATH")

def load_from_cache(data_id, supplier, save_cache=True, verbose=False):
    hash_str = xxhash.xxh128_hexdigest(data_id)
    npy_file = CACHE_FOLDER / (hash_str + ".npy")
    pkl_file = CACHE_FOLDER / (hash_str + ".pkl")
    
    os.makedirs(CACHE_FOLDER, exist_ok=True)
    
    if npy_file.exists():
        data = np.load(npy_file, allow_pickle=True)
        if verbose:
            print(f"Données chargées depuis '{npy_file}'.")
        return data
    elif pkl_file.exists():
        with open(pkl_file, "rb") as file:
            data = pickle.load(file)
        if verbose:
            print(f"Données chargées depuis '{pkl_file}'.")
        return data
    else:
        if verbose:
            print(f"Aucun fichier cache trouvé. Génération des données...")
        data = supplier()
        if save_cache:
            if isinstance(data, np.ndarray):
                np.save(npy_file, data)
                if verbose:
                    print(f"Données sauvegardées vers '{npy_file}'.")
            else:
                with open(pkl_file, "wb") as file:
                    pickle.dump(data, file)
                if verbose:
                    print(f"Données sauvegardées vers '{pkl_file}'.")
        return data

def model_hash(model):
    weights = model.get_weights()
    weights_bytes = b"".join(w.tobytes() for w in weights)
    hex_hash = xxhash.xxh3_64_hexdigest(weights_bytes)
    return hex_hash

def data_hash(data):
    return xxhash.xxh3_64_hexdigest(pickle.dumps(data))

def clear_cache():
    if CACHE_FOLDER.exists() and CACHE_FOLDER.is_dir():
        for file in CACHE_FOLDER.iterdir():
            if file.is_file():
                file.unlink()

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "empty":
        clear_cache()
        print("Tous les fichiers du cache ont été supprimés")
    else:
        print("Pour supprimer tous les fichiers du cache : 'cache empty'")