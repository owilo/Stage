import json
import tensorflow as tf

from Code.Models import *
from Code.Utils import cache

import json

def default_formatter(model, criteria):
    display_fields = [k for k in criteria if k != "type"]
    parts = [f"category : {model.get('category', 'N/A')}"]
    for key in display_fields:
        if key == "category":
            continue
        val = tuple(model[key]) if isinstance(model.get(key), list) else model.get(key)
        parts.append(f"{key} : {val}")
    return "-".join(parts)

def ae_formatter(model, criteria):
    return f"{model['category']} | {tuple(model['input_shape'])} → {tuple(model['latent_shape'])} → {tuple(model['output_shape'])} | Dataset : {(100.0 * (model['dataset_range'][1] - model['dataset_range'][0])):.2f}%"

def cleanup_models(models_file="models.json"):
    models_path = cache.MODEL_FOLDER / models_file
    try:
        with open(models_path, "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        print(f"Aucun fichier '{models_file}'.")
        return []

    removed_entries = []
    updated_models = []
    for model in models:
        model_file = cache.MODEL_FOLDER / model.get("category", "") / model.get("file", "")
        if model_file.exists():
            updated_models.append(model)
        else:
            removed_entries.append(model)

    with open(models_path, "w") as f:
        json.dump(updated_models, f, indent=4)

    if removed_entries:
        print(f"{len(removed_entries)} modèle(s) supprimé(s), car le(s) fichier(s) renseigné(s) n'existe(nt) pas.")

    return removed_entries

def save_model(model, model_definition, models_file="models.json"):
    cleanup_models(models_file)
    try:
        with open(cache.MODEL_FOLDER / models_file, "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        models = []

    models.append(model_definition)

    with open(models_file, "w") as f:
        json.dump(models, f, indent=4)

    model_path = cache.MODEL_FOLDER / model_definition["category"]
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / model_definition["file"])

def list_models(criteria={}, formatter=default_formatter, header="Liste des modèles :", models_file="models.json"):
    cleanup_models(models_file)
    with open(cache.MODEL_FOLDER / models_file, "r") as f:
        models = json.load(f)
    
    def matches(model):
        for key, value in criteria.items():
            if key not in model:
                return False
            model_val = model[key]
            if isinstance(model_val, list) and isinstance(value, tuple):
                if tuple(model_val) != value:
                    return False
            else:
                if model_val != value:
                    return False
        return True

    matching_models = [model for model in models if matches(model)]
    
    if not header is None:
        print(header)
    if not formatter is None:
        for idx, model in enumerate(matching_models):
            print(f"{idx}. {formatter(model, criteria)}")
    
    return matching_models

def select_model(models):
    if len(models) == 0:
        print("Aucun modèle trouvé - abort")
        exit(1)

    if len(models) == 1:
        print("(Modèle sélectionné par défaut)")
        model = 0
    else:
        model = int(input("Sélectionner un modèle : "))

    if model < 0 and model > len(models):
        print("Modèle invalide - abort")
        exit(1)

    return tf.keras.models.load_model(cache.MODEL_FOLDER / models[model]["category"] / models[model]["file"]), models[model]
