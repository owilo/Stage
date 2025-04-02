import tensorflow as tf
import json
from collections import OrderedDict

from code.models import *
from code.utils import cache, formatters

def cleanup_models(models_file="models.json"):
    models_path = cache.MODEL_FOLDER / models_file
    try:
        with open(models_path, "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        print(f"Aucun fichier '{models_file}'.")
        return []

    removed_entries = []
    latest_models = OrderedDict()
    
    for model in models:
        model_file = cache.MODEL_FOLDER / model.get("category", "") / (model.get("name", "") + ".keras")
        if model_file.exists():
            latest_models[(model.get("name", "") + ".keras")] = model
        else:
            removed_entries.append(model)
    
    updated_models = list(latest_models.values())
    
    with open(models_path, "w") as f:
        json.dump(updated_models, f, indent=4)
    
    if removed_entries:
        print(f"{len(removed_entries)} modèle(s) supprimé(s), car le(s) fichier(s) renseigné(s) n'existe(nt) pas.")
    if len(models) > len(updated_models):
        print(f"{len(models) - len(updated_models)} modèle(s) en double supprimé(s).")
    
    return removed_entries

def save_model(model, model_definition, models_file="models.json"):
    cleanup_models(models_file)
    try:
        with open(cache.MODEL_FOLDER / models_file, "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        models = []

    models.append(model_definition)

    with open(cache.MODEL_FOLDER / models_file, "w") as f:
        json.dump(models, f, indent=4)

    model_path = cache.MODEL_FOLDER / model_definition["category"]
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / (model_definition["name"] + ".keras"))

def list_models(criteria={}, formatter=formatters.automatic, header="Liste des modèles :", models_file="models.json"):
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

def select_model(models, auto_choice=None):
    if not models:
        print("Aucun modèle trouvé - abort")
        exit(1)

    if len(models) == 1:
        print("(Modèle sélectionné par défaut)")
        model = 0
    else:
        if auto_choice is None:
            user_input = input("Sélectionner un modèle (index ou nom) : ")
            try:
                model = int(user_input)
            except ValueError:
                model_name = user_input.strip()
                model = next((i for i, m in enumerate(models) if m["name"] == model_name), -1)
        elif isinstance(auto_choice, int):
            model = auto_choice
        elif isinstance(auto_choice, str):
            model = next((i for i, m in enumerate(models) if m["name"] == auto_choice.strip()), -1)
        else:
            model = -1

    if model < 0 or model >= len(models):
        print("Modèle invalide - abort")
        exit(1)

    return tf.keras.models.load_model(cache.MODEL_FOLDER / models[model]["category"] / (models[model]["name"] + ".keras")), models[model]