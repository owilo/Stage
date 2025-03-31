import json
import tensorflow as tf

from Code.Models import *
from Code.Utils import cache

def save_model():
    pass

def cleanup():
    pass

AE_FORMATTER = lambda model: f"{model['category']} - {tuple(model['input_shape'])} → {tuple(model['latent_shape'])} → {tuple(model['output_shape'])} | Dataset : {(100.0 * (model['dataset_range'][1] - model['dataset_range'][0])):.2f}%"

def list_models(criteria={}, formatter=None, models_file="models.json"):
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
    
    if formatter is None:
        def default_formatter(model):
            display_fields = [k for k in criteria if k != "type"]
            parts = [f"category : {model.get('category', 'N/A')}"]
            for key in display_fields:
                if key == "category":
                    continue
                val = tuple(model[key]) if isinstance(model.get(key), list) else model.get(key)
                parts.append(f"{key} : {val}")
            return "-".join(parts)
        formatter = default_formatter
    
    for idx, model in enumerate(matching_models):
        print(f"{idx}. {formatter(model)}")
    
    return matching_models

def select_model(models):
    model = int(input("Sélectionner un modèle : "))
    if model < 0 and model > len(models):
        print("Modèle invalide - abort")
        exit(1)

    return tf.keras.models.load_model(cache.MODEL_FOLDER / models[model]["category"] / models[model]["file"])   
