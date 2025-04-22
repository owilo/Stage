import tensorflow as tf
import json
import argparse
import sys
import ast
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


def list_models(criteria=None, formatter=formatters.automatic, header="Liste des modèles :", models_file="models.json"):
    criteria = criteria or {}
    cleanup_models(models_file)
    with open(cache.MODEL_FOLDER / models_file, "r") as f:
        models = json.load(f)
    
    def matches(model):
        for key, value in criteria.items():
            if key not in model:
                return False
            model_val = model[key]
            if isinstance(model_val, list) and isinstance(value, (tuple, list)):
                if tuple(model_val) != tuple(value):
                    return False
            else:
                if model_val != value:
                    return False
        return True

    matching_models = [model for model in models if matches(model)]
    
    if header is not None:
        print(header)
    for idx, model in enumerate(matching_models):
        print(f"{idx}. {formatter(model, criteria)}")
    
    return matching_models


def delete_models(criteria=None, models_file="models.json"):
    criteria = criteria or {}
    models_path = cache.MODEL_FOLDER / models_file
    try:
        with open(models_path, "r") as f:
            models = json.load(f)
    except FileNotFoundError:
        print(f"Aucun fichier '{models_file}' à nettoyer.")
        return []

    to_delete = []
    keep = []
    
    for model in models:
        match = True
        for key, value in criteria.items():
            if key not in model:
                match = False
                break
            model_val = model[key]
            if isinstance(model_val, list) and isinstance(value, (tuple, list)):
                if tuple(model_val) != tuple(value):
                    match = False
                    break
            else:
                if model_val != value:
                    match = False
                    break
        if match:
            to_delete.append(model)
        else:
            keep.append(model)

    if not to_delete:
        print("Aucun modèle ne correspond aux critères fournis.")
        return []

    for model in to_delete:
        model_file = cache.MODEL_FOLDER / model.get("category", "") / (model.get("name", "") + ".keras")
        try:
            model_file.unlink()
            print(f"Fichier supprimé: {model_file}")
        except FileNotFoundError:
            print(f"Fichier non trouvé pour suppression: {model_file}")

    with open(models_path, "w") as f:
        json.dump(keep, f, indent=4)

    print(f"{len(to_delete)} modèle(s) supprimé(s) du JSON et du disque.")
    return to_delete


def parse_criteria(args):
    criteria = {}
    it = iter(args)
    for arg in it:
        if arg.startswith("--"):
            key = arg.lstrip("-")
            try:
                val_str = next(it)
            except StopIteration:
                print(f"Erreur: valeur manquante pour l'argument {arg}")
                sys.exit(1)
            try:
                val = ast.literal_eval(val_str)
            except (ValueError, SyntaxError):
                val = val_str
            criteria[key] = val
    return criteria

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestion des modèles Keras")
    subparsers = parser.add_subparsers(dest='command', required=True)

    subparsers.add_parser('list', help='Liste les modèles selon les critères fournis')
    subparsers.add_parser('delete', help='Supprime les modèles selon les critères fournis')

    args, unknown = parser.parse_known_args()
    criteria = parse_criteria(unknown)

    if args.command == 'list':
        list_models(criteria)
    elif args.command == 'delete':
        delete_models(criteria)
    else:
        parser.print_help()