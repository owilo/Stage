def default(model, criteria):
    display_fields = [k for k in criteria if k != "type"]
    parts = [f"name : {model.get('name', 'N/A')}", f"category : {model.get('category', 'N/A')}"]
    for key in display_fields:
        if key == "name" or key == "category":
            continue
        val = tuple(model[key]) if isinstance(model.get(key), list) else model.get(key)
        parts.append(f"{key} : {val}")
    return " | ".join(parts)

def autoencoder(model, criteria):
    return f"{model['name']} ({model['category']}) | {tuple(model['input_shape'])} → {tuple(model['latent_shape'])} → {tuple(model['output_shape'])} | Dataset : {(100.0 * (model['dataset_range'][1] - model['dataset_range'][0])):.2f}%"

def classifier(model, criteria):
    return f"{model['name']} ({model['category']}) | {tuple(model['input_shape'])} → {tuple(model['output_shape'])} | Dataset : {(100.0 * (model['dataset_range'][1] - model['dataset_range'][0])):.2f}%"

def automatic(model, criteria):
    return {
        "autoencoder": autoencoder,
        "classifier": classifier,
    }.get(model["type"], default)(model, criteria)