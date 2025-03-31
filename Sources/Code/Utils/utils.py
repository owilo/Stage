import numpy as np
import tensorflow as tf

def group_by_class(x, y):
    return {cls: x[y == cls] for cls in np.unique(y)}

def split_dataset(x, y, p, seed=0):
    x = np.array(x)
    y = np.array(y)

    yl = y
    # Si y est catégorique, on le transforme en simple tableau de labels
    if y.ndim > 1 and y.shape[1] > 1:
        if np.all(np.sum(y, axis=1) == 1):
            yl = np.argmax(y, axis=1)
        else:
            raise ValueError("y n'est pas un tableau de labels.")
    
    rng = np.random.default_rng(seed)
    
    indices_1 = []
    indices_2 = []
    
    for cls in np.unique(yl):
        cls_indices = np.where(yl == cls)[0]
        rng.shuffle(cls_indices)
        split_idx = int(len(cls_indices) * p)

        indices_1.extend(cls_indices[:split_idx])
        indices_2.extend(cls_indices[split_idx:])
    
    indices_1 = np.array(indices_1)
    indices_2 = np.array(indices_2)
    
    indices_1 = np.sort(indices_1)
    indices_2 = np.sort(indices_2)
    
    x1, y1 = x[indices_1], y[indices_1]
    x2, y2 = x[indices_2], y[indices_2]
    
    return x1, y1, x2, y2

def split_src_to_dst(x, y):
    d = group_by_class(x, y)

    dst_classes = np.array(list(d.keys()), dtype=int)
    m = len(dst_classes)

    x_src_list = []
    y_src_list = []
    y_dst_list = []

    for key, items in d.items():
        n = len(items)
        
        x_src_list.append(np.array(items))
        y_src_list.append(np.full(n, key))
        
        bin_indices = np.floor(np.linspace(0, m, n, endpoint=False)).astype(int)
        y_dst_list.append(dst_classes[bin_indices])

    x_src = np.concatenate(x_src_list)
    y_src = np.concatenate(y_src_list)
    y_dst = np.concatenate(y_dst_list)
    
    return x_src, y_src, y_dst

def classify(x, classifier):
    x = np.asarray(x, dtype=np.float32)

    single_image = (x.ndim == 3)
    if single_image:
        x = np.expand_dims(x, axis=0)

    predictions = classifier.predict(x)

    if predictions.shape[-1] == 1:
        guessed = (predictions >= 0.5).astype(int)
        certainty = 1.0 - np.abs(guessed - predictions)

        guessed = np.squeeze(guessed, axis=-1)
        predictions = np.squeeze(predictions, axis=-1)
        certainty = np.squeeze(certainty, axis=-1)
    else:
        guessed = np.argmax(predictions, axis=1)
        certainty = np.max(predictions, axis=1)

    if single_image:
        return guessed[0], predictions[0], certainty[0]
    return guessed, predictions, certainty

def preprocess_dataset(x_train, x_test):
    x_train = x_train.astype("float32") / 255.
    x_train = x_train.reshape(-1, 28, 28, 1)

    x_test = x_test.astype("float32") / 255.
    x_test = x_test.reshape(-1, 28, 28, 1)
    return x_train, x_test

def shuffle(*arrays, seed=0):
    rng = np.random.default_rng(seed)
    assert all(len(arr) == len(arrays[0]) for arr in arrays), "Tous les tableaux doivent être de même longueur"
    indices = np.arange(len(arrays[0]))
    rng.shuffle(indices)
    return tuple(arr[indices] for arr in arrays)

def resize(x, shape):
    return tf.image.resize(x, shape).numpy()