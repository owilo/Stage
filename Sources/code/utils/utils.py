import numpy as np
import tensorflow as tf

def group_by_class(x, y):
    return {cls: x[y == cls] for cls in np.unique(y)}

def split_dataset(x, y, p, seed=0):
    if not isinstance(x, np.ndarray):
        x = np.array(x)
    if not isinstance(y, np.ndarray):
        y = np.array(y)

    if y.ndim > 1 and y.shape[1] > 1:
        if np.all(np.sum(y, axis=1) == 1):
            yl = np.argmax(y, axis=1)
        else:
            raise ValueError("y n'est pas un tableau de labels.")
    else:
        yl = y

    if seed is not None:
        rng = np.random.default_rng(seed)
    
    mask = np.zeros(len(x), dtype=bool)
    
    for cls in np.unique(yl):
        cls_indices = np.flatnonzero(yl == cls)
        if seed is not None:
            rng.shuffle(cls_indices)
        split_idx = int(len(cls_indices) * p)
        mask[cls_indices[:split_idx]] = True

    x1, y1 = x[mask], y[mask]
    x2, y2 = x[~mask], y[~mask]
    
    return x1, y1, x2, y2

def split_src_to_dst(x, y):
    d = group_by_class(x, y)

    dst_classes = np.array(list(d.keys()), dtype=int)

    x_src_list = []
    y_src_list = []
    y_dst_list = []

    for key, items in d.items():
        n = len(items)
        
        x_src_list.append(np.array(items))
        y_src_list.append(np.full(n, key))
        y_dst_list.append(np.resize(dst_classes, n))

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
    return tf.image.resize(x, shape[:2]).numpy()