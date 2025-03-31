from . import cache
import numpy as np
import tensorflow as tf    

def encode(autoencoder, x, y, n_times=1, save_cache=False, return_dist=False, verbose=None):
    """
    Applique alternativement l'encodage et le décodage n_times fois pour obtenir le résultat final encodé.
    """
    if verbose is None:
        verbose = save_cache

    key = cache.model_hash(autoencoder) + cache.data_hash(x) + cache.data_hash(y) + str(n_times) + str(int(return_dist))

    if autoencoder.decoder.requires_labels():
        y = tf.keras.utils.to_categorical(y)

    def _encode():
        if n_times < 1:
            raise ValueError("n_times doit être supérieur ou égal à 1")
        mean, log_var, r = autoencoder.encoder.predict(x) # todo
        for _ in range(1, n_times):
            r = decode(autoencoder, r, y)
            mean, log_var, r = autoencoder.encoder.predict(r)
        if return_dist:
            return mean, log_var, r
        else:
            return r

    return cache.load_from_cache(key, _encode, save_cache, verbose)


def decode(autoencoder, z, y, n_times=1, save_cache=False, verbose=None):
    """
    Applique alternativement le décodage et l'encodage n_times fois pour obtenir le résultat final décodé.
    """
    if verbose is None:
        verbose = save_cache

    key = cache.model_hash(autoencoder) + cache.data_hash(z) + cache.data_hash(y) + str(n_times)

    if autoencoder.decoder.requires_labels():
        y = tf.keras.utils.to_categorical(y)
    
    def autoencoder_dependant_decode(zp, yp):
        if autoencoder.decoder.requires_labels():
            return autoencoder.decoder.predict((zp, yp))
        else:
            return autoencoder.decoder.predict(zp)

    def _decode():
        if n_times < 1:
            raise ValueError("n_times doit être supérieur ou égal à 1")
        r = autoencoder_dependant_decode(z, y)
        for _ in range(1, n_times):
            _, _, r = autoencoder.encoder.predict(r) #todo
            r = autoencoder_dependant_decode(r, y)
        return r

    return cache.load_from_cache(key, _decode, save_cache, verbose)

def class_distributions(z, y):
    if len(z) != len(y):
        raise ValueError(f"z ({len(z)}) et y ({len(y)}) doivent être de la même taille")

    result = {}
        
    unique_labels = np.unique(y)
    
    for label in unique_labels:
        z_label = z[y == label]
        
        mean = np.mean(z_label, axis=0)
        std = np.std(z_label, axis=0) # todo mean=mean depending on numpy version
        
        result[label] = (mean, std)
    
    return result

def encode_class_distributions(autoencoder, x, y, n_times=1, save_cache=False, verbose=None):
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) et y ({len(y)}) doivent être de la même taille")
    
    if verbose is None:
        verbose = save_cache

    key = "distrib" + cache.model_hash(autoencoder) + cache.data_hash(x) + cache.data_hash(y) + str(n_times)
    
    return cache.load_from_cache(key, lambda: class_distributions(encode(autoencoder, x, y, n_times, False), y), save_cache, verbose)

def translate(z, y_src, y_dst, class_distributions, use_std=True):
    z = np.asarray(z)

    if np.isscalar(y_src):
        y_src = np.full(len(z), y_src)
    if np.isscalar(y_dst):
        y_dst = np.full(len(z), y_dst)

    if len(z) != len(y_src) or len(y_src) != len(y_dst):
        raise ValueError(f"x ({len(z)}), y_src ({len(y_src)}) et y_dst ({len(y_dst)}) doivent être de la même taille")

    y_src = np.array(y_src)
    y_dst = np.array(y_dst)

    src_mean = np.array([class_distributions[c][0] for c in y_src])
    src_std  = np.array([class_distributions[c][1] for c in y_src])
    dst_mean = np.array([class_distributions[c][0] for c in y_dst])
    dst_std  = np.array([class_distributions[c][1] for c in y_dst])
    
    if use_std:
        return dst_mean + (dst_std / src_std) * (z - src_mean)
    else:
        return z + dst_mean - src_mean
        
def style_class_transform(z, y, num_classes=None):
    return (z, tf.keras.utils.to_categorical(y, num_classes=num_classes))