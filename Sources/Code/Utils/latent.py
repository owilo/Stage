from . import cache
import numpy as np
import tensorflow as tf

def decode(autoencoder, z, y):
    if autoencoder.decoder.requires_labels():
        return autoencoder.decoder.predict((z, y))
    else:
        return autoencoder.decoder.predict(z)

def encode_n(autoencoder, x, y, n, save_cache=False):
    """
    Applique alternativement l'encodage et le décodage n fois pour obtenir le résultat final encodé.
    """
    key = cache.model_hash(autoencoder) + cache.data_hash(x) + cache.data_hash(y) + str(n)

    if autoencoder.decoder.requires_labels():
        y = tf.keras.utils.to_categorical(y)

    def _encode():
        if n < 1:
            raise ValueError("n doit être supérieur ou égal à 1")
        _, _, r = autoencoder.encoder.predict(x) # todo
        for _ in range(1, n):
            r = decode(autoencoder, r, y)
            _, _, r = autoencoder.encoder.predict(r)
        return r

    return cache.load_from_cache(key, _encode, save_cache)


def decode_n(autoencoder, z, y, n, save_cache=False):
    """
    Applique alternativement le décodage et l'encodage n fois pour obtenir le résultat final décodé.
    """
    key = cache.model_hash(autoencoder) + cache.data_hash(z) + cache.data_hash(y) + str(n)

    if autoencoder.decoder.requires_labels():
        y = tf.keras.utils.to_categorical(y)
    
    def _decode():
        if n < 1:
            raise ValueError("n doit être supérieur ou égal à 1")
        r = decode(autoencoder, z, y)
        for _ in range(1, n):
            _, _, r = autoencoder.encoder.predict(r) #todo
            r = decode(autoencoder, r, y)
        return r

    return cache.load_from_cache(key, _decode, save_cache)

def class_distributions_n(autoencoder, x, y, n, save_cache=False):
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) et y ({len(y)}) doivent être de la même taille")

    key = "gms" + cache.model_hash(autoencoder) + cache.data_hash(x) + cache.data_hash(y) + str(n)

    def _class_distributions():
        z = encode_n(autoencoder, x, y, n, False)

        result = {}
        
        unique_labels = np.unique(y)
        
        for label in unique_labels:
            z_label = z[y == label]
            
            mean = np.mean(z_label, axis=0)
            std = np.std(z_label, axis=0) # todo mean=mean dépendamment de la version de numpy
            
            result[label] = (mean, std)
        
        return result
    
    return cache.load_from_cache(key, _class_distributions, save_cache)

def translate(z, y_src, y_dst, class_distributions, use_std=True):
    if len(z) != len(y_src) or len(y_src) != len(y_dst):
        raise ValueError(f"x ({len(z)}), y_src ({len(y_src)}) et y_dst ({len(y_dst)}) doivent être de la même taille")
    
    z = np.asarray(z)

    if np.isscalar(y_src) and np.isscalar(y_dst):
        src_mean, src_std = class_distributions[y_src]
        dst_mean, dst_std = class_distributions[y_dst]
        if use_std:
            return dst_mean + (dst_std / src_std) * (z - src_mean)
        else:
            return z + dst_mean - src_mean
    else:
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