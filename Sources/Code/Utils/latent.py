from . import cache
import numpy as np

def encode_n(autoencoder, data, n, save_cache=False):
    """
    Applique alternativement l'encodage et le décodage n fois pour obtenir le résultat final encodé.
    """
    key = cache.model_hash(autoencoder) + cache.data_hash(data) + str(n)
    
    def _encode():
        if n < 1:
            raise ValueError("n doit être supérieur ou égal à 1")
        _, _, result = autoencoder.encoder.predict(data) # todo
        for _ in range(1, n):
            result = autoencoder.decoder.predict(result)
            _, _, result = autoencoder.encoder.predict(result)
        return result

    return cache.load_from_cache(key, _encode, save_cache)


def decode_n(autoencoder, data, n, save_cache=False):
    """
    Applique alternativement le décodage et l'encodage n fois pour obtenir le résultat final décodé.
    """
    key = cache.model_hash(autoencoder) + cache.data_hash(data) + str(n)
    
    def _decode():
        if n < 1:
            raise ValueError("n doit être supérieur ou égal à 1")
        result = autoencoder.decoder.predict(data)
        for _ in range(1, n):
            _, _, result = autoencoder.encoder.predict(result) #todo
            result = autoencoder.decoder.predict(result)
        return result

    return cache.load_from_cache(key, _decode, save_cache)


def class_distributions_n(autoencoder, x, y, n, save_cache=False):
    if len(x) != len(y):
        raise ValueError(f"x ({len(x)}) et y ({len(y)}) doivent être de la même taille")

    key = "gms" + cache.model_hash(autoencoder) + cache.data_hash(x) + cache.data_hash(y) + str(n)

    def _class_distributions():
        z = encode_n(autoencoder, x, n, False)

        result = {}
        
        unique_labels = np.unique(y)
        
        for label in unique_labels:
            z_label = z[y == label]
            
            mean = np.mean(z_label, axis=0)
            std = np.std(z_label, axis=0, mean=mean)
            
            result[label] = (mean, std)
        
        return result
    
    return cache.load_from_cache(key, _class_distributions, save_cache)

def translate(z, source_y, destination_y, class_distributions, use_std=True):
    src_mean, src_std = class_distributions[source_y]
    dst_mean, dst_std = class_distributions[destination_y]
    if use_std:
        return dst_mean + (dst_std / src_std) * (z - src_mean)
    else:
        return z + dst_mean - src_mean