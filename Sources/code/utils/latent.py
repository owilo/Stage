from code.utils import cache
import numpy as np
import scipy
import tensorflow as tf
import ot
from sklearn.preprocessing import StandardScaler

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

def transform_mt(z_src, y_src, y_dst, z_train, y_train):
    """
    Transformation d'une distribution normale multidimensionnelle en une autre
    """
    z_src = np.asarray(z_src)
    n = z_src.shape[0]
    
    if np.isscalar(y_src):
        y_src = np.full(len(z_src), y_src)
    if np.isscalar(y_dst):
        y_dst = np.full(len(z_src), y_dst)
        
    if len(z_src) != len(y_src) or len(y_src) != len(y_dst):
        raise ValueError(f"z_src ({len(z_src)}), y_src ({len(y_src)}) et y_dst ({len(y_dst)}) doivent être de la même taille")

    classes_needed = set(np.concatenate([np.unique(y_src), np.unique(y_dst)]))
    stats = {}
    for cls in classes_needed:
        indices = np.where(y_train == cls)[0]
        z_cls = z_train[indices]
        mu = np.mean(z_cls, axis=0)
        sigma = np.cov(z_cls, rowvar=False)
        L = np.linalg.cholesky(sigma)
        stats[cls] = (mu, L)

    z_dst = np.zeros_like(z_src)
    
    for i in range(n):
        src_cls = y_src[i]
        dst_cls = y_dst[i]
        mu_src, L_src = stats[src_cls]
        mu_dst, L_dst = stats[dst_cls]
        
        # https://en.wikipedia.org/wiki/Whitening_transformation
        z_whitened = np.linalg.solve(L_src, (z_src[i] - mu_src))
        z_dst[i] = mu_dst + L_dst.dot(z_whitened)
        
    if z_dst.shape[0] == 1:
        return z_dst[0]
    return z_dst

def compute_mappings_ot(z_classes, y_classes, subsample_ratio=0.3, save_cache=True, verbose=None):
    # c'est TRÈS long, mieux vaut laisser le cache !
    # sinon augmenter reg ou réduire le nombre d'itérations/subsample_ratio
    # si reg augmente beaucoup, ot.sinkhorn devrait suffire

    if verbose is None:
        verbose = save_cache

    key = "ot_mappings" + cache.data_hash(z_classes) + cache.data_hash(y_classes) + str(subsample_ratio)

    def _compute_mappings_ot():
        classes = np.unique(y_classes)
        class_pairs = [(src, dst) for src in classes for dst in classes if src != dst]
        
        mappings = {}
        
        for src_cls, dst_cls in class_pairs:
            print(f"Paire: ({src_cls}, {dst_cls})")
            
            src_indices = np.where(y_classes == src_cls)[0]
            dst_indices = np.where(y_classes == dst_cls)[0]
            
            src_sample_size = int(len(src_indices) * subsample_ratio)
            dst_sample_size = int(len(dst_indices) * subsample_ratio)

            src_sample = np.random.choice(src_indices, src_sample_size, replace=False)
            dst_sample = np.random.choice(dst_indices, dst_sample_size, replace=False)

            z_src_class = z_classes[src_sample]
            z_dst_class = z_classes[dst_sample]

            scaler = StandardScaler()
            z_src_scaled = scaler.fit_transform(z_src_class)
            z_dst_scaled = scaler.transform(z_dst_class)

            M = ot.dist(z_src_scaled, z_dst_scaled, metric='euclidean')
            a = np.ones((len(z_src_scaled),)) / len(z_src_scaled)
            b = np.ones((len(z_dst_scaled),)) / len(z_dst_scaled)

            # transport_plan = ot.sinkhorn(a, b, M, reg=0.275, numItermax=10000)
            #transport_plan = ot.emd(a, b, M, numItermax=5000000)
            transport_plan = ot.bregman.sinkhorn_log(a, b, M, reg=0.1, numItermax=20000)

            mappings[(src_cls, dst_cls)] = {
                'scaler': scaler,
                'transport_plan': transport_plan,
                'z_src_scaled': z_src_scaled,
                'z_dst_scaled': z_dst_scaled
            }
        
        return mappings
    
    return cache.load_from_cache(key, _compute_mappings_ot, save_cache, verbose)

def transform_ot(z, y_src, y_dst, mappings):
    """
    Transformation d'une distribution quelconque en une autre
    """
    z_dst = np.copy(z)
    for (src_cls, dst_cls), mapping in mappings.items():
        if src_cls == dst_cls:
            continue

        mask = (y_src == src_cls) & (y_dst == dst_cls)
        if np.any(mask):
            scaler = mapping['scaler']
            transport_plan = mapping['transport_plan']
            z_src_scaled = mapping['z_src_scaled']
            z_dst_scaled = mapping['z_dst_scaled']

            z_batch = scaler.transform(z[mask])

            transformed = np.zeros_like(z_batch)
            n_src = len(z_src_scaled)
            for i, sample in enumerate(z_batch):
                distances = np.linalg.norm(z_src_scaled - sample, axis=1)
                closest_idx = np.argmin(distances)
                transport_weights = transport_plan[closest_idx]
                transformed_sample = np.dot(transport_weights * n_src, z_dst_scaled)
                transformed[i] = transformed_sample

            z_dst[mask] = scaler.inverse_transform(transformed)
    return z_dst