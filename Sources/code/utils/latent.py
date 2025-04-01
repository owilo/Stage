from code.utils import cache
import numpy as np
import scipy
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

"""def ecdf(data):
    x = np.sort(data)
    y = np.linspace(1/len(data), 1, len(data))
    return x, y

def inverse_ecdf(data, kind='linear'):
    x, e = ecdf(data)
    return scipy.interpolate.interp1d(e, x, kind=kind, fill_value="extrapolate")

def quantile_mapping(x_value, source_data, dest_data, kind='linear'):
    source_sorted, _ = ecdf(source_data)
    
    p = np.searchsorted(source_sorted, x_value, side='right') / len(source_data)
    
    dest_inv_ecdf = inverse_ecdf(dest_data, kind=kind)
    
    return dest_inv_ecdf(p)
    
def transform(z_src, y_src, y_dst, z_classes, y_classes):
    z_src = np.asarray(z_src)

    if np.isscalar(y_src):
        y_src = np.full(len(z_src), y_src)
    if np.isscalar(y_dst):
        y_dst = np.full(len(z_src), y_dst)

    if len(z_src) != len(y_src) or len(y_src) != len(y_dst):
        raise ValueError(f"z_src ({len(z_src)}), y_src ({len(y_src)}) et y_dst ({len(y_dst)}) doivent être de la même taille")
    
    z_dst = z_src.copy()
    for i in range(z_src.shape[0]):
        z_src_c = z_classes[y_classes == y_src[i]]
        z_dst_c = z_classes[y_classes == y_dst[i]]
        for j in range(1, z_src[i].shape[0]):
            z_dst[i, j] = quantile_mapping(z_src[i, j], z_src_c[:, j], z_dst_c[:, j])

    return z_dst"""

"""def transform(z, y_src, y_dst, z_classes, y_classes):
    z = np.asarray(z)
    n_samples = z.shape[0]
    
    if np.isscalar(y_src):
        y_src = np.full(n_samples, y_src)
    if np.isscalar(y_dst):
        y_dst = np.full(n_samples, y_dst)

    y_src = np.array(y_src)
    y_dst = np.array(y_dst)
    
    if len(z) != len(y_src) or len(y_src) != len(y_dst):
        raise ValueError(f"z_src ({len(z)}), y_src ({len(y_src)}) et y_dst ({len(y_dst)}) doivent être de la même taille")
    
    unique_classes = np.unique(y_classes)
    class_stats = {}
    for cls in unique_classes:
        cls_mask = (y_classes == cls)
        cls_data = z_classes[cls_mask]
        mean = np.mean(cls_data, axis=0)
        cov = np.cov(cls_data, rowvar=False) # todo class_distributions? not too expensive here either
        class_stats[cls] = {'mean': mean, 'cov': cov}
    
    translated = np.zeros_like(z)
    for i in range(n_samples):
        src_cls = y_src[i]
        dst_cls = y_dst[i]
        
        src_mean = class_stats[src_cls]['mean']
        src_cov = class_stats[src_cls]['cov']
        dst_mean = class_stats[dst_cls]['mean']
        dst_cov = class_stats[dst_cls]['cov']
        
        # https://en.wikipedia.org/wiki/Whitening_transformation
        # whitening : src_cov^(-1/2) * (z - src_mean)
        whitened = scipy.linalg.solve(scipy.linalg.sqrtm(src_cov), (z[i] - src_mean).T).T
        # coloring : dst_cov^(1/2) * whitened + dst_mean
        translated[i] = (scipy.linalg.sqrtm(dst_cov) @ whitened.T).T + dst_mean
       
        # cholesky c'est plus rapide, peut être pas plus efficace
        # L_src = scipy.linalg.cholesky(src_cov, lower=True)
        # whitened = scipy.linalg.solve_triangular(L_src, (z[i] - src_mean).T, lower=True).T
        
        # L_dst = scipy.linalg.cholesky(dst_cov, lower=True)
        # translated[i] = (L_dst @ whitened.T).T + dst_mean
        
    return translated"""

"""import numpy as np
import scipy.linalg
from scipy.stats import ks_2samp, wasserstein_distance

def transform(
    z, 
    y_src, 
    y_dst, 
    z_classes, 
    y_classes, 
    shrinkage=0.01, 
    alpha=0.05, 
    validate=False
):
    z = np.asarray(z)
    y_src = np.broadcast_to(y_src, len(z)).flatten()
    y_dst = np.broadcast_to(y_dst, len(z)).flatten()
    
    classes = np.unique(np.concatenate([y_src, y_dst]))
    class_stats = {}
    for cls in classes:
        mask = y_classes == cls
        if np.sum(mask) < 2:
            raise ValueError(f"Class {cls} has insufficient samples")
            
        cls_data = z_classes[mask]
        mean = np.mean(cls_data, axis=0)
        
        n_features = cls_data.shape[1]
        emp_cov = np.cov(cls_data, rowvar=False)
        cov = (1 - shrinkage) * emp_cov + shrinkage * np.eye(n_features)
        
        try:
            L = scipy.linalg.cholesky(cov, lower=True)
        except scipy.linalg.LinAlgError:
            L = scipy.linalg.cholesky(cov + 1e-6*np.eye(n_features), lower=True)
            
        class_stats[cls] = {'mean': mean, 'L': L}

    src_cls, dst_cls = np.unique(y_src)[0], np.unique(y_dst)[0]
    src_data = z_classes[y_classes == src_cls]
    dst_data = z_classes[y_classes == dst_cls]
    
    pvals = [ks_2samp(src_data[:,i], dst_data[:,i])[1] for i in range(z.shape[1])]
    sig_features = np.where(np.array(pvals) < alpha)[0]

    translated = z.copy()
    for cls_pair in np.unique(np.column_stack((y_src, y_dst)), axis=0):
        src_mask = (y_src == cls_pair[0]) & (y_dst == cls_pair[1])
        if not np.any(src_mask):
            continue
            
        s = class_stats[cls_pair[0]]
        d = class_stats[cls_pair[1]]
        
        whitened = scipy.linalg.solve_triangular(
            s['L'], (z[src_mask] - s['mean']).T, lower=True
        ).T
        
        translated[src_mask] = (d['L'] @ whitened.T).T + d['mean']
        
        if len(sig_features) < z.shape[1]:
            translated[src_mask][:, ~sig_features] = z[src_mask][:, ~sig_features]

    if validate:
        val_data = translated[y_dst == dst_cls]
        metrics = {
            'wasserstein': [wasserstein_distance(val_data[:,i], dst_data[:,i]) 
                           for i in range(z.shape[1])],
            'mean_diff': np.linalg.norm(np.mean(val_data, 0) - class_stats[dst_cls]['mean']),
            'cov_diff': np.linalg.norm(np.cov(val_data.T) - class_stats[dst_cls]['cov'])
        }
        return translated, metrics
    
    return translated"""

import numpy as np
import ot
from sklearn.preprocessing import StandardScaler

def compute_mappings_ot(z_classes, y_classes, subsample_ratio=0.4):
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

        X_src = z_classes[src_sample]
        X_dst = z_classes[dst_sample]

        scaler = StandardScaler()
        X_src_scaled = scaler.fit_transform(X_src)
        X_dst_scaled = scaler.transform(X_dst)

        M = ot.dist(X_src_scaled, X_dst_scaled, metric='euclidean')
        a = np.ones((len(X_src_scaled),)) / len(X_src_scaled)
        b = np.ones((len(X_dst_scaled),)) / len(X_dst_scaled)

        # transport_plan = ot.sinkhorn(a, b, M, reg=0.275, numItermax=10000)
        #transport_plan = ot.emd(a, b, M, numItermax=5000000)
        transport_plan = ot.bregman.sinkhorn_log(a, b, M, reg=0.15, numItermax=10000)

        mappings[(src_cls, dst_cls)] = {
            'scaler': scaler,
            'transport_plan': transport_plan,
            'X_src_scaled': X_src_scaled,
            'X_dst_scaled': X_dst_scaled
        }
    
    return mappings

def transform_ot(z, y_src, y_dst, mappings):
    z_transformed = np.copy(z)
    for (src_cls, dst_cls), mapping in mappings.items():
        if src_cls == dst_cls:
            continue

        mask = (y_src == src_cls) & (y_dst == dst_cls)
        if np.any(mask):
            scaler = mapping['scaler']
            transport_plan = mapping['transport_plan']
            X_src_scaled = mapping['X_src_scaled']
            X_dst_scaled = mapping['X_dst_scaled']

            z_batch = scaler.transform(z[mask])

            transformed = np.zeros_like(z_batch)
            n_src = len(X_src_scaled)
            for i, sample in enumerate(z_batch):
                distances = np.linalg.norm(X_src_scaled - sample, axis=1)
                closest_idx = np.argmin(distances)
                transport_weights = transport_plan[closest_idx]
                transformed_sample = np.dot(transport_weights * n_src, X_dst_scaled)
                transformed[i] = transformed_sample

            z_transformed[mask] = scaler.inverse_transform(transformed)
    return z_transformed