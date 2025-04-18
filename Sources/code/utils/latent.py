from code.utils import cache
import numpy as np
import scipy
import tensorflow as tf
import ot
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

def autoencoder_dependant_decode(autoencoder, zp, yp):
    if autoencoder.decoder.requires_labels():
        return autoencoder.decoder.predict((zp, yp))
    else:
        return autoencoder.decoder.predict(zp)

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
            r = autoencoder_dependant_decode(autoencoder, r, y)
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

    def _decode():
        if n_times < 1:
            raise ValueError("n_times doit être supérieur ou égal à 1")
        r = autoencoder_dependant_decode(z, y)
        for _ in range(1, n_times):
            _, _, r = autoencoder.encoder.predict(r) #todo
            r = autoencoder_dependant_decode(autoencoder, r, y)
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

def transform_mg(z_src, y_src, y_dst, z_train, y_train, alpha=None):
    z_src = np.asarray(z_src)
    n, d = z_src.shape
    
    if alpha is None:
        alpha_vec = np.ones((n, d))
    else:
        a = np.asarray(alpha)
        if a.ndim == 0:
            alpha_vec = np.full((n, d), float(a))
        elif a.ndim == 1 and a.shape[0] == d:
            alpha_vec = np.tile(a, (n, 1))
        elif a.shape == (n, d):
            alpha_vec = a
        else:
            raise ValueError("alpha invalide")
    
    # labels broadcasting
    if np.isscalar(y_src):
        y_src = np.full(n, y_src)
    if np.isscalar(y_dst):
        y_dst = np.full(n, y_dst)
    if not (len(y_src)==len(y_dst)==n):
        raise ValueError("z_src, y_src et y_dst doivent avoir la même longueur")
    
    # calculer moyennes et cholesky pour chaque classe
    classes = set(np.unique(np.concatenate([y_src, y_dst])))
    stats = {}
    for cls in classes:
        idx = np.where(y_train == cls)[0]
        zc = z_train[idx]
        mu = np.mean(zc, axis=0)
        cov = np.cov(zc, rowvar=False)
        L = np.linalg.cholesky(cov)
        stats[cls] = (mu, L)
    
    # transformation
    z_dst = np.zeros_like(z_src)
    for i in range(n):
        mu_s, L_s = stats[y_src[i]]
        mu_d, L_d = stats[y_dst[i]]
        # sphérisation
        z_white = np.linalg.solve(L_s, (z_src[i] - mu_s))
        # perturbation réversible
        z_pert = alpha_vec[i] * z_white
        # reprojection
        z_dst[i] = mu_d + L_d.dot(z_pert)
    
    return z_dst if n > 1 else z_dst[0]

def compute_mappings_ot(z_classes, y_classes, subsample_ratio=0.3, save_cache=True, verbose=None):
    """
    Calcule le transport optimal de la distribution d'une classe vers toutes les autres.
    """

    """
    C'est TRÈS long, mieux vaut laisser le cache !
    Sinon augmenter reg ou réduire le nombre d'itérations/subsample_ratio
    Si reg augmente beaucoup, ot.sinkhorn devrait suffire
    """

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
    Transformation d'une distribution quelconque en une autre (AE ou VAE)
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

def classify_mg(z, z_train, y_train):
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(z_train, y_train)

    predicted_classes = qda.predict(z)
    class_probs = qda.predict_proba(z)
    classes = qda.classes_

    n_samples = z.shape[0]
    predicted_indices = np.argmax(class_probs, axis=1)
    certainties = class_probs[np.arange(n_samples), predicted_indices]

    class_probs_dict = {cls: class_probs[:, i] for i, cls in enumerate(classes)}

    return predicted_classes, class_probs_dict, certainties