import os
import numpy as np

def cache_array(filename, array_generator, save_cache=True, verbose=True):
    file_path = os.path.join("./Cache", filename)

    if os.path.exists(file_path):
        if verbose:
            print(f"Chargement des données depuis {filename}")
        return np.load(file_path)
    else:
        if verbose:
            print(f"Fichier {filename} introuvable, génération des données...")
        array = array_generator()
        if (save_cache):
            if verbose:
                print(f"Sauvegarde des données dans {filename}")
            np.save(file_path, array)
        return array
    
def encoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-encoded-{n}.npy", lambda: encoder.predict(
        (decoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_encoding or save_last, verbose)

def decoded(x, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    return cache_array(f"{name}-{encoder.name}-{decoder.name}-decoded-{n}.npy", lambda: decoder.predict(
        (encoded(x, name, encoder, decoder, n - 1, batch_size, False, save_encoding, save_decoding, verbose) if n > 1 else x),
        batch_size = batch_size
    ), save_decoding or save_last, verbose)

def encoded_means(x, y, name, encoder, decoder, n, batch_size = 1, save_last = True, save_encoding = False, save_decoding = False, verbose = True):
    def calculate_means():
        x_encoded = encoded(x, name, encoder, decoder, n, batch_size, False, save_encoding, save_decoding, verbose)
        encoded_means = [None] * 10
        for i in range(10):
            encoded_means[i] = np.mean(x_encoded[y == i], axis=0)
            encoded_means[i] = np.expand_dims(encoded_means[i], axis=0)
        return np.array(encoded_means)

    return cache_array(f"{name}-{encoder.name}-{decoder.name}-{n}.npy", calculate_means, save_last, verbose)