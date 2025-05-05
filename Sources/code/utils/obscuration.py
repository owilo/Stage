import numpy as np
import tensorflow as tf
from Crypto.Cipher import AES
from Crypto.Util import Counter

def selective_encryption(img, affected_bits, key, decrypt=False):
    mask = np.uint8(affected_bits)
    x_uint = (img * 255).astype(np.uint8)

    ctr = Counter.new(128, initial_value=0)
    cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    plane = x_uint & mask
    flat = plane.flatten().tobytes()

    if not decrypt:
        processed_flat = cipher.encrypt(flat)
    else:
        processed_flat = cipher.decrypt(flat)

    proc_plane = np.frombuffer(processed_flat, dtype=np.uint8).reshape(plane.shape)
    x_proc_uint = (x_uint & (~mask)) | (proc_plane & mask)
    x_proc = x_proc_uint.astype(np.float32) / 255.0
    return x_proc

def bit_flip(img, block_size, seed=0):
    arr = (img * 255).astype(np.uint8)
    H, W, C = arr.shape

    np.random.seed(seed)

    bv = np.zeros((block_size, block_size, C, 8), dtype=np.uint8)
    total_bits = block_size * block_size * C * 8

    bv_flat = bv.reshape(-1)
    bv_flat[total_bits // 2:] = 1
    np.random.shuffle(bv_flat)
    bv = bv_flat.reshape(block_size, block_size, C, 8)

    bit_weights = (1 << np.arange(8, dtype=np.uint8))
    block_mask = np.tensordot(bv, bit_weights, axes=([3], [0]))

    out = arr.copy()

    for top in range(0, H, block_size):
        for left in range(0, W, block_size):
            bh = min(block_size, H - top)
            bw = min(block_size, W - left)

            sub = arr[top : top + bh, left : left + bw]

            mask = block_mask[:bh, :bw, :]

            out[top : top + bh, left : left + bw] = sub ^ mask

    return out.astype(np.float32) / 255.0

"""def fgsm_attack(image, label, model, epsilon):
    with tf.GradientTape() as tape:
        tape.watch(image)
        prediction = model(image)
        loss = tf.keras.losses.sparse_categorical_crossentropy(label, prediction)
    
    gradient = tape.gradient(loss, image)
    signed_grad = tf.sign(gradient)
    
    adv_image = image + epsilon * signed_grad
    adv_image = tf.clip_by_value(adv_image, 0, 1)
    return adv_image"""