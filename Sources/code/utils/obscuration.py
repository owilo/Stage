import numpy as np

import numpy as np

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
