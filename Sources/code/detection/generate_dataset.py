# generate_dataset.py

import os
import random
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.datasets import mnist
from PIL import Image
from code.utils import cache, utils, latent, models

GRID_SIZE = 8
NUM_CLASSES = 10
DIGIT_SIZE = 28
OUT_DIR = cache.DATASETS_FOLDER / "multidigits"
NUM_TRAIN = 8000
NUM_VAL = 2000
IMAGE_SIZE = 128
MAX_DIGITS_PER_IMAGE = 5
SEED = 42

def create_mnist_yolo_dataset(
    num_images=10000,
    image_size=IMAGE_SIZE,
    max_digits=5,
    seed=42,
    max_attempts_per_digit=1000
):
    images = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    raw_annotations = []

    rng = np.random.default_rng(seed)

    max_coord = image_size - DIGIT_SIZE

    for i in range(num_images):
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        occupied_boxes = []
        bboxes = []

        n_digits = int(rng.integers(1, max_digits + 1))

        for _ in range(n_digits):
            idx = int(rng.integers(0, x_mnist.shape[0]))
            digit_img = x_mnist[idx]
            label = int(y_mnist[idx])

            digit_resized = cv2.resize(digit_img, (DIGIT_SIZE, DIGIT_SIZE))
            h, w = DIGIT_SIZE, DIGIT_SIZE

            placed = False
            for attempt in range(max_attempts_per_digit):
                x0_new = int(rng.integers(0, max_coord + 1))
                y0_new = int(rng.integers(0, max_coord + 1))
                x1_new = x0_new + w
                y1_new = y0_new + h

                collision = False
                for (x0_old, y0_old, x1_old, y1_old) in occupied_boxes:
                    if not (
                        (x1_new <= x0_old)
                        or (x1_old <= x0_new)
                        or (y1_new <= y0_old)
                        or (y1_old <= y0_new)
                    ):
                        collision = True
                        break

                if not collision:
                    placed = True
                    break

            if not placed:
                break

            canvas[y0_new : y1_new, x0_new : x1_new] = digit_resized

            occupied_boxes.append((x0_new, y0_new, x1_new, y1_new))

            x_center = (x0_new + w / 2) / image_size
            y_center = (y0_new + h / 2) / image_size
            w_norm = w / image_size
            h_norm = h / image_size
            bboxes.append([label, x_center, y_center, w_norm, h_norm])

        images[i] = canvas
        raw_annotations.append(bboxes)

    return images, raw_annotations

def write_yolo_dataset(
    images: np.ndarray,
    raw_annotations: list,
    out_dir: str,
    split: str
):
    img_dir = os.path.join(out_dir, split, "images")
    lbl_dir = os.path.join(out_dir, split, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    N = images.shape[0]
    for idx in range(N):
        img_array = (images[idx] * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        filename_base = f"{split}_{idx:05d}"
        img_path = os.path.join(img_dir, filename_base + ".png")
        cv2.imwrite(img_path, img_bgr)

        label_lines = []
        for (cls_id, x_c, y_c, w_n, h_n) in raw_annotations[idx]:
            label_lines.append(f"{cls_id} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}")
        lbl_path = os.path.join(lbl_dir, filename_base + ".txt")
        with open(lbl_path, "w") as f:
            f.write("\n".join(label_lines))

utils.deterministic()
utils.set_random_seed(42)

(x_mnist, y_mnist), _ = mnist.load_data()
x_mnist, = utils.preprocess_dataset(x_mnist)

def main():
    if OUT_DIR.exists():
        print(f"Removing existing dataset folder: {OUT_DIR}")
        # CAUTION: this will delete anything inside `multidigits`
        import shutil
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # (b) Generate the TRAIN split
    # --------------------------------------------------
    print("Generating TRAIN split...")
    images_train, labels_train = create_mnist_yolo_dataset(
        num_images=NUM_TRAIN,
        image_size=IMAGE_SIZE,
        max_digits=MAX_DIGITS_PER_IMAGE,
        seed=SEED,                # use the same seed to be reproducible
        max_attempts_per_digit=1000,
    )

    write_yolo_dataset(
        images=images_train,
        raw_annotations=labels_train,
        out_dir=str(OUT_DIR),
        split="train",
    )
    print(f"→ Wrote {NUM_TRAIN} training images + labels to {OUT_DIR / 'train'}")

    # (c) Generate the VAL split
    # --------------------------------------------------
    print("Generating VAL split...")
    # You can bump `seed` by 1 (or pick a different seed) so that train/val don't overlap
    images_val, labels_val = create_mnist_yolo_dataset(
        num_images=NUM_VAL,
        image_size=IMAGE_SIZE,
        max_digits=MAX_DIGITS_PER_IMAGE,
        seed=SEED + 1,
        max_attempts_per_digit=1000,
    )

    write_yolo_dataset(
        images=images_val,
        raw_annotations=labels_val,
        out_dir=str(OUT_DIR),
        split="val",
    )
    print(f"→ Wrote {NUM_VAL} validation images + labels to {OUT_DIR / 'val'}")

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()