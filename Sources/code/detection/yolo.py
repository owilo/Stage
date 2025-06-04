import os
import random
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.datasets import mnist
from code.utils import cache
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.patheffects as path_effects

IMAGE_SIZE = 128
GRID_SIZE = 8
NUM_CLASSES = 10

def create_mnist_yolo_dataset(
    num_images=10000,
    image_size=IMAGE_SIZE,
    max_digits=5,
    seed=42
):
    (x_mnist, y_mnist), _ = mnist.load_data()
    x_mnist = x_mnist.astype(np.float32) / 255.0

    images = np.zeros((num_images, image_size, image_size), dtype=np.float32)
    raw_annotations = []

    rng = np.random.default_rng(seed)

    for i in range(num_images):
        canvas = np.zeros((image_size, image_size), dtype=np.float32)
        bboxes = []
        n_digits = rng.integers(1, max_digits + 1)

        for _ in range(n_digits):
            idx = rng.integers(0, x_mnist.shape[0])
            digit_img = x_mnist[idx]  # 28×28
            label = int(y_mnist[idx])

            digit_resized = cv2.resize(digit_img, (28, 28))  # still 28×28

            x_min = int(rng.integers(0, image_size - 28))
            y_min = int(rng.integers(0, image_size - 28))

            sub = canvas[y_min : y_min + 28, x_min : x_min + 28]
            canvas[y_min : y_min + 28, x_min : x_min + 28] = np.maximum(sub, digit_resized)

            x_center = (x_min + 28 / 2) / image_size
            y_center = (y_min + 28 / 2) / image_size
            w_norm = 28 / image_size
            h_norm = 28 / image_size

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

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=128,
        batch=8,
        name="mnist_yolo8n"
    )

    out_dir = cache.DATASETS_FOLDER / "multidigits"
    val_img_dir = os.path.join(out_dir, "val", "images")
    all_val_images = sorted(os.listdir(val_img_dir))
    sample_imgs = [os.path.join(val_img_dir, f) for f in random.sample(all_val_images, 10)]

    results = model.predict(sample_imgs, conf=0.25, save=False)

    NUM_CLASSES = 10
    cmap = cm.get_cmap("Paired", NUM_CLASSES)

    for i, res in enumerate(results):

        img_bgr = cv2.imread(res.path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(img_rgb)
        ax.axis("off")

        boxes = res.boxes.xyxy.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        for (x1, y1, x2, y2), cls_id, conf in zip(boxes, classes, confs):
            rgba = cmap(cls_id)

            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=1.5,
                edgecolor=rgba,
                facecolor="none"
            )
            ax.add_patch(rect)

            label = f"{int(cls_id)} {conf:.2f}"
            txt = ax.text(
                x1,
                y1,
                label,
                fontsize=10,
                color="white",
                backgroundcolor=rgba,
                verticalalignment="bottom",
                bbox=dict(facecolor=rgba, edgecolor="none", pad=1)
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.0, foreground="black"),
                path_effects.Normal()
            ])

        plt.savefig(cache.RESULTS_FOLDER / "YOLO" / f"mnist-yolo-{i}.png", bbox_inches="tight")