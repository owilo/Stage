import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices=false"
import random
import cv2
import numpy as np
from ultralytics import YOLO
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from PIL import Image

from code.utils import cache, utils, latent, models

utils.deterministic()
utils.set_random_seed(42)

(x_mnist, y_mnist), _ = mnist.load_data()
x_mnist, = utils.preprocess_dataset(x_mnist)

autoencoder, _ = models.select_model(models.list_models(
    criteria={"type": "autoencoder", "labels": False, "dataset_range": (0, 1)}
))

z_mnist = latent.encode(
    autoencoder,
    x=x_mnist,
    y=y_mnist,
    n_times=2,
    save_cache=True
)

key = 0
utils.set_random_seed(key)

def digit_transform(x_src: Image.Image, y_src: int) -> Image.Image:
    x_src = np.array(x_src)
    x_src = x_src.reshape((1, 28, 28, 1))
    z_src = latent.encode(
        autoencoder,
        x=x_src,
        y=y_src,
        n_times=2,
        save_cache=False
    )

    u = np.random.randint(0, 10)
    y_dst = (u + y_src) % 10
    z_dst = latent.transform_mg(z_src, y_src, y_dst, z_mnist, y_mnist)
    z_dst = np.expand_dims(z_dst, axis=0)

    x_dst = autoencoder.decoder.predict(z_dst)

    img_array = x_dst[0, :, :, 0]
    img_array = (img_array * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(img_array, mode='L')

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=35,
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
    cmap = plt.cm.get_cmap("Paired", NUM_CLASSES)

    for i, res in enumerate(results):
        # ------------------------------------------------------------------
        # 1) Load the image in BGR, convert to RGB
        img_bgr = cv2.imread(res.path)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Pull out boxes, classes, confidences
        boxes = res.boxes.xyxy.cpu().numpy()        # shape (N, 4)
        classes = res.boxes.cls.cpu().numpy().astype(int)
        confs = res.boxes.conf.cpu().numpy()

        # ------------------------------------------------------------------
        # 2) DRAW & SAVE “BEFORE” IMAGE (just bounding boxes + labels, no digit_transform)
        # Make a copy so we don’t overwrite img_rgb itself
        before_img = img_rgb.copy()

        fig1, ax1 = plt.subplots(figsize=(5, 5))
        ax1.imshow(before_img)
        ax1.axis("off")

        for (x1f, y1f, x2f, y2f), cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)
            rgba = cmap(cls_id)

            width = x2 - x1
            height = y2 - y1
            rect = patches.Rectangle(
                (x1, y1),
                width,
                height,
                linewidth=1.5,
                edgecolor=rgba,
                facecolor="none",
            )
            ax1.add_patch(rect)

            label_txt = f"{int(cls_id)} {conf:.2f}"
            txt = ax1.text(
                x1,
                y1,
                label_txt,
                fontsize=10,
                color="white",
                backgroundcolor=rgba,
                verticalalignment="bottom",
                bbox=dict(facecolor=rgba, edgecolor="none", pad=1),
            )
            txt.set_path_effects([
                path_effects.Stroke(linewidth=1.0, foreground="black"),
                path_effects.Normal()
            ])

        # Save the “before” image
        save_path_before = cache.RESULTS_FOLDER / "YOLO" / f"mnist-yolo-{i}.png"
        plt.savefig(save_path_before, bbox_inches="tight")
        plt.close(fig1)

        # ------------------------------------------------------------------
        # 3) APPLY digit_transform ON EACH DETECTED BOX, PASTE INTO img_rgb
        #    (this modifies img_rgb in-place)
        for (x1f, y1f, x2f, y2f), cls_id, conf in zip(boxes, classes, confs):
            x1, y1, x2, y2 = int(x1f), int(y1f), int(x2f), int(y2f)

            # A) Crop out the detected digit from img_rgb (H×W×3)
            digit_crop_rgb = img_rgb[y1:y2, x1:x2]         

            # B) Convert the crop to a PIL ‘L’ (grayscale) image
            pil_crop = Image.fromarray(digit_crop_rgb)       # assumes RGB
            pil_gray = pil_crop.convert("L")                 # single-channel

            # C) Resize to (28,28)
            pil_28 = pil_gray.resize((28, 28), resample=Image.BILINEAR)

            # D) Apply your TensorFlow‐based digit_transform
            transformed_28 = digit_transform(pil_28, cls_id)
            # → expects PIL ‘L’ (28×28), returns PIL ‘L’ (28×28)

            # E) Convert back to NumPy uint8 and replicate to 3 channels
            arr_28 = np.array(transformed_28, dtype=np.uint8)      # shape (28,28)
            arr_28_rgb = np.stack([arr_28]*3, axis=-1)              # shape (28,28,3)

            # F) Resize that 3-channel 28×28 up to the original box’s size
            box_h = y2 - y1
            box_w = x2 - x1
            resized_back = cv2.resize(arr_28_rgb, (box_w, box_h), interpolation=cv2.INTER_LINEAR)

            # G) Overwrite that region in img_rgb
            img_rgb[y1:y2, x1:x2] = resized_back

        # ------------------------------------------------------------------
        # 4) DRAW & SAVE “AFTER” IMAGE (boxes + labels drawn on top of the transformed img_rgb)
        fig2, ax2 = plt.subplots(figsize=(5, 5))
        ax2.imshow(img_rgb)
        ax2.axis("off")

        # Save the “after” (transformed) image
        save_path_after = cache.RESULTS_FOLDER / "YOLO" / f"mnist-yolo-transformed-{i}.png"
        plt.savefig(save_path_after, bbox_inches="tight")
        plt.close(fig2)