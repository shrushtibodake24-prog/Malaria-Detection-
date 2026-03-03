import os
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Seaborn is required by the task. Auto-install if missing to avoid runtime errors.
try:
    import seaborn as sns
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

from PIL import Image


def _set_plot_style() -> None:
    # Robust style selection across seaborn/matplotlib versions
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except OSError:
        try:
            plt.style.use("seaborn-darkgrid")
        except OSError:
            plt.style.use("default")
    sns.set_context("notebook")


def _iter_image_paths(folder: Path, max_files: int | None = None) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths: list[Path] = []
    for p in folder.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            paths.append(p)
            if max_files is not None and len(paths) >= max_files:
                break
    return paths


def plot_sample_images(dataset_dir: str = "dataset", n_per_class: int = 6) -> None:
    parasitized_dir = Path(dataset_dir) / "Parasitized"
    uninfected_dir = Path(dataset_dir) / "Uninfected"

    if not parasitized_dir.exists() or not uninfected_dir.exists():
        raise FileNotFoundError(
            f"Expected dataset folders at '{parasitized_dir}' and '{uninfected_dir}'."
        )

    # Load a small random sample of raw images
    p_paths = _iter_image_paths(parasitized_dir)
    u_paths = _iter_image_paths(uninfected_dir)

    if len(p_paths) == 0 or len(u_paths) == 0:
        raise RuntimeError("No images found in dataset folders.")

    n_per_class = min(n_per_class, len(p_paths), len(u_paths))
    p_pick = random.sample(p_paths, n_per_class)
    u_pick = random.sample(u_paths, n_per_class)

    fig, axes = plt.subplots(2, n_per_class, figsize=(2.4 * n_per_class, 5.2))
    fig.suptitle("Sample Malaria Cell Images (Raw Dataset)", fontsize=16, fontweight="bold", y=1.02)

    for i, path in enumerate(p_pick):
        img = Image.open(path).convert("RGB")
        axes[0, i].imshow(img)
        axes[0, i].set_title("Parasitized", fontsize=11, fontweight="bold")
        axes[0, i].axis("off")

    for i, path in enumerate(u_pick):
        img = Image.open(path).convert("RGB")
        axes[1, i].imshow(img)
        axes[1, i].set_title("Uninfected", fontsize=11, fontweight="bold")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig("sample_images_grid.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_class_distribution(processed_data_dir: str = "processed_data") -> None:
    processed = Path(processed_data_dir)
    y_train_path = processed / "y_train.npy"
    y_val_path = processed / "y_val.npy"

    if not y_train_path.exists() or not y_val_path.exists():
        raise FileNotFoundError("Missing y_train.npy/y_val.npy. Run preprocess_data.py first.")

    y_train = np.load(y_train_path)
    y_val = np.load(y_val_path)

    train_uninfected = int(np.sum(y_train == 0))
    train_parasitized = int(np.sum(y_train == 1))
    val_uninfected = int(np.sum(y_val == 0))
    val_parasitized = int(np.sum(y_val == 1))

    splits = ["Train", "Train", "Validation", "Validation"]
    classes = ["Uninfected", "Parasitized", "Uninfected", "Parasitized"]
    counts = [train_uninfected, train_parasitized, val_uninfected, val_parasitized]

    plt.figure(figsize=(10, 5))
    ax = sns.barplot(x=splits, y=counts, hue=classes, palette="Set2")
    ax.set_title("Class Distribution (Training vs Validation)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Dataset Split", fontsize=12)
    ax.set_ylabel("Number of Images", fontsize=12)
    ax.legend(title="Class", loc="upper right")

    # Add count labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt="%d", padding=3, fontsize=10)

    plt.tight_layout()
    plt.savefig("class_distribution_train_val.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_augmented_images(processed_data_dir: str = "processed_data", n_aug: int = 8) -> None:
    # Uses the SAME preprocessing scale as earlier: images are already normalized to [0,1] in X_train.npy
    processed = Path(processed_data_dir)
    x_train_path = processed / "X_train.npy"
    y_train_path = processed / "y_train.npy"

    if not x_train_path.exists() or not y_train_path.exists():
        raise FileNotFoundError("Missing X_train.npy/y_train.npy. Run preprocess_data.py first.")

    # Avoid loading the full 2.5GB array into RAM
    X_train = np.load(x_train_path, mmap_mode="r")
    y_train = np.load(y_train_path)

    # TensorFlow is used only for augmentation previews
    try:
        import tensorflow as tf  # noqa: F401
        from tensorflow.keras.preprocessing.image import ImageDataGenerator
    except ImportError as e:
        raise ImportError(
            "TensorFlow is required to display augmented images. "
            "Activate your tf_env and ensure tensorflow is installed."
        ) from e

    # Pick a reproducible sample index (prefer one from each class if possible)
    idx_par = int(np.where(y_train == 1)[0][0]) if np.any(y_train == 1) else 0
    idx_un = int(np.where(y_train == 0)[0][0]) if np.any(y_train == 0) else 0

    samples = [
        ("Parasitized sample", np.array(X_train[idx_par], dtype=np.float32)),
        ("Uninfected sample", np.array(X_train[idx_un], dtype=np.float32)),
    ]

    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
    )

    fig, axes = plt.subplots(len(samples), n_aug, figsize=(2.2 * n_aug, 4.8))
    fig.suptitle("Augmented Images Preview (Light Augmentation)", fontsize=16, fontweight="bold", y=1.02)

    for row, (title, img) in enumerate(samples):
        x = np.expand_dims(img, axis=0)  # (1,224,224,3)
        gen = datagen.flow(x, batch_size=1, shuffle=False, seed=42)
        for col in range(n_aug):
            aug = next(gen)[0]
            aug = np.clip(aug, 0.0, 1.0)
            axes[row, col].imshow(aug)
            axes[row, col].axis("off")
            if col == 0:
                axes[row, col].set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig("augmented_images_preview.png", dpi=150, bbox_inches="tight")
    plt.show()


def main() -> None:
    random.seed(42)
    np.random.seed(42)
    _set_plot_style()

    # STEP 3 Visualizations
    plot_sample_images(dataset_dir="dataset", n_per_class=6)
    plot_class_distribution(processed_data_dir="processed_data")
    plot_augmented_images(processed_data_dir="processed_data", n_aug=8)


if __name__ == "__main__":
    main()

