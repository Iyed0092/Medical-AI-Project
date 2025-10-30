# src/imaging/cnn_tf/train.py
"""
Train script for small CNN.
Generates synthetic grayscale "radiograph-like" images so script is runnable without dataset.

Usage:
    python src/imaging/cnn_tf/train.py --output_dir ./experiments/cnn_run1 --epochs 3 --batch_size 32
"""
import os
import json
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from pathlib import Path

from model import get_cnn_model

def generate_synthetic_data(n_samples=1000, img_size=64, seed=42):
    """
    Generate simple synthetic images:
    - class 0: random noise
    - class 1: random noise + a bright circle in center
    Images normalized to [0,1].
    """
    rng = np.random.RandomState(seed)
    X = rng.normal(loc=0.0, scale=0.5, size=(n_samples, img_size, img_size)).astype(np.float32)
    X = (X - X.min())/(X.max() - X.min() + 1e-9)
    X = X[..., None]  # add channel
    y = rng.binomial(1, 0.5, size=(n_samples,)).astype(np.int32)

    # add a bright circular signal for class 1
    rr, cc = np.ogrid[:img_size, :img_size]
    center = img_size // 2
    radius = img_size // 8
    mask = (rr - center) ** 2 + (cc - center) ** 2 <= radius**2
    for i in range(n_samples):
        if y[i] == 1:
            X[i, mask, 0] = np.clip(X[i, mask, 0] + rng.uniform(0.5, 1.0), 0.0, 1.0)
    return X, y

def main(args):
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic data...")
    X, y = generate_synthetic_data(n_samples=args.n_samples, img_size=args.img_size, seed=args.seed)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=args.seed, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed, stratify=y_tmp)

    print(f"Train/Val/Test sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")

    model = get_cnn_model(input_shape=(args.img_size, args.img_size, 1), num_classes=1)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(outdir / "best_model.h5"), save_best_only=True, monitor='val_auc', mode='max'),
        tf.keras.callbacks.EarlyStopping(monitor='val_auc', patience=3, mode='max', restore_best_weights=True)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2
    )

    model.save(str(outdir / "final_model"))
    metrics = {
        "history": {k: [float(x) for x in v] for k, v in history.history.items()},
        "train_size": int(len(X_train)),
        "val_size": int(len(X_val)),
        "test_size": int(len(X_test))
    }
    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    np.savez_compressed(outdir / "test_data.npz", X_test=X_test, y_test=y_test)
    print(f"Training finished. Artifacts saved to {outdir.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./experiments/cnn_run1")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
