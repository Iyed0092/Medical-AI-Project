import argparse
from pathlib import Path
import numpy as np
import json
import importlib
import sys

def ensure_packages():
    missing = []
    for pkg in ("shap", "matplotlib"):
        if importlib.util.find_spec(pkg) is None:
            missing.append(pkg)
    if missing:
        msg = f"Missing packages: {', '.join(missing)}. Install with: pip install " + " ".join(missing)
        raise RuntimeError(msg)

ensure_packages()
import shap
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from typing import Tuple

def load_data(run_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    p = run_dir / "test_data.npz"
    if not p.exists():
        raise FileNotFoundError("test_data.npz not found; run training first")
    npz = np.load(p)
    return npz["X_test"], npz["y_test"]

def load_model(run_dir: Path):
    best = run_dir / "best_model.h5"
    if best.exists():
        return tf.keras.models.load_model(str(best))
    return tf.keras.models.load_model(str(run_dir / "final_model"))

def compute_shap(model, background, samples):
    try:
        explainer = shap.GradientExplainer(model, background)
        shap_vals = explainer.shap_values(samples)
        return shap_vals
    except Exception:
        explainer = shap.KernelExplainer(lambda x: model.predict(x), background.reshape((background.shape[0], -1)))
        flat = samples.reshape((samples.shape[0], -1))
        shap_vals_flat = explainer.shap_values(flat, nsamples=100)
        if isinstance(shap_vals_flat, list):
            s = np.array(shap_vals_flat)
        else:
            s = np.array([shap_vals_flat])
        s = s.reshape((s.shape[0],) + samples.shape[1:])
        return s

def mean_abs_shap_map(shap_vals):
    if isinstance(shap_vals, list) or isinstance(shap_vals, tuple):
        arr = np.array(shap_vals)
        arr = np.mean(np.abs(arr), axis=0)
    else:
        arr = np.mean(np.abs(shap_vals), axis=0)
    if arr.ndim == 3:
        arr = np.mean(arr, axis=-1)
    return arr

def save_map_and_overlay(map2d, img, out_png):
    mp = map2d - np.min(map2d)
    if mp.max() > 0:
        mp = mp / mp.max()
    heat = np.uint8(255 * mp)
    heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
    heat_color = cv2.applyColorMap(heat, cv2.COLORMAP_VIRIDIS)
    img_uint8 = np.uint8(255 * np.squeeze(img))
    if img_uint8.ndim == 2:
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_uint8
    overlay = cv2.addWeighted(heat_color, 0.5, img_color, 0.5, 0)
    cv2.imwrite(str(out_png / "shap_heatmap.png"), heat_color)
    cv2.imwrite(str(out_png / "shap_overlay.png"), overlay)
    plt.figure(figsize=(6,6))
    plt.imshow(map2d, cmap="viridis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(str(out_png / "shap_map.png"), dpi=150)
    plt.close()

def main(args):
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X_test, y_test = load_data(run_dir)
    model = load_model(run_dir)
    n = X_test.shape[0]
    idxs = list(range(n)) if args.indices is None else [int(x) for x in args.indices.split(",")]
    background_size = min(20, n)
    background = X_test[np.random.RandomState(args.seed).choice(n, background_size, replace=False)]
    for idx in idxs:
        if idx < 0 or idx >= n:
            continue
        sample = X_test[idx:idx+1]
        shap_vals = compute_shap(model, background, sample)
        map2d = mean_abs_shap_map(shap_vals)[0]
        sample_out = out_dir / f"sample_{idx}"
        sample_out.mkdir(parents=True, exist_ok=True)
        save_map_and_overlay(map2d, sample[0], sample_out)
        preds = model.predict(sample).ravel()
        pred_score = float(preds[0]) if preds.shape[0] else float(preds)
        meta = {"index": int(idx), "label": int(y_test[idx]), "pred_score": pred_score}
        with open(sample_out / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
    combined = {"processed_samples": len(idxs)}
    with open(out_dir / "summary.json", "w") as f:
        json.dump(combined, f, indent=2)
    print("Saved SHAP outputs to", out_dir.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./experiments/cnn_run1")
    parser.add_argument("--output_dir", type=str, default="./experiments/cnn_run1/shap_out")
    parser.add_argument("--indices", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
