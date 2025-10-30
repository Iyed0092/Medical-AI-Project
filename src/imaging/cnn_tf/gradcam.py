import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import cv2
import os

def find_last_conv_layer(model):
    conv_layers = [l.name for l in model.layers if 'conv' in l.name and 'Conv' in l.__class__.__name__ or 'conv' in l.__class__.__name__.lower()]
    if conv_layers:
        return conv_layers[-1]
    for l in reversed(model.layers):
        if hasattr(l, 'output') and len(getattr(l, 'output').shape) == 4:
            return l.name
    raise RuntimeError('No conv layer found')

def make_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        top_index = tf.argmax(predictions[0]) if predictions.shape[-1] > 1 else 0
        loss = predictions[:, top_index]
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.zeros(shape=conv_outputs.shape[:2], dtype=tf.float32)
    for i in range(pooled_grads.shape[-1]):
        heatmap += conv_outputs[:, :, i] * pooled_grads[i]
    heatmap = tf.nn.relu(heatmap)
    max_val = tf.reduce_max(heatmap)
    if max_val == 0:
        return np.zeros_like(heatmap.numpy())
    heatmap = heatmap / max_val
    return heatmap.numpy()

def save_heatmap(heatmap, img, out_path, alpha=0.4):
    h = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h = np.uint8(255 * h)
    h = cv2.applyColorMap(h, cv2.COLORMAP_JET)
    img_uint8 = np.uint8(255 * np.squeeze(img))
    if img_uint8.ndim == 2:
        img_color = cv2.cvtColor(img_uint8, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_uint8
    overlay = cv2.addWeighted(h, alpha, img_color, 1 - alpha, 0)
    cv2.imwrite(str(out_path / "heatmap.png"), h)
    cv2.imwrite(str(out_path / "overlay.png"), overlay)

def main(args):
    run_dir = Path(args.run_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not run_dir.exists():
        raise FileNotFoundError("run_dir not found")
    npz_path = run_dir / "test_data.npz"
    if not npz_path.exists():
        raise FileNotFoundError("test_data.npz not found; run training first")
    data = np.load(npz_path)
    X_test = data["X_test"]
    y_test = data["y_test"]
    model = None
    best = run_dir / "best_model.h5"
    if best.exists():
        model = tf.keras.models.load_model(str(best))
    else:
        model = tf.keras.models.load_model(str(run_dir / "final_model"))
    last_conv = find_last_conv_layer(model)
    indices = list(range(len(X_test))) if args.index is None else [args.index]
    for idx in indices:
        img = X_test[idx]
        lab = int(y_test[idx])
        img_batch = np.expand_dims(img, axis=0).astype(np.float32)
        heatmap = make_gradcam_heatmap(model, img_batch, last_conv)
        sample_out = out_dir / f"sample_{idx}"
        sample_out.mkdir(parents=True, exist_ok=True)
        save_heatmap(heatmap, img, sample_out, alpha=0.4)
        preds = model.predict(img_batch).ravel()
        pred_score = float(preds[0]) if preds.shape[0] else float(preds)
        meta = {"index": int(idx), "label": int(lab), "pred_score": pred_score, "last_conv_layer": last_conv}
        with open(sample_out / "meta.json", "w") as f:
            import json
            json.dump(meta, f, indent=2)
    print("Saved heatmaps to", out_dir.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./experiments/cnn_run1")
    parser.add_argument("--output_dir", type=str, default="./experiments/cnn_run1/gradcam_out")
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    main(args)
