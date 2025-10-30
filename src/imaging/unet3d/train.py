import argparse
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from model import get_unet3d
import json

def make_synthetic_volumes(n_samples=100, shape=64, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(n_samples, shape, shape, shape)).astype('float32')
    X = (X - X.min())/(X.max()-X.min()+1e-9)
    y = np.zeros((n_samples, shape, shape, shape), dtype='uint8')
    rr, cc, zz = np.ogrid[:shape, :shape, :shape]
    for i in range(n_samples):
        cx = rng.randint(shape//4, 3*shape//4)
        cy = rng.randint(shape//4, 3*shape//4)
        cz = rng.randint(shape//4, 3*shape//4)
        r = rng.randint(max(1, shape//10), max(2, shape//6))
        mask = (rr - cx)**2 + (cc - cy)**2 + (zz - cz)**2 <= r**2
        y[i][mask] = 1
        X[i][mask] = np.clip(X[i][mask] + rng.uniform(0.3,1.0), 0.0, 1.0)
    X = X[..., None]
    y = y[..., None].astype('float32')
    return X, y

def main(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    X, y = make_synthetic_volumes(n_samples=args.n_samples, shape=args.shape, seed=args.seed)
    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.3, random_state=args.seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=args.seed)
    model = get_unet3d(input_shape=(args.shape, args.shape, args.shape, 1), base_filters=args.base_filters)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=str(out / "best_model.h5"), save_best_only=True, monitor='val_loss', mode='min'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)
    ]
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=args.epochs, batch_size=args.batch_size, callbacks=callbacks, verbose=2)
    model.save(str(out / "final_model"))
    np.savez_compressed(out / "test_data.npz", X_test=X_test, y_test=y_test)
    metrics = {"history": {k: [float(x) for x in v] for k, v in history.history.items()}}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved artifacts to", out.resolve())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="./experiments/unet3d_run1")
    parser.add_argument("--n_samples", type=int, default=80)
    parser.add_argument("--shape", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--base_filters", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)
