# src/imaging/cnn_tf/eval.py
"""
Loads saved model and evaluates on saved synthetic test set.

Usage:
    python src/imaging/cnn_tf/eval.py --run_dir ./experiments/cnn_run1
"""
import argparse
from pathlib import Path
import numpy as np
import json
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import tensorflow as tf

def load_test_data(run_dir):
    p = Path(run_dir)
    npz = np.load(p / "test_data.npz")
    return npz["X_test"], npz["y_test"]

def main(args):
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"{run_dir} does not exist. Run training first.")

    print("Loading test data...")
    X_test, y_test = load_test_data(run_dir)

    # load best model if exists, otherwise final_model
    best_h5 = run_dir / "best_model.h5"
    if best_h5.exists():
        model = tf.keras.models.load_model(str(best_h5))
        print("Loaded best_model.h5")
    else:
        model = tf.keras.models.load_model(str(run_dir / "final_model"))
        print("Loaded final_model")

    preds = model.predict(X_test, batch_size=32).ravel()
    # threshold at 0.5 for binary
    preds_bin = (preds >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds_bin)
    f1 = f1_score(y_test, preds_bin)
    try:
        auc = roc_auc_score(y_test, preds)
    except Exception:
        auc = float("nan")

    results = {"accuracy": float(acc), "f1": float(f1), "auc": float(auc)}
    print("Evaluation results:", results)

    # save results
    with open(run_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./experiments/cnn_run1")
    args = parser.parse_args()
    main(args)
