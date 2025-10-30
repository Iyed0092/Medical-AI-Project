import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.models import load_model
import json

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2. * intersection + smooth) / (y_true_f.sum() + y_pred_f.sum() + smooth)

def main(args):
    run = Path(args.run_dir)
    if not run.exists():
        raise FileNotFoundError("run_dir not found")
    npz = np.load(run / "test_data.npz")
    X_test = npz["X_test"]
    y_test = npz["y_test"]
    best = run / "best_model.h5"
    if best.exists():
        model = load_model(str(best))
    else:
        model = load_model(str(run / "final_model"))
    preds = model.predict(X_test, batch_size=1)
    preds_bin = (preds >= 0.5).astype('float32')
    dices = [float(dice_score(y_test[i], preds_bin[i])) for i in range(len(y_test))]
    results = {"mean_dice": float(np.mean(dices)), "median_dice": float(np.median(dices)), "n": int(len(dices))}
    with open(run / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Evaluation results:", results)

if __name__ == "__main__":
    import json, argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, default="./experiments/unet3d_run1")
    args = parser.parse_args()
    main(args)
