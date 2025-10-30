import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error

def dice_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = (y_true_f * y_pred_f).sum()
    return (2.*intersection + smooth)/(y_true_f.sum() + y_pred_f.sum() + smooth)

def mean_ite(y_true, y_pred):
    return float(np.mean(y_pred))

def mse(y_true, y_pred):
    return float(mean_squared_error(y_true, y_pred))

def auc_score(y_true, y_pred):
    try:
        return float(roc_auc_score(y_true, y_pred))
    except Exception:
        return float('nan')
