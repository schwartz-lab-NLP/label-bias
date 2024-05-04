import numpy as np
import pandas as pd
from sklearn.metrics import f1_score as f1


def rsd(y_true, y_pred):
    classwise_acc = [(y_pred[y_true == label] == label).mean() for label in np.unique(y_true)]
    acc = (y_pred == y_true).mean()
    if acc == 0.0:
        rsd = 0.0
    else:
        rsd = np.std(classwise_acc) / acc
    return rsd


def bias_score(y_true, y_probs):
    mean_probs_per_label = [y_probs[y_true == label].mean(axis=0) for label in np.unique(y_true)]
    bias_estimate = np.mean(mean_probs_per_label, axis=0)
    bias_score = 0.5 * np.abs(bias_estimate - (np.ones_like(bias_estimate)/len(bias_estimate))).sum()
    return bias_score
