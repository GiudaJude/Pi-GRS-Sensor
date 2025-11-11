import numpy as np
from collections import deque
import os

FEATURE_NAMES = [
    'Mean', 'Std', 'Var', 'Min', 'Max', 'Range', 'Median',
    'Q25', 'Q75', 'IQR'
]


def extract_standard_features_from_window(window):
    """Compute 10 standard statistical features from a 1-D numpy window.

    Returns a 1-D numpy array of length 10 (float32).
    """
    w = np.asarray(window, dtype=np.float32)
    if w.size == 0:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    mean_eda = np.mean(w)
    std_eda = np.std(w)
    var_eda = np.var(w)
    min_eda = np.min(w)
    max_eda = np.max(w)
    range_eda = max_eda - min_eda
    median_eda = np.median(w)
    q25 = np.percentile(w, 25)
    q75 = np.percentile(w, 75)
    iqr = q75 - q25

    return np.asarray([mean_eda, std_eda, var_eda, min_eda, max_eda, range_eda, median_eda, q25, q75, iqr], dtype=np.float32)


def windows_from_signal(signal, fs=4.0, window_seconds=8.0, step_seconds=4.0):
    """Yield windows (numpy arrays) from a 1-D signal array.

    Defaults assume 4 Hz sampling (WESAD wrist) and 8s windows with 50% overlap.
    """
    sig = np.asarray(signal)
    window_size = max(1, int(round(window_seconds * fs)))
    step_size = max(1, int(round(step_seconds * fs)))

    for start in range(0, len(sig) - window_size + 1, step_size):
        yield sig[start:start + window_size]


def features_from_signal(signal, fs=4.0, window_seconds=8.0, step_seconds=4.0):
    """Return a feature matrix (n_windows, n_features) computed from signal."""
    feats = []
    for w in windows_from_signal(signal, fs=fs, window_seconds=window_seconds, step_seconds=step_seconds):
        feats.append(extract_standard_features_from_window(w))
    if len(feats) == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack(feats)


# Compact LDA loader / predictor (numpy-only) — same format as saved by trainer
def load_compact_lda(path='ML Testing/lda_compact.npz'):
    path = os.path.expanduser(path)
    d = np.load(path)
    params = {
        'coef': d['coef'].astype(np.float32),
        'intercept': d['intercept'].astype(np.float32),
        'classes': d['classes'].astype(np.int64),
        'scaler_mean': d['scaler_mean'].astype(np.float32),
        'scaler_scale': d['scaler_scale'].astype(np.float32)
    }
    return params


def predict_compact_lda(X, params):
    """Predict using compact LDA params (numpy-only).

    X: array shape (n_samples, n_features)
    params: dict returned by load_compact_lda
    Returns: predicted class labels (original class integers)
    """
    Xf = np.asarray(X, dtype=np.float32)
    if Xf.ndim == 1:
        Xf = Xf.reshape(1, -1)
    # apply scaler
    Xs = (Xf - params['scaler_mean']) / params['scaler_scale']
    # linear decision: coef dot x + intercept
    logits = np.dot(Xs, params['coef'].T) + params['intercept']
    # handle binary or multi-class
    if logits.ndim == 2 and logits.shape[1] > 1:
        preds = np.argmax(logits, axis=1)
    else:
        preds = (logits.ravel() > 0).astype(np.int64)
    try:
        mapped = params['classes'][preds]
        return mapped
    except Exception:
        return preds


if __name__ == '__main__':
    # quick local test
    x = np.sin(np.linspace(0, 10, 100)) + 0.1 * np.random.randn(100)
    F = features_from_signal(x, fs=4.0, window_seconds=8.0, step_seconds=4.0)
    print('Feature matrix shape:', F.shape)
