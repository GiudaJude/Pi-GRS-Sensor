"""CSV-based test runner for the compact LDA model.

This script mirrors the behavior of `device_main.py` but reads prerecorded
GSR samples from a CSV file instead of streaming from hardware. The CSV is
expected to contain one numeric value per row in the first column: GSR in
micro-Siemens (µS). If a header is present it will be ignored.

Usage:
    python device_test_csv.py path/to/csvdata.csv --model "ML Testing/lda_compact.npz" --fs 4 --window 8 --step 4

It will compute sliding windows (window and step are in seconds) and print
predictions for each window using the compact LDA saved by training.
"""

import argparse
import csv
import os
import sys
import numpy as np
from collections import deque
from data_processing import features_from_signal, load_compact_lda, predict_compact_lda


def load_model_auto(path):
    """Auto-detect model file type and load.

    Returns a tuple (kind, model)
      - kind == 'npz' -> model is compact params dict for predict_compact_lda
      - kind == 'joblib' -> model is a sklearn Pipeline (requires joblib & sklearn installed)
    """
    p = str(path)
    ext = os.path.splitext(p)[1].lower()
    if ext == '.npz':
        params = load_compact_lda(p)
        return 'npz', params
    elif ext in ('.joblib', '.ppl', '.pkl'):
        # lazy import joblib to avoid mandatory dependency on devices that only need .npz
        try:
            import joblib
        except Exception as e:
            raise RuntimeError(f"joblib is required to load {p}: {e}")
        pipeline = joblib.load(p)
        return 'joblib', pipeline
    else:
        # Unknown extension: try npz first, then joblib as fallback
        try:
            params = load_compact_lda(p)
            return 'npz', params
        except Exception:
            try:
                import joblib
                pipeline = joblib.load(p)
                return 'joblib', pipeline
            except Exception as e:
                raise RuntimeError(f"Failed to load model {p}: {e}")


def read_gsr_csv(path, col=None, sniff_rows=32):
    """Read numeric GSR values from a CSV file.

    If `col` is provided (int), use that column index. If `col` is None,
    auto-detect the most likely GSR column by sampling up to `sniff_rows`
    rows and selecting the numeric column with the largest standard deviation
    (timestamps are low-variance and small, while GSR µS values vary more).

    Returns a list of floats.
    """
    # Read a small sample first to sniff column candidates
    sample = []
    with open(path, 'r', newline='') as cf:
        reader = csv.reader(cf)
        for i, r in enumerate(reader):
            if not r:
                continue
            sample.append(r)
            if i + 1 >= sniff_rows:
                break

    if len(sample) == 0:
        return []

    ncols = max(len(row) for row in sample)

    # If user provided a column index, use it (but safe-guard out-of-range)
    if col is not None:
        col_idx = int(col)
        # fall back to 0 if out of bounds
        if col_idx < 0 or col_idx >= ncols:
            col_idx = 0
    else:
        # Header-aware detection: if the first non-empty sample row contains
        # text like 'gsr' or 'gsr_us' prefer that column.
        header_row = None
        for row in sample:
            # treat a row as header if any cell contains alphabetic characters
            if any(any(ch.isalpha() for ch in (cell or '')) for cell in row):
                header_row = row
                break

        if header_row is not None:
            # normalize header names and look for common GSR/EDA column names
            header_lower = [ (h or '').strip().lower() for h in header_row ]
            preferred_names = ['gsr', 'gsr_us', 'gsrµs', 'eda', 'eda_us', 'eda_us', 'eda(µs)', 'eda_us']
            found = False
            for name in preferred_names:
                if name in header_lower:
                    col_idx = header_lower.index(name)
                    found = True
                    break
            if not found:
                # If header exists but no preferred names, pick the first numeric column after header
                col_idx = None
                for c in range(ncols):
                    vals = []
                    for row in sample[1:]:
                        if c >= len(row):
                            continue
                        try:
                            vals.append(float(row[c]))
                        except Exception:
                            break
                    if len(vals) > 0:
                        col_idx = c
                        break
                if col_idx is None:
                    col_idx = 0
        else:
            # auto-detect numeric column using variance but penalize monotonic columns
            col_scores = []
            for c in range(ncols):
                vals = []
                for row in sample:
                    if c >= len(row):
                        continue
                    try:
                        v = float(row[c])
                        vals.append(v)
                    except Exception:
                        continue
                if len(vals) < 2:
                    col_scores.append((c, -1.0))
                    continue
                arr = np.array(vals, dtype=float)
                # base score: std + small mean magnitude
                score = float(np.std(arr)) + 0.01 * float(np.abs(np.mean(arr)))
                # penalize strongly if the column is mostly strictly increasing (likely timestamps)
                diffs = np.diff(arr)
                frac_increasing = float(np.sum(diffs > 0)) / max(1.0, len(diffs))
                if frac_increasing > 0.9:
                    score *= 0.1
                col_scores.append((c, score))

            # pick column with max score
            col_idx = max(col_scores, key=lambda x: x[1])[0]

    # Now read full CSV using selected column index
    vals = []
    with open(path, 'r', newline='') as cf:
        reader = csv.reader(cf)
        for r in reader:
            if not r:
                continue
            if col_idx >= len(r):
                continue
            try:
                v = float(r[col_idx])
                vals.append(v)
            except Exception:
                # skip header or malformed row
                continue

    print(f"Auto-detected CSV column {col_idx} for GSR values (use --col to override)")
    return vals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('csv', help='Path to CSV containing GSR values (µS) in first column')
    p.add_argument('--model', default='ML Testing/lda_compact.npz', help='Path to compact LDA .npz')
    p.add_argument('--fs', type=float, default=4.0, help='Sampling rate in Hz')
    p.add_argument('--window', type=float, default=8.0, help='Window length in seconds')
    p.add_argument('--step', type=float, default=4.0, help='Step length in seconds (hop)')
    p.add_argument('--col', type=int, default=0, help='CSV column index containing numeric GSR values')
    p.add_argument('--save', help='Optional: save predictions to CSV file')
    p.add_argument('--verbose', action='store_true', help='Print per-window features, scaled values, and logits for debugging')
    p.add_argument('--smoother', type=int, default=0, help='Majority-vote smoothing window (in windows). 0 = disabled')
    p.add_argument('--emit-on-change', action='store_true', help='Only print predictions when the label changes (reduces console output)')
    p.add_argument('--min-emit-interval', type=float, default=0.0, help='Minimum seconds between printed emissions when --emit-on-change is used')
    args = p.parse_args()

    csv_path = os.path.expanduser(args.csv)
    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        sys.exit(1)

    print(f"Reading CSV: {csv_path}")
    signal = read_gsr_csv(csv_path, col=args.col)
    if len(signal) == 0:
        print("No numeric samples found in CSV (check column/index).")
        sys.exit(1)

    print(f"Loaded {len(signal)} samples from CSV (interpreted as µS). fs={args.fs} Hz")

    # compute features for all sliding windows
    features = features_from_signal(signal, fs=args.fs, window_seconds=args.window, step_seconds=args.step)
    if features.shape[0] == 0:
        print("No windows extracted with the given window/step parameters.")
        sys.exit(1)

    print(f"Computed feature matrix: {features.shape} (n_windows x n_features)")

    # load model (auto-detect .npz or joblib/.pkl)
    try:
        kind, model_obj = load_model_auto(args.model)
        print(f"Loaded model ({kind}): {args.model}")
    except Exception as e:
        print(f"Failed to load model {args.model}: {e}")
        model_obj = None
        kind = None

    # Print a brief model summary to help diagnose extreme probabilities
    try:
        if model_obj is not None:
            if kind == 'npz':
                print("Model summary (npz): classes=", model_obj.get('classes'))
                mm = model_obj.get('scaler_mean')
                ss = model_obj.get('scaler_scale')
                if mm is not None and ss is not None:
                    print("  scaler_mean (first 5):", np.array2string(mm[:5], precision=4))
                    print("  scaler_scale (first 5):", np.array2string(ss[:5], precision=4))
            elif kind == 'joblib':
                try:
                    clf = model_obj
                    print("Model summary (joblib):")
                    if hasattr(clf, 'classes_'):
                        print("  classes_:", clf.classes_)
                    print("  has_predict_proba:", hasattr(clf, 'predict_proba'))
                    print("  has_decision_function:", hasattr(clf, 'decision_function'))
                except Exception:
                    print("  (could not introspect joblib model)")
    except Exception as e:
        print(f"Warning: failed to print model summary: {e}")

    # Predict for each window and print timestamp and prediction
    predictions = []
    times = []
    # optional smoother
    class MajorityVoteSmoother:
        def __init__(self, window=3):
            self.buf = deque(maxlen=window)

        def update(self, label_or_prob):
            # accept either a probability (float) or 0/1 label
            if isinstance(label_or_prob, float):
                lbl = 1 if label_or_prob > 0.5 else 0
            else:
                try:
                    lbl = int(label_or_prob)
                except Exception:
                    lbl = 0
            self.buf.append(lbl)
            # majority
            if len(self.buf) == 0:
                return lbl
            return 1 if sum(self.buf) > (len(self.buf) / 2) else 0

    smoother = MajorityVoteSmoother(window=args.smoother) if args.smoother and args.smoother > 0 else None
    # emission control state (used when --emit-on-change is enabled)
    last_emitted_label = None
    last_emit_time = -1e9
    window_size = int(round(args.window * args.fs))
    step_size = int(round(args.step * args.fs))

    for i in range(features.shape[0]):
        start_sample = i * step_size
        timestamp = start_sample / args.fs
        X = features[i]
        if model_obj is None:
            label = None
            prob = None
        else:
            if kind == 'npz':
                # For diagnostics optionally print feature/scaler/logit info
                if args.verbose:
                    mean = model_obj['scaler_mean']
                    scale = model_obj['scaler_scale']
                    print(f"Window {i} raw features: {X}")
                    # avoid division by zero in printed scaled values
                    safe_scale = np.where(scale == 0, 1e-6, scale)
                    Xs = (np.asarray(X, dtype=np.float32) - mean) / safe_scale
                    logits = np.dot(Xs.reshape(1, -1), model_obj['coef'].T) + model_obj['intercept']
                    print(f"  scaler_mean: {mean}")
                    print(f"  scaler_scale: {scale}")
                    print(f"  scaled features: {Xs}")
                    print(f"  logits: {logits}")

                # compute logits and a sigmoid-based probability for binary LDA
                mean = model_obj['scaler_mean']
                scale = model_obj['scaler_scale']
                safe_scale = np.where(scale == 0, 1e-6, scale)
                Xs = (np.asarray(X, dtype=np.float32) - mean) / safe_scale
                logits = np.dot(Xs.reshape(1, -1), model_obj['coef'].T) + model_obj['intercept']
                # if multi-class logits, convert to softmax; if single logit, use sigmoid
                if logits.ndim == 2 and logits.shape[1] > 1:
                    ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                    probs = ex / np.sum(ex, axis=1, keepdims=True)
                    idx = int(np.argmax(probs, axis=1)[0])
                    label = int(model_obj['classes'][idx])
                    prob = float(probs[0, idx])
                else:
                    val = float(logits.ravel()[0])
                    # clamp logits to a safe range to avoid overflow and extremely large probabilities
                    val = float(np.clip(val, -50.0, 50.0))
                    prob = 1.0 / (1.0 + np.exp(-val))
                    pred = predict_compact_lda(X, model_obj)
                    label = int(pred[0] if hasattr(pred, '__len__') else pred)
            elif kind == 'joblib':
                # model_obj is a sklearn Pipeline or estimator
                try:
                    # ensure 2D input for sklearn
                    X2 = np.asarray(X, dtype=np.float32).reshape(1, -1)
                    yhat = model_obj.predict(X2)
                    label = int(yhat[0]) if hasattr(yhat, '__len__') else int(yhat)
                    # try to get a probability/confidence
                    prob = None
                    try:
                        if hasattr(model_obj, 'predict_proba'):
                            proba = model_obj.predict_proba(X2)
                            # pick the probability of the predicted class
                            if proba.shape[1] == 2:
                                prob = float(proba[0, 1]) if label == 1 else float(proba[0, 0])
                            else:
                                # multiclass: prob of predicted index
                                prob = float(np.max(proba))
                        elif hasattr(model_obj, 'decision_function'):
                            df = model_obj.decision_function(X2)
                            val = float(df.ravel()[0])
                            prob = 1.0 / (1.0 + np.exp(-val))
                    except Exception:
                        prob = None
                    if args.verbose:
                        print(f"Window {i} raw features: {X}")
                        # try to access decision_function or predict_proba if available
                        try:
                            if hasattr(model_obj, 'decision_function'):
                                df = model_obj.decision_function(X2)
                                print(f"  decision_function: {df}")
                            if hasattr(model_obj, 'predict_proba'):
                                proba = model_obj.predict_proba(X2)
                                print(f"  predict_proba: {proba}")
                        except Exception as e:
                            print(f"  Could not get extra diagnostics from sklearn model: {e}")
                except Exception as e:
                    print(f"  Prediction failed for joblib model: {e}")
                    label = None
            else:
                label = None
        # map label to human-readable string
        label_str = None
        if label is None:
            label_str = 'N/A'
        elif label == 1:
            label_str = 'Stress'
        elif label == 0:
            label_str = 'Baseline'
        else:
            label_str = str(label)

        # apply optional smoothing
        if smoother is not None:
            sm_label = smoother.update(prob if prob is not None else label)
            if sm_label is None:
                sm_label_str = 'N/A'
            elif sm_label == 1:
                sm_label_str = 'Stress'
            elif sm_label == 0:
                sm_label_str = 'Baseline'
            else:
                sm_label_str = str(sm_label)
        else:
            sm_label = None
            sm_label_str = None

        # Decide whether to print this window's summary based on emission control
        emission_label = sm_label if smoother is not None else label
        should_emit = True
        if args.emit_on_change:
            # when emission on change is enabled, only print when the (smoothed) label changes
            if emission_label is None:
                should_emit = False
            else:
                time_since = timestamp - last_emit_time
                if emission_label != last_emitted_label and time_since >= args.min_emit_interval:
                    should_emit = True
                else:
                    should_emit = False

        if should_emit:
            if prob is None:
                if smoother is None:
                    print(f"{timestamp:8.2f}s -> Prediction: {label_str} ({label})")
                else:
                    print(f"{timestamp:8.2f}s -> Prediction: {label_str} ({label})  smoothed: {sm_label_str} ({sm_label})")
            else:
                if smoother is None:
                    print(f"{timestamp:8.2f}s -> Prediction: {label_str} ({label})  prob={prob:.3f}")
                else:
                    print(f"{timestamp:8.2f}s -> Prediction: {label_str} ({label})  prob={prob:.3f}  smoothed: {sm_label_str} ({sm_label})")
            last_emitted_label = emission_label
            last_emit_time = timestamp

        # always store per-window prediction for saving/analysis
        predictions.append({'label': label, 'label_str': label_str, 'prob': prob, 'label_smooth': sm_label, 'label_smooth_str': sm_label_str})
        times.append(timestamp)

    if args.save:
        out_path = os.path.expanduser(args.save)
        try:
            with open(out_path, 'w', newline='') as of:
                w = csv.writer(of)
                # include smoothed label columns if smoothing was used
                if args.smoother and args.smoother > 0:
                    w.writerow(['time_s', 'label', 'label_str', 'probability', 'label_smooth', 'label_smooth_str'])
                    for t, p in zip(times, predictions):
                        w.writerow([
                            f"{t:.3f}",
                            p.get('label'),
                            p.get('label_str'),
                            '' if p.get('prob') is None else f"{p.get('prob'):.6f}",
                            p.get('label_smooth'),
                            p.get('label_smooth_str')
                        ])
                else:
                    w.writerow(['time_s', 'label', 'label_str', 'probability'])
                    for t, p in zip(times, predictions):
                        w.writerow([f"{t:.3f}", p.get('label'), p.get('label_str'), '' if p.get('prob') is None else f"{p.get('prob'):.6f}"])
            print(f"Saved predictions to {out_path}")
        except Exception as e:
            print(f"Failed to save predictions: {e}")

    # Post-run summary: count extreme probabilities
    try:
        probs = [p.get('prob') for p in predictions if p.get('prob') is not None]
        if len(probs) > 0:
            probs = np.asarray(probs, dtype=np.float32)
            n_total = len(probs)
            n_one = int(np.sum(probs >= 0.999999))
            n_zero = int(np.sum(probs <= 1e-6))
            print(f"\nPrediction probabilities summary: n={n_total}, #~1={n_one}, #~0={n_zero}")
            if n_one > 0:
                print("Warning: Some probabilities are effectively 1. This can be caused by very large logits or a mismatch between training and test feature scales.")
                idxs = np.where(probs >= 0.999999)[0][:5]
                for idx in idxs:
                    print(f"  Example prob= {probs[idx]:.12f}")
    except Exception:
        pass


if __name__ == '__main__':
    main()
