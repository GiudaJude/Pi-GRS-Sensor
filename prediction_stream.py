import time
import argparse
from collections import deque
import joblib
import os
import numpy as np

# local imports
import sensor
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
        try:
            pipeline = joblib.load(p)
            return 'joblib', pipeline
        except Exception as e:
            raise RuntimeError(f"joblib is required to load {p}: {e}")
    else:
        # Try npz first, then joblib as fallback
        try:
            params = load_compact_lda(p)
            return 'npz', params
        except Exception:
            try:
                pipeline = joblib.load(p)
                return 'joblib', pipeline
            except Exception as e:
                raise ValueError(f"Unsupported or unreadable model file: {e}")


def run_stream(channel, model_path='ML Testing/lda_compact.npz', fs=4.0, window_seconds=8.0, step_seconds=4.0, smoother_windows=0, emit_on_change=False, min_emit_interval=0.0, verbose=False, threshold=None, hysteresis_exit=None, min_state_windows=1, labels_only=False):
    # pass fs so the simulator (if used) samples with the same rate
    sensor_obj = sensor.GroveGSRSensor(channel, fs=fs)
    sensor_obj = sensor.GroveGSRSensor(channel, fs=fs)
    if sensor_obj.simulate:
        print("Sensor wrapper: RUNNING IN SIMULATION mode (no grove.adc detected or ADC init failed).")
    else:
        print("Sensor wrapper: USING REAL ADC HARDWARE.")
        
    interval = 1.0 / fs

    window_size = max(1, int(round(window_seconds * fs)))
    step_size = max(1, int(round(step_seconds * fs)))

    buf = deque(maxlen=window_size)
    step_counter = 0

    class MajorityVoteSmoother:
        def __init__(self, window=3):
            self.buf = deque(maxlen=window)

        def update(self, label_or_prob):
            if isinstance(label_or_prob, float):
                lbl = 1 if label_or_prob > 0.5 else 0
            else:
                try:
                    lbl = int(label_or_prob)
                except Exception:
                    lbl = 0
            self.buf.append(lbl)
            if len(self.buf) == 0:
                return lbl
            return 1 if sum(self.buf) > (len(self.buf) / 2) else 0

    smoother = MajorityVoteSmoother(window=smoother_windows) if smoother_windows and smoother_windows > 0 else None
    # Hysteresis gate mirroring csv_prediction
    class HysteresisGate:
        def __init__(self, enter_threshold=0.7, exit_threshold=0.5, min_windows=3):
            self.enter = float(enter_threshold)
            self.exit = float(exit_threshold)
            self.min_windows = int(max(1, min_windows))
            self.state = 0
            self.last_change_idx = -10**9
            self.idx = 0
        def update(self, p_stress):
            try:
                ps = float(p_stress)
            except Exception:
                ps = 0.0
            can_switch = (self.idx - self.last_change_idx) >= self.min_windows
            if self.state == 0 and ps >= self.enter and can_switch:
                self.state = 1
                self.last_change_idx = self.idx
            elif self.state == 1 and ps < self.exit and can_switch:
                self.state = 0
                self.last_change_idx = self.idx
            self.idx += 1
            return self.state
    gate = HysteresisGate(
        enter_threshold=threshold if threshold is not None else 0.7,
        exit_threshold=hysteresis_exit if hysteresis_exit is not None else 0.5,
        min_windows=int(max(1, min_state_windows))
    )
    last_emitted_label = None
    last_emit_time = -1e9

    # load model once
    model_kind, model_obj = None, None
    try:
        model_kind, model_obj = load_model_auto(model_path)
        print(f"Loaded model ({model_kind}): {model_path}")
    except Exception as e:
        print(f"Warning: could not load model {model_path}: {e}")
        model_kind, model_obj = None, None

    print(f"Starting GSR streaming on channel {channel} (fs={fs} Hz). Window={window_seconds}s step={step_seconds}s")
    start = time.time()

    try:
        while True:
            # use read_raw() from the wrapper; convert to µS using the class helper
            adc_value = sensor_obj.read_raw()
            gsr_us = sensor.GroveGSRSensor.adc_to_us(adc_value)
            now = time.time() - start
            if gsr_us is None:
                # skip invalid samples
                time.sleep(interval)
                continue

            buf.append(gsr_us)
            step_counter += 1

            # When we have a full window and reached the step interval, compute features & predict
            if len(buf) == window_size and step_counter >= step_size:
                step_counter = 0
                window = list(buf)
                feats = features_from_signal(window, fs=fs, window_seconds=window_seconds, step_seconds=step_seconds)
                # features_from_signal returns one or zero rows for a single window input
                if feats.shape[0] > 0:
                    X = feats[0]
                    if model_obj is not None:
                        if model_kind == 'npz':
                            # Compute scaled features and logits similar to csv_prediction
                            mean = model_obj.get('scaler_mean')
                            scale = model_obj.get('scaler_scale')
                            safe_scale = np.where(scale == 0, 1e-6, scale) if scale is not None else 1.0
                            Xs = (np.asarray(X, dtype=np.float32) - mean) / safe_scale if mean is not None else np.asarray(X, dtype=np.float32)
                            logits = np.dot(Xs.reshape(1, -1), model_obj['coef'].T) + model_obj['intercept']
                            if logits.ndim == 2 and logits.shape[1] > 1:
                                ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                                probs = ex / np.sum(ex, axis=1, keepdims=True)
                                idx = int(np.argmax(probs, axis=1)[0])
                                label_int = int(model_obj['classes'][idx])
                                prob = float(probs[0, idx])
                            else:
                                val = float(logits.ravel()[0])
                                val = float(np.clip(val, -50.0, 50.0))
                                prob = 1.0 / (1.0 + np.exp(-val))
                                pred = predict_compact_lda(X, model_obj)
                                label_int = int(pred[0] if hasattr(pred, '__len__') else pred)
                            label = 'Stress' if label_int == 1 else 'Baseline'
                            if verbose:
                                print(f"Window features: {X}")
                                print(f"  scaled: {Xs}")
                                print(f"  logits: {logits}")
                        elif model_kind == 'joblib':
                            X2 = X.reshape(1, -1)
                            pred = model_obj.predict(X2)
                            prob = None
                            if hasattr(model_obj, 'predict_proba'):
                                proba = model_obj.predict_proba(X2)
                                # always take stress probability (class 1) if binary
                                prob = float(proba[0, 1]) if proba.shape[1] == 2 else float(np.max(proba))
                            elif hasattr(model_obj, 'decision_function'):
                                df = model_obj.decision_function(X2)
                                val = float(df.ravel()[0])
                                prob = 1.0 / (1.0 + np.exp(-val))
                            label = 'Stress' if pred[0] == 1 else 'Baseline'
                        # Optional smoothing and emission control
                        emission_label = None
                        sm_label_str = None
                        if smoother is not None:
                            sm_label = smoother.update(prob if prob is not None else (1 if label == 'Stress' else 0))
                            emission_label = sm_label
                            sm_label_str = 'Stress' if sm_label == 1 else 'Baseline'
                        else:
                            # apply gate even without smoother
                            gated = gate.update(prob if prob is not None else (1 if label == 'Stress' else 0))
                            emission_label = gated
                            label = 'Stress' if gated == 1 else 'Baseline'

                        should_emit = True
                        if emit_on_change:
                            time_since = now - last_emit_time
                            if emission_label != last_emitted_label and time_since >= min_emit_interval:
                                should_emit = True
                            else:
                                should_emit = False

                        if should_emit:
                            last_emitted_label = emission_label
                            last_emit_time = now
                            out_label = sm_label_str if sm_label_str is not None else label
                            if labels_only:
                                # Print just the label string without timestamp or probability
                                print(out_label)
                            else:
                                if prob is not None:
                                    print(f"{now:8.2f}s -> Prediction: {out_label} (prob={prob:.3f})")
                                else:
                                    print(f"{now:8.2f}s -> Prediction: {out_label}")
                    else:
                        print(f"{now:8.2f}s -> Features ready (no model): {X}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print('\nStopping streaming')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('channel', type=int, help='ADC channel for the GSR sensor')
    p.add_argument('--model', default='ML Testing/lda_compact.npz', help='Path to compact LDA .npz')
    p.add_argument('--fs', type=float, default=4.0, help='Sampling rate in Hz')
    p.add_argument('--window', type=float, default=60, help='Window length in seconds')
    p.add_argument('--step', type=float, default=30, help='Step length in seconds (hop)')
    p.add_argument('--smoother', type=int, default=0, help='Majority-vote smoothing window (in windows). 0 = disabled')
    p.add_argument('--emit-on-change', action='store_true', help='Only print predictions when the label changes')
    p.add_argument('--min-emit-interval', type=float, default=0.0, help='Minimum seconds between emissions when using --emit-on-change')
    p.add_argument('--verbose', action='store_true', help='Print per-window diagnostics')
    p.add_argument('--threshold', type=float, default=None, help='Stress entry threshold on probability')
    p.add_argument('--hysteresis-exit', type=float, default=None, help='Stress exit threshold on probability')
    p.add_argument('--min-state-windows', type=int, default=3, help='Minimum number of windows to stay in a state before switching')
    p.add_argument('--threshold-file', type=str, default='', help='Optional path to a saved threshold text file')
    p.add_argument('--labels-only', action='store_true', help='Print only the label (Stress/Baseline) without timestamps or probabilities')
    args = p.parse_args()

    # Load threshold from file if provided
    thr = args.threshold
    thr_exit = args.hysteresis_exit
    if args.threshold_file:
        try:
            with open(os.path.expanduser(args.threshold_file), 'r') as f:
                txt = f.read().strip()
                if not thr:
                    thr = float(txt)
                if thr_exit is None:
                    thr_exit = max(0.0, min(1.0, float(thr) - 0.1))
            print(f"Using threshold from file: {args.threshold_file} -> {thr:.3f}")
        except Exception as e:
            print(f"Warning: failed to read threshold file: {e}")

    run_stream(
        args.channel,
        model_path=args.model,
        fs=args.fs,
        window_seconds=args.window,
        step_seconds=args.step,
        smoother_windows=args.smoother,
        emit_on_change=args.emit_on_change,
        min_emit_interval=args.min_emit_interval,
        verbose=args.verbose,
        threshold=thr,
        hysteresis_exit=thr_exit,
        min_state_windows=args.min_state_windows,
        labels_only=args.labels_only,
    )
