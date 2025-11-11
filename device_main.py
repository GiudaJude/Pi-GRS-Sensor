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
        raise ValueError(f"Unsupported model file extension: {ext}")


def run_stream(channel, model_path='ML Testing/lda_compact.npz', fs=4.0, window_seconds=8.0, step_seconds=4.0):
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
                feats = features_from_signal(window, fs=fs, window_seconds=window_seconds, step_seconds=window_seconds)
                # features_from_signal returns one or zero rows for a single window input
                if feats.shape[0] > 0:
                    X = feats[0]
                    if model_obj is not None:
                        if model_kind == 'npz':
                            pred = predict_compact_lda(X, model_obj)
                            prob = 1.0 / (1.0 + np.exp(-pred[1])) if len(pred) > 1 else None
                            label = 'Stress' if pred[0] == 1 else 'Baseline'
                        elif model_kind == 'joblib':
                            X2 = X.reshape(1, -1)
                            pred = model_obj.predict(X2)
                            prob = None
                            if hasattr(model_obj, 'predict_proba'):
                                proba = model_obj.predict_proba(X2)
                                prob = proba[0, 1] if pred[0] == 1 else proba[0, 0]
                            label = 'Stress' if pred[0] == 1 else 'Baseline'
                        if prob is not None:
                            print(f"{now:8.2f}s -> Prediction: {label} (prob={prob:.3f})")
                        else:
                            print(f"{now:8.2f}s -> Prediction: {label}")
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
    p.add_argument('--window', type=float, default=100, help='Window length in seconds')
    p.add_argument('--step', type=float, default=30, help='Step length in seconds (hop)')
    args = p.parse_args()

    run_stream(args.channel, model_path=args.model, fs=args.fs, window_seconds=args.window, step_seconds=args.step)
