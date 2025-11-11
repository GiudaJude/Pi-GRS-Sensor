import numpy as np
import csv
import os
from data_processing import features_from_signal, load_compact_lda, predict_compact_lda

CSV = 'csvdata.csv'
NPZ = 'ML Testing/lda_compact.npz'
JOBLIB = 'ML Testing/lda_full_pipeline.joblib'
FS = 4.0
WINDOW = 8.0
STEP = 4.0
SHOW = 6  # windows to show

def read_col(path, col=1):
    vals=[]
    with open(path,'r',newline='') as f:
        r=csv.reader(f)
        for row in r:
            if not row: continue
            try:
                vals.append(float(row[col]))
            except Exception:
                continue
    return np.array(vals, dtype=float)

print("Loading CSV:", CSV)
sig = read_col(CSV, col=1)
print(" raw sample (first 20):", sig[:20])
print(" raw stats: n, min, median, max, mean, std =", len(sig), np.min(sig), np.median(sig), np.max(sig), np.mean(sig), np.std(sig))

print("\nComputing features...")
F = features_from_signal(sig, fs=FS, window_seconds=WINDOW, step_seconds=STEP)
print(" features shape:", F.shape)
if F.shape[0] == 0:
    raise SystemExit("No windows extracted; check fs/window/step")

print("\nLoading compact .npz:", NPZ)
npz = load_compact_lda(NPZ)
mm = npz['scaler_mean']; ss = npz['scaler_scale']; coef = npz['coef']; intr = npz['intercept']; cls = npz['classes']
print(" classes:", cls)
print(" scaler_mean (first10):", mm[:10])
print(" scaler_scale (first10):", ss[:10])
print(" coef shape:", coef.shape, " intercept:", intr)

def sigmoid(x): return 1.0/(1.0+np.exp(-x))

for i in range(min(SHOW, F.shape[0])):
    x = F[i].astype(np.float32)
    safe_ss = np.where(ss==0, 1e-6, ss)
    xs = (x - mm) / safe_ss
    logits = np.dot(xs.reshape(1,-1), coef.T) + intr
    if logits.ndim==2 and logits.shape[1]==1:
        val = float(np.clip(logits.ravel()[0], -50, 50))
        prob = sigmoid(val)
    elif logits.ndim==2:
        # multiclass softmax
        ex = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = ex / np.sum(ex, axis=1, keepdims=True)
        prob = float(np.max(probs))
    else:
        prob = None
    print(f"\nWindow {i} raw features: {x}")
    print(" scaled:", xs)
    print(" logits:", logits, " prob:", prob)

# If joblib exists, compare predictions
if os.path.exists(JOBLIB):
    try:
        import joblib
        pipeline = joblib.load(JOBLIB)
        print("\nLoaded pipeline joblib; pipeline classes (if available):", getattr(pipeline, 'classes_', None))
        for i in range(min(SHOW, F.shape[0])):
            X2 = F[i].reshape(1,-1).astype(np.float32)
            yhat = pipeline.predict(X2)
            proba = None
            if hasattr(pipeline, 'predict_proba'):
                proba = pipeline.predict_proba(X2)
            print(f" pipeline window {i} predict: {yhat}, proba: {proba}")
    except Exception as e:
        print("Failed loading joblib pipeline:", e)

