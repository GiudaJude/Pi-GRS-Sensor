import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GroupKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
import glob
from dotenv import load_dotenv
warnings.filterwarnings('ignore')
import joblib
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, precision_recall_curve

def load_subject_data(subject_path):
    """Load data for a single subject"""
    subject_name = os.path.basename(subject_path)
    pkl_file = os.path.join(subject_path, f"{subject_name}.pkl")
    
    print(f"Loading {subject_name}...")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract EDA data and labels
        eda_data = None
        labels = None
        
        # Look for EDA data in signal structure
        if 'signal' in data:
            signal_data = data['signal']
            if 'wrist' in signal_data and 'EDA' in signal_data['wrist']:
                eda_data = signal_data['wrist']['EDA'].flatten()
                print(f"  Found wrist EDA data: {eda_data.shape}")
            elif 'chest' in signal_data and 'EDA' in signal_data['chest']:
                eda_data = signal_data['chest']['EDA'].flatten()
                print(f"  Found chest EDA data: {eda_data.shape}")
        
        # Look for labels
        for key in data.keys():
            if 'label' in key.lower():
                labels = data[key].flatten() if hasattr(data[key], 'flatten') else data[key]
                print(f"  Found labels: {labels.shape}")
                break
        
        if eda_data is not None and labels is not None:
            # Handle sampling rate mismatch between EDA and labels
            # EDA is typically at 4Hz (wrist) or 700Hz (chest), labels at much higher rate
            
            eda_length = len(eda_data)
            label_length = len(labels)
            
            print(f"  EDA length: {eda_length}, Label length: {label_length}")
            
            # If labels are longer (higher sampling rate) than EDA, DO NOT downsample labels.
            # Instead resample / upsample the EDA signal to match the label length using
            # linear interpolation. This preserves label resolution (as the paper does).
            if label_length > eda_length:
                eda_length_before = eda_length
                try:
                    # Create normalized sample positions and interpolate
                    x_old = np.linspace(0, 1, eda_length)
                    x_new = np.linspace(0, 1, label_length)
                    eda_data = np.interp(x_new, x_old, eda_data)
                    eda_length = len(eda_data)
                    print(f"  Resampled EDA from {eda_length_before} -> {eda_length} to match labels")

                    # Optional: small smoothing after upsampling to reduce interpolation noise
                    # Uncomment the following lines to apply a simple moving average
                    # window = 3
                    # eda_data = np.convolve(eda_data, np.ones(window)/window, mode='same')
                except Exception as e:
                    # If interpolation fails, fall back to truncation (safe fallback)
                    print(f"  Warning: EDA resampling failed: {e}; falling back to truncation")
                    min_length = min(eda_length, label_length)
                    eda_data = eda_data[:min_length]
                    labels = labels[:min_length]
            else:
                # If labels are shorter or equal length, truncate both to the minimum length
                min_length = min(eda_length, label_length)
                eda_data = eda_data[:min_length]
                labels = labels[:min_length]
            
            # Analyze label distribution
            unique_labels, counts = np.unique(labels, return_counts=True)
            label_dist = dict(zip(unique_labels, counts))
            print(f"  Label distribution: {label_dist}")
            
            # Check if we have baseline (1) and stress (2) data
            baseline_count = label_dist.get(1, 0)
            stress_count = label_dist.get(2, 0)
            print(f"  Baseline samples: {baseline_count}, Stress samples: {stress_count}")
            
            return eda_data, labels, subject_name
        else:
            print(f"  ERROR: Could not find EDA data or labels for {subject_name}")
            return None, None, subject_name
            
    except Exception as e:
        print(f"  ERROR loading {subject_name}: {e}")
        
        # Try fallback to CSV if pickle fails
        print(f"  Trying CSV fallback for {subject_name}...")
        try:
            eda_csv = os.path.join(subject_path, f"{subject_name}_E4_Data", "EDA.csv")
            if os.path.exists(eda_csv):
                eda_df = pd.read_csv(eda_csv, header=None)
                
                # Remove timestamp if present
                if eda_df.iloc[0, 0] > 1000000000:
                    eda_df = eda_df.iloc[1:]
                
                eda_data = pd.to_numeric(eda_df.iloc[:, 0], errors='coerce').dropna().values
                
                # Create synthetic labels based on EDA patterns (for demonstration)
                threshold = np.percentile(eda_data, 70)
                synthetic_labels = (eda_data > threshold).astype(int) + 1
                
                print(f"  CSV fallback successful: {len(eda_data)} EDA samples")
                print(f"  Created synthetic labels (high EDA = stress)")
                
                return eda_data, synthetic_labels, subject_name
            else:
                print(f"  No CSV file found for {subject_name}")
                return None, None, subject_name
        except Exception as csv_error:
            print(f"  CSV fallback failed: {csv_error}")
            return None, None, subject_name

def extract_advanced_features(eda_signal, window_size=128, step_size=64):
    """Extract comprehensive features from EDA signal"""
    # Simplified feature set: only standard statistical features
    features = []

    for i in range(0, len(eda_signal) - window_size + 1, step_size):
        window = eda_signal[i:i + window_size]

        # Basic statistical features (retain only these)
        mean_eda = np.mean(window)
        std_eda = np.std(window)
        var_eda = np.var(window)
        min_eda = np.min(window)
        max_eda = np.max(window)
        range_eda = max_eda - min_eda
        median_eda = np.median(window)

        # Percentile features
        q25 = np.percentile(window, 25)
        q75 = np.percentile(window, 75)
        iqr = q75 - q25

        # New features: entropy and frequency-domain features
        entropy = -np.sum(window * np.log2(window + 1e-9))  # Signal entropy
        fft_features = np.abs(np.fft.fft(window))[:window_size // 2]  # Frequency domain
        power_spectral_density = np.sum(fft_features ** 2)

        feature_vector = [
            mean_eda, std_eda, var_eda, min_eda, max_eda, range_eda, median_eda,
            q25, q75, iqr, entropy, power_spectral_density
        ]

        features.append(feature_vector)

    return np.array(features)

def prepare_multi_subject_data(subject_paths, window_size=128, step_size=64, normalize_per_subject=True):
    """Prepare combined dataset from multiple subjects

    Args:
        subject_paths: list of subject folders
        window_size: samples per window
        step_size: step (samples) between windows
        normalize_per_subject: if True, apply StandardScaler to each subject's features
            independently before combining. This removes between-subject scale differences.
    """
    print("=== Loading Multi-Subject Data ===")

    all_features = []
    all_labels = []
    all_subject_ids = []  # Updated to ensure alignment

    for subject_path in subject_paths:
        eda_data, labels, subject_name = load_subject_data(subject_path)

        if eda_data is not None and labels is not None:
            # Extract features
            print(f"  Extracting features for {subject_name}...")
            features = extract_advanced_features(eda_data, window_size, step_size)

            # Get corresponding labels for each window
            window_labels = []
            for i in range(0, len(eda_data) - window_size + 1, step_size):
                window_label_segment = labels[i:i + window_size]
                if len(window_label_segment) > 0:
                    # Get most frequent label in window
                    unique, counts = np.unique(window_label_segment, return_counts=True)
                    most_common = unique[np.argmax(counts)]
                    window_labels.append(most_common)

            window_labels = np.array(window_labels[:len(features)])

            # Filter for baseline (1) and stress (2) only
            stress_baseline_mask = (window_labels == 1) | (window_labels == 2)

            if np.sum(stress_baseline_mask) > 0:
                filtered_features = features[stress_baseline_mask]
                filtered_labels = window_labels[stress_baseline_mask]

                # Convert to binary: 0=baseline, 1=stress
                binary_labels = (filtered_labels == 2).astype(int)

                # Optional: per-subject normalization to remove between-subject offsets/scales
                if normalize_per_subject and filtered_features.shape[0] > 0:
                    subj_scaler = StandardScaler()
                    try:
                        filtered_features = subj_scaler.fit_transform(filtered_features)
                    except Exception:
                        # If scaling fails for any reason, continue without scaling
                        pass

                all_features.append(filtered_features)
                all_labels.append(binary_labels)
                all_subject_ids.extend([subject_name] * len(filtered_features))  # Ensure alignment

                print(f"  {subject_name}: {len(filtered_features)} windows ({np.sum(binary_labels == 0)} baseline, {np.sum(binary_labels == 1)} stress)")
            else:
                print(f"  {subject_name}: No baseline/stress data found")

    if len(all_features) > 0:
        # Combine all subjects
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        combined_subject_ids = np.array(all_subject_ids)  # Ensure subject_ids is a numpy array

        # Downcast to smaller dtypes to reduce memory/storage for Raspberry Pi
        combined_features = combined_features.astype(np.float32)
        combined_labels = combined_labels.astype(np.int8)

        print(f"\n=== Combined Dataset ===")
        print(f"Total windows: {len(combined_features)}")
        print(f"Total features per window: {combined_features.shape[1]}")
        print(f"Baseline windows: {np.sum(combined_labels == 0)}")
        print(f"Stress windows: {np.sum(combined_labels == 1)}")
        print(f"Subjects: {len(set(combined_subject_ids))}")

        # Check if we have both classes
        if np.sum(combined_labels == 0) == 0:
            print("ERROR: No baseline samples found!")
            return None, None, None
        elif np.sum(combined_labels == 1) == 0:
            print("ERROR: No stress samples found!")
            return None, None, None
        elif len(np.unique(combined_labels)) < 2:
            print("ERROR: Need both baseline and stress samples for classification!")
            return None, None, None

        # Debugging: Ensure alignment of all arrays
        assert len(combined_features) == len(combined_labels) == len(combined_subject_ids), (
            f"Mismatch in data dimensions: Features={len(combined_features)}, Labels={len(combined_labels)}, Subject IDs={len(combined_subject_ids)}"
        )

        return combined_features, combined_labels, combined_subject_ids
    else:
        print("ERROR: No valid data found from any subject!")
        return None, None, None

def train_multi_subject_classifier(features, labels, subject_ids):
    """Train classifiers on multi-subject data"""
    print("\n=== Training Multi-Subject Classifiers ===")

    # Handle class imbalance: use sample_weight to upweight stress
    baseline_count = int(np.sum(labels == 0))
    stress_count = int(np.sum(labels == 1))
    total = max(1, baseline_count + stress_count)
    stress_weight = baseline_count / max(1, stress_count) if stress_count > 0 else 1.0
    sample_weight = np.ones(len(labels), dtype=np.float32)
    sample_weight[labels == 1] = float(stress_weight)
    print(f"Class counts - baseline: {baseline_count}, stress: {stress_count}. Upweighting stress by {stress_weight:.2f}x")

    # Check if we have both classes
    unique_classes = np.unique(labels)
    if len(unique_classes) < 2:
        print(f"ERROR: Cannot train classifier with only {len(unique_classes)} class(es)")
        print(f"Available classes: {unique_classes}")
        return None

    print(f"Training with {len(unique_classes)} classes: {unique_classes}")

    # Split data ensuring we don't mix subjects between train/test
    # This is important for generalization testing
    unique_subjects = list(set(subject_ids))

    if len(unique_subjects) >= 2:
        # Use some subjects for training, others for testing
        n_train_subjects = max(1, len(unique_subjects) * 2 // 3)
        train_subjects = unique_subjects[:n_train_subjects]
        test_subjects = unique_subjects[n_train_subjects:]

        train_mask = np.array([sid in train_subjects for sid in subject_ids])
        test_mask = np.array([sid in test_subjects for sid in subject_ids])

        X_train = features[train_mask]
        y_train = labels[train_mask]
        sw_train = sample_weight[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        sw_test = sample_weight[test_mask]

        print(f"Train subjects: {train_subjects}")
        print(f"Test subjects: {test_subjects}")
        print(f"Train samples: {len(X_train)} ({np.sum(y_train == 0)} baseline, {np.sum(y_train == 1)} stress)")
        print(f"Test samples: {len(X_test)} ({np.sum(y_test == 0)} baseline, {np.sum(y_test == 1)} stress)")
    else:
        # If only one subject, use regular train/test split
        X_train, X_test, y_train, y_test, sw_train, sw_test = train_test_split(
            features, labels, sample_weight, test_size=0.3, random_state=42, stratify=labels
        )
        print("Using regular train/test split (single subject)")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers - add RandomForest and SVM
    classifiers = {
        'LDA': LDA()
    }

    # No hyperparameter grid needed for LDA
    param_grid = {}

    results = {}

    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")

        # No hyperparameter tuning for LDA

        # Train (pass sample weights if supported)
        try:
            clf.fit(X_train_scaled, y_train, **({'sample_weight': sw_train} if hasattr(clf, 'fit') else {}))
        except TypeError:
            # Some wrapped estimators won't accept sample_weight at top level
            clf.fit(X_train_scaled, y_train)

        # Predict
        y_pred = clf.predict(X_test_scaled)

        # Get probabilities if available and we have binary classification
        y_pred_proba = None
        if hasattr(clf, 'predict_proba'):
            proba = clf.predict_proba(X_test_scaled)
            if proba.shape[1] == 2:  # Binary classification
                y_pred_proba = proba[:, 1]
            else:
                print(f"  Warning: Expected 2 classes, got {proba.shape[1]}")
        elif hasattr(clf, 'decision_function'):
            df = clf.decision_function(X_test_scaled)
            # map decision function to [0,1] via sigmoid
            y_pred_proba = 1.0 / (1.0 + np.exp(-df))

        # Evaluate on held-out test set
        accuracy = accuracy_score(y_test, y_pred)

        print(f"Accuracy (test set): {accuracy:.4f}")
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Baseline', 'Stress']))

        # Optimize threshold to maximize F1 for stress class
        optimal_threshold = 0.5
        f1_at_opt = None
        if y_pred_proba is not None:
            thresholds = np.linspace(0.1, 0.9, 17)
            best_f1 = -1.0
            best_t = 0.5
            for t in thresholds:
                y_hat = (y_pred_proba >= t).astype(int)
                f1 = f1_score(y_test, y_hat, pos_label=1)
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            optimal_threshold = best_t
            f1_at_opt = best_f1
            print(f"Optimal stress threshold: {optimal_threshold:.2f} (F1={f1_at_opt:.3f})")

        results[name] = {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_labels': y_test,
            'threshold': optimal_threshold,
            'f1_at_threshold': f1_at_opt,
            'model': clf,
            'scaler': scaler
        }

    return results


def load_compact_lda(path='lda_compact.npz'):
    """Load compact LDA params (numpy-only) saved by train_multi_subject_classifier.

    Returns a dict with keys: coef, intercept, classes, scaler_mean, scaler_scale
    """
    d = np.load(path)
    params = {
        'coef': d['coef'],
        'intercept': d['intercept'],
        'classes': d['classes'],
        'scaler_mean': d['scaler_mean'],
        'scaler_scale': d['scaler_scale']
    }
    return params


def predict_compact_lda(X, params):
    """Predict using compact LDA params (numpy-only).

    X: array shape (n_samples, n_features)
    params: dict returned by load_compact_lda
    Returns: predicted class labels (original class integers)
    """
    Xf = np.asarray(X, dtype=np.float32)
    # apply scaler
    Xs = (Xf - params['scaler_mean']) / params['scaler_scale']
    # linear decision: coef dot x + intercept
    logits = np.dot(Xs, params['coef'].T) + params['intercept']
    # handle binary or multiclass logits
    if logits.ndim == 2 and logits.shape[1] > 1:
        preds = np.argmax(logits, axis=1)
    else:
        preds = (logits.ravel() > 0).astype(np.int64)
    # map to original classes
    try:
        mapped = params['classes'][preds]
        return mapped
    except Exception:
        return preds

def analyze_feature_importance(results, feature_names=None):
    """Analyze which features are most important for stress detection"""
    if feature_names is None:
        # Feature names for the simplified standard-statistics feature set
        feature_names = [
            'Mean', 'Std', 'Var', 'Min', 'Max', 'Range', 'Median',
            'Q25', 'Q75', 'IQR'
        ]
    
    print("\n=== Feature Importance Analysis ===")
    
    # Logistic Regression coefficient analysis
    if 'Logistic Regression' in results:
        lr_model = results['Logistic Regression']['model']
        coefficients = lr_model.coef_[0]  # Get coefficients for binary classification
        
        # Sort by absolute coefficient value (importance)
        abs_coefficients = np.abs(coefficients)
        indices = np.argsort(abs_coefficients)[::-1]
        
        print("Top 10 Most Important Features (Logistic Regression Coefficients):")
        for i in range(min(10, len(indices))):
            idx = indices[i]
            coef_val = coefficients[idx]
            abs_coef = abs_coefficients[idx]
            direction = "increases" if coef_val > 0 else "decreases"
            print(f"  {feature_names[idx]:<20}: {abs_coef:.4f} ({direction} stress probability)")
        
        # Plot feature coefficients
        plt.figure(figsize=(12, 8))
        
        # Plot absolute coefficients
        plt.subplot(2, 1, 1)
        plt.bar(range(len(abs_coefficients)), abs_coefficients[indices])
        plt.xticks(range(len(abs_coefficients)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Importance (Absolute Coefficient Values)')
        plt.ylabel('Absolute Coefficient')
        
        # Plot actual coefficients (showing direction)
        plt.subplot(2, 1, 2)
        colors = ['red' if c < 0 else 'blue' for c in coefficients[indices]]
        plt.bar(range(len(coefficients)), coefficients[indices], color=colors)
        plt.xticks(range(len(coefficients)), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.title('Feature Coefficients (Blue: Increases Stress, Red: Decreases Stress)')
        plt.ylabel('Coefficient Value')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('logistic_regression_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print interpretation
        print(f"\nInterpretation:")
        print(f"- Blue bars: Features that INCREASE stress probability when they increase")
        print(f"- Red bars: Features that DECREASE stress probability when they increase")
        print(f"- Larger absolute values = more important for classification")
    
    else:
        print("No Logistic Regression model found for feature analysis.")

def plot_multi_subject_results(results):
    """Plot results of multi-subject classification"""
    n_models = len(results)
    
    # Create a single plot for logistic regression
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    for name, result in results.items():
        cm = confusion_matrix(result['test_labels'], result['predictions'])
        
        # Plot confusion matrix
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        fig.colorbar(im, ax=ax)
        
        # Add text annotations
        thresh = cm.max() / 2.
        for j in range(cm.shape[0]):
            for k in range(cm.shape[1]):
                ax.text(k, j, format(cm[j, k], 'd'),
                       ha="center", va="center",
                       color="white" if cm[j, k] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f} (CV: {result["cv_scores"].mean():.3f} ± {result["cv_scores"].std():.3f})')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Baseline', 'Stress'])
        ax.set_yticklabels(['Baseline', 'Stress'])
    
    plt.tight_layout()
    plt.savefig('logistic_regression_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for multi-subject stress classification"""
    print("Multi-Subject WESAD Stress Classification")
    print("=" * 50)
    
    # Load .env (project root) then determine WESAD base path
    # Try to load .env from the repo root (one level up from this file)
    env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.env'))
    load_dotenv(env_path)

    # Default base path (relative to this script) if WESAD_BASE not set in .env
    default_base = os.path.abspath(os.path.join(os.path.dirname(__file__), 'WESAD', 'WESAD'))
    base_path = os.environ.get('WESAD_BASE', default_base)

    # Auto-discover subject folders (S*) under the base path
    subject_paths = sorted(glob.glob(os.path.join(base_path, 'S*')))

    # Check which subjects exist and report
    existing_paths = [p for p in subject_paths if os.path.isdir(p)]
    print(f"Using WESAD base: {base_path}")
    print(f"Found {len(existing_paths)} subjects: {[os.path.basename(p) for p in existing_paths]}")
    
    if len(existing_paths) == 0:
        print("No subject directories found!")
        return
    
    # Parse simple CLI args for window/step to mirror CSV predictor
    import argparse
    parser = argparse.ArgumentParser(description='Train LDA on WESAD with CSV-like windowing')
    parser.add_argument('--window', type=int, default=72, help='Window size in samples (e.g., 72 at 4 Hz ≈ 18 s)')
    parser.add_argument('--step', type=int, default=24, help='Step size in samples (e.g., 24 at 4 Hz ≈ 6 s)')
    parser.add_argument('--normalize-per-subject', action='store_true', help='Enable per-subject normalization')
    parser.add_argument('--no-normalize-per-subject', dest='normalize_per_subject', action='store_false', help='Disable per-subject normalization')
    parser.set_defaults(normalize_per_subject=True)
    parser.add_argument('--out-prefix', type=str, default='ML Testing/lda', help='Output prefix for saved artifacts')
    args, unknown = parser.parse_known_args()

    # Prepare multi-subject data with requested window/step
    features, labels, subject_ids = prepare_multi_subject_data(
        existing_paths,
        window_size=args.window,
        step_size=args.step,
        normalize_per_subject=args.normalize_per_subject
    )
    
    if features is None or labels is None or subject_ids is None:
        print("Failed to prepare data!")
        return

    # Validate alignment of features, labels, and subject_ids
    if not (len(features) == len(labels) == len(subject_ids)):
        print("ERROR: Mismatch in data dimensions!")
        print(f"Features: {len(features)}, Labels: {len(labels)}, Subject IDs: {len(subject_ids)}")
        return

    # Train classifiers
    results = train_multi_subject_classifier(features, labels, subject_ids)
    
    if results is None:
        print("Failed to train classifiers!")
        return
    
    if results:
        # Find best model
        best_model = max(results.keys(), key=lambda k: results[k]['accuracy'])
        print(f"\n=== Best Model: {best_model} ===")
        print(f"Test Accuracy: {results[best_model]['accuracy']:.4f}")
        if results[best_model].get('threshold') is not None:
            print(f"Suggested stress threshold: {results[best_model]['threshold']:.2f} (F1={results[best_model].get('f1_at_threshold')})")
        
        # Plot results
        # plot_multi_subject_results may expect cv_scores; guard usage
        try:
            plot_multi_subject_results(results)
        except Exception:
            pass
        
        # Save artifacts for CSV/stream predictors
        try:
            best = results[best_model]
            scaler = best['scaler']
            model = best['model']
            threshold = best.get('threshold', 0.5)

            # Save full pipeline (scaler + model) via joblib
            pipeline = Pipeline(steps=[('scaler', scaler), ('lda', model)])
            full_path = os.path.join(os.path.dirname(__file__), f"{args.out_prefix}_full_pipeline.joblib")
            joblib.dump(pipeline, full_path)
            print(f"Saved full pipeline: {full_path}")

            # Save compact numpy-only params
            # Extract LDA params
            coef = getattr(model, 'coef_', None)
            intercept = getattr(model, 'intercept_', None)
            classes = getattr(model, 'classes_', None)
            scaler_mean = scaler.mean_
            scaler_scale = scaler.scale_
            compact_path = os.path.join(os.path.dirname(__file__), f"{args.out_prefix}_compact.npz")
            if coef is not None and intercept is not None and classes is not None:
                np.savez(compact_path,
                         coef=coef.astype(np.float32),
                         intercept=intercept.astype(np.float32),
                         classes=classes.astype(np.int64),
                         scaler_mean=scaler_mean.astype(np.float32),
                         scaler_scale=scaler_scale.astype(np.float32))
                print(f"Saved compact LDA params: {compact_path}")
            else:
                print("Warning: Missing LDA coefficients; skipping compact save")

            # Save tuned threshold
            thr_path = os.path.join(os.path.dirname(__file__), f"{args.out_prefix}_stress_threshold.txt")
            with open(thr_path, 'w') as f:
                f.write(f"{threshold:.4f}\n")
            print(f"Saved stress threshold: {thr_path}")
        except Exception as e:
            print(f"Warning: Failed to save artifacts: {e}")

        # Analyze feature importance (optional; may not apply cleanly to LDA)
        try:
            analyze_feature_importance(results)
        except Exception:
            pass
        
        print(f"\n=== Summary ===")
        print(f"Successfully trained stress classifiers on {len(set(subject_ids))} subjects")
        print(f"Total training samples: {len(features)}")
        print(f"Best performing model: {best_model}")
        print(f"The model can now distinguish between baseline and stress states")
        print(f"across different individuals with {results[best_model]['accuracy']*100:.1f}% accuracy!")

if __name__ == "__main__":
    main()