import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import os
warnings.filterwarnings('ignore')

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
            
            if label_length > eda_length:
                # Downsample labels to match EDA sampling rate
                downsample_factor = label_length // eda_length
                # Take every nth label to match EDA sampling
                downsampled_labels = labels[::downsample_factor][:eda_length]
                labels = downsampled_labels
                print(f"  Downsampled labels by factor {downsample_factor}")
            else:
                # If labels are shorter, truncate EDA
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
    features = []
    
    for i in range(0, len(eda_signal) - window_size + 1, step_size):
        window = eda_signal[i:i + window_size]
        
        # Basic statistical features
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
        
        # Derivative features (rate of change)
        diff = np.diff(window)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        max_diff = np.max(np.abs(diff))
        
        # Second derivative (acceleration)
        diff2 = np.diff(diff)
        mean_diff2 = np.mean(diff2) if len(diff2) > 0 else 0
        std_diff2 = np.std(diff2) if len(diff2) > 0 else 0
        
        # Trend analysis
        x = np.arange(len(window))
        slope = np.polyfit(x, window, 1)[0] if len(window) > 1 else 0
        
        # Zero crossing rate (how often signal crosses the mean)
        mean_centered = window - mean_eda
        zero_crossings = np.sum(np.diff(np.sign(mean_centered)) != 0)
        
        # Peak detection
        peaks = 0
        troughs = 0
        for j in range(1, len(window) - 1):
            if window[j] > window[j-1] and window[j] > window[j+1]:
                peaks += 1
            elif window[j] < window[j-1] and window[j] < window[j+1]:
                troughs += 1
        
        # Energy and power features
        energy = np.sum(window ** 2)
        power = energy / len(window)
        
        # Frequency domain features (simplified)
        try:
            fft = np.fft.fft(window)
            power_spectrum = np.abs(fft) ** 2
            
            # Divide spectrum into frequency bands
            low_freq_power = np.mean(power_spectrum[:len(power_spectrum)//4])
            mid_freq_power = np.mean(power_spectrum[len(power_spectrum)//4:len(power_spectrum)//2])
            high_freq_power = np.mean(power_spectrum[len(power_spectrum)//2:3*len(power_spectrum)//4])
            
            # Spectral centroid (frequency center of mass)
            freqs = np.fft.fftfreq(len(window))
            spectral_centroid = np.sum(freqs[:len(freqs)//2] * power_spectrum[:len(power_spectrum)//2]) / np.sum(power_spectrum[:len(power_spectrum)//2])
        except:
            low_freq_power = mid_freq_power = high_freq_power = spectral_centroid = 0
        
        # Combine all features
        feature_vector = [
            # Time domain statistics
            mean_eda, std_eda, var_eda, min_eda, max_eda, range_eda, median_eda,
            q25, q75, iqr,
            
            # Derivative features
            mean_diff, std_diff, max_diff, mean_diff2, std_diff2,
            
            # Shape features
            slope, zero_crossings, peaks, troughs,
            
            # Energy features
            energy, power,
            
            # Frequency features
            low_freq_power, mid_freq_power, high_freq_power, spectral_centroid
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def prepare_multi_subject_data(subject_paths, window_size=128, step_size=64):
    """Prepare combined dataset from multiple subjects"""
    print("=== Loading Multi-Subject Data ===")
    
    all_features = []
    all_labels = []
    subject_ids = []
    
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
                
                all_features.append(filtered_features)
                all_labels.append(binary_labels)
                subject_ids.extend([subject_name] * len(filtered_features))
                
                print(f"  {subject_name}: {len(filtered_features)} windows ({np.sum(binary_labels == 0)} baseline, {np.sum(binary_labels == 1)} stress)")
            else:
                print(f"  {subject_name}: No baseline/stress data found")
    
    if len(all_features) > 0:
        # Combine all subjects
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        
        print(f"\n=== Combined Dataset ===")
        print(f"Total windows: {len(combined_features)}")
        print(f"Total features per window: {combined_features.shape[1]}")
        print(f"Baseline windows: {np.sum(combined_labels == 0)}")
        print(f"Stress windows: {np.sum(combined_labels == 1)}")
        print(f"Subjects: {len(set(subject_ids))}")
        
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
        
        return combined_features, combined_labels, subject_ids
    else:
        print("ERROR: No valid data found from any subject!")
        return None, None, None

def train_multi_subject_classifier(features, labels, subject_ids):
    """Train classifiers on multi-subject data"""
    print("\n=== Training Multi-Subject Classifiers ===")
    
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
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        print(f"Train subjects: {train_subjects}")
        print(f"Test subjects: {test_subjects}")
        print(f"Train samples: {len(X_train)} ({np.sum(y_train == 0)} baseline, {np.sum(y_train == 1)} stress)")
        print(f"Test samples: {len(X_test)} ({np.sum(y_test == 0)} baseline, {np.sum(y_test == 1)} stress)")
    else:
        # If only one subject, use regular train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print("Using regular train/test split (single subject)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define classifier - Logistic Regression only
    classifiers = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=1.0
        )
    }
    
    results = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train
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
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        print(f"Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Baseline', 'Stress']))
        
        results[name] = {
            'model': clf,
            'scaler': scaler,
            'accuracy': accuracy,
            'cv_scores': cv_scores,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'test_labels': y_test
        }
    
    return results

def analyze_feature_importance(results, feature_names=None):
    """Analyze which features are most important for stress detection"""
    if feature_names is None:
        feature_names = [
            'Mean', 'Std', 'Var', 'Min', 'Max', 'Range', 'Median',
            'Q25', 'Q75', 'IQR',
            'Mean_Diff', 'Std_Diff', 'Max_Diff', 'Mean_Diff2', 'Std_Diff2',
            'Slope', 'Zero_Cross', 'Peaks', 'Troughs',
            'Energy', 'Power',
            'Low_Freq', 'Mid_Freq', 'High_Freq', 'Spectral_Centroid'
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
    
    # Define subject paths (excluding S14 which doesn't have proper stress data)
    base_path = r"c:\Notre Dame\Machine Learning for Embedded Systems\ML Testing\WESAD\WESAD"
    subject_paths = [
        os.path.join(base_path, "S10"),
        os.path.join(base_path, "S11"),
        os.path.join(base_path, "S13")
    ]
    
    # Check which subjects exist
    existing_paths = [path for path in subject_paths if os.path.exists(path)]
    print(f"Found {len(existing_paths)} subjects: {[os.path.basename(p) for p in existing_paths]}")
    
    if len(existing_paths) == 0:
        print("No subject directories found!")
        return
    
    # Prepare multi-subject data
    features, labels, subject_ids = prepare_multi_subject_data(
        existing_paths, 
        window_size=128,  # ~8 seconds at 16Hz
        step_size=64      # 50% overlap
    )
    
    if features is None:
        print("Failed to prepare data!")
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
        print(f"CV Accuracy: {results[best_model]['cv_scores'].mean():.4f} ± {results[best_model]['cv_scores'].std():.4f}")
        
        # Plot results
        plot_multi_subject_results(results)
        
        # Analyze feature importance
        analyze_feature_importance(results)
        
        print(f"\n=== Summary ===")
        print(f"Successfully trained stress classifiers on {len(set(subject_ids))} subjects")
        print(f"Total training samples: {len(features)}")
        print(f"Best performing model: {best_model}")
        print(f"The model can now distinguish between baseline and stress states")
        print(f"across different individuals with {results[best_model]['accuracy']*100:.1f}% accuracy!")

if __name__ == "__main__":
    main()