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
    """Load EDA data and labels for a single subject"""
    subject_name = os.path.basename(subject_path)
    pkl_file = os.path.join(subject_path, f"{subject_name}.pkl")
    
    print(f"Loading {subject_name}...")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        # Extract wrist EDA data (more convenient than chest)
        eda_data = data['signal']['wrist']['EDA'].flatten()
        labels = data['label'].flatten()
        
        # Handle sampling rate mismatch - downsample labels to match EDA
        label_length = len(labels)
        eda_length = len(eda_data)
        
        if label_length > eda_length:
            downsample_factor = label_length // eda_length
            downsampled_labels = labels[::downsample_factor][:eda_length]
            labels = downsampled_labels
        
        print(f"  EDA samples: {len(eda_data)}")
        print(f"  Labels after downsampling: {len(labels)}")
        
        # Check label distribution
        unique_labels, counts = np.unique(labels, return_counts=True)
        baseline_count = counts[unique_labels == 1][0] if 1 in unique_labels else 0
        stress_count = counts[unique_labels == 2][0] if 2 in unique_labels else 0
        
        print(f"  Baseline samples: {baseline_count}")
        print(f"  Stress samples: {stress_count}")
        
        if baseline_count > 0 and stress_count > 0:
            return eda_data, labels, subject_name
        else:
            print(f"  WARNING: {subject_name} missing baseline or stress data")
            return None, None, subject_name
            
    except Exception as e:
        print(f"  ERROR: Could not load {subject_name}: {e}")
        return None, None, subject_name

def extract_eda_features(eda_signal, window_size=64, step_size=32):
    """Extract features from EDA signal using sliding windows"""
    features = []
    
    for i in range(0, len(eda_signal) - window_size + 1, step_size):
        window = eda_signal[i:i + window_size]
        
        # Statistical features
        mean_eda = np.mean(window)
        std_eda = np.std(window)
        min_eda = np.min(window)
        max_eda = np.max(window)
        range_eda = max_eda - min_eda
        
        # Change features
        diff = np.diff(window)
        mean_diff = np.mean(diff)
        std_diff = np.std(diff)
        
        # Trend
        slope = np.polyfit(range(len(window)), window, 1)[0] if len(window) > 1 else 0
        
        # Combine features
        feature_vector = [
            mean_eda, std_eda, min_eda, max_eda, range_eda,
            mean_diff, std_diff, slope
        ]
        
        features.append(feature_vector)
    
    return np.array(features)

def prepare_classification_data(subjects_data, window_size=64, step_size=32):
    """Prepare combined dataset for stress classification"""
    all_features = []
    all_labels = []
    subject_ids = []
    
    print("\n=== Preparing Classification Data ===")
    
    for eda_data, labels, subject_name in subjects_data:
        if eda_data is not None:
            # Extract features
            features = extract_eda_features(eda_data, window_size, step_size)
            
            # Get corresponding labels for each window
            window_labels = []
            for i in range(0, len(eda_data) - window_size + 1, step_size):
                window_segment = labels[i:i + window_size]
                # Use most common label in window
                unique, counts = np.unique(window_segment, return_counts=True)
                most_common = unique[np.argmax(counts)]
                window_labels.append(most_common)
            
            window_labels = np.array(window_labels[:len(features)])
            
            # Filter for baseline (1) and stress (2) only
            baseline_stress_mask = (window_labels == 1) | (window_labels == 2)
            
            if np.sum(baseline_stress_mask) > 0:
                filtered_features = features[baseline_stress_mask]
                filtered_labels = window_labels[baseline_stress_mask]
                
                # Convert to binary: 0=baseline, 1=stress
                binary_labels = (filtered_labels == 2).astype(int)
                
                all_features.append(filtered_features)
                all_labels.append(binary_labels)
                subject_ids.extend([subject_name] * len(filtered_features))
                
                baseline_windows = np.sum(binary_labels == 0)
                stress_windows = np.sum(binary_labels == 1)
                print(f"{subject_name}: {len(filtered_features)} windows ({baseline_windows} baseline, {stress_windows} stress)")
    
    if len(all_features) > 0:
        combined_features = np.vstack(all_features)
        combined_labels = np.hstack(all_labels)
        
        print(f"\nCombined Dataset:")
        print(f"  Total windows: {len(combined_features)}")
        print(f"  Features per window: {combined_features.shape[1]}")
        print(f"  Baseline windows: {np.sum(combined_labels == 0)}")
        print(f"  Stress windows: {np.sum(combined_labels == 1)}")
        print(f"  Subjects: {len(set(subject_ids))}")
        
        return combined_features, combined_labels, subject_ids
    else:
        return None, None, None

def train_stress_classifier(features, labels, subject_ids):
    """Train logistic regression for stress classification"""
    print("\n=== Training Stress Classifier ===")
    
    # Use leave-one-subject-out approach for robust evaluation
    unique_subjects = list(set(subject_ids))
    print(f"Subjects: {unique_subjects}")
    
    if len(unique_subjects) >= 2:
        # Use 2 subjects for training, 1 for testing
        test_subject = unique_subjects[0]  # Use first subject as test
        train_subjects = unique_subjects[1:]
        
        train_mask = np.array([sid in train_subjects for sid in subject_ids])
        test_mask = np.array([sid == test_subject for sid in subject_ids])
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        print(f"Training on: {train_subjects}")
        print(f"Testing on: {test_subject}")
        print(f"Train samples: {len(X_train)} ({np.sum(y_train == 0)} baseline, {np.sum(y_train == 1)} stress)")
        print(f"Test samples: {len(X_test)} ({np.sum(y_test == 0)} baseline, {np.sum(y_test == 1)} stress)")
    else:
        # Fallback to regular split
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.3, random_state=42, stratify=labels
        )
        print("Using regular train/test split")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Cross-validation on training set
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    print(f"\nResults:")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Baseline', 'Stress']))
    
    # Feature importance (coefficients)
    feature_names = ['Mean', 'Std', 'Min', 'Max', 'Range', 'Mean_Diff', 'Std_Diff', 'Slope']
    coefficients = clf.coef_[0]
    
    print(f"\nFeature Importance (Logistic Regression Coefficients):")
    for name, coef in zip(feature_names, coefficients):
        direction = "↑ stress" if coef > 0 else "↓ stress"
        print(f"  {name:<12}: {coef:>8.4f} ({direction})")
    
    return {
        'model': clf,
        'scaler': scaler,
        'accuracy': accuracy,
        'cv_scores': cv_scores,
        'test_labels': y_test,
        'predictions': y_pred,
        'coefficients': coefficients,
        'feature_names': feature_names
    }

def plot_results(results):
    """Plot confusion matrix and feature importance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confusion matrix
    cm = confusion_matrix(results['test_labels'], results['predictions'])
    im1 = ax1.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax1.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    ax1.set_title(f'Confusion Matrix\nAccuracy: {results["accuracy"]:.3f}')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['Baseline', 'Stress'])
    ax1.set_yticklabels(['Baseline', 'Stress'])
    
    # Feature importance
    coefficients = results['coefficients']
    feature_names = results['feature_names']
    colors = ['red' if c < 0 else 'blue' for c in coefficients]
    
    bars = ax2.bar(range(len(coefficients)), coefficients, color=colors)
    ax2.set_xlabel('Features')
    ax2.set_ylabel('Coefficient Value')
    ax2.set_title('Feature Importance\n(Blue: ↑Stress, Red: ↓Stress)')
    ax2.set_xticks(range(len(feature_names)))
    ax2.set_xticklabels(feature_names, rotation=45, ha='right')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('stress_classification_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Main function for stress classification"""
    print("WESAD Stress Classification with Logistic Regression")
    print("=" * 55)
    
    # Load data from subjects S10, S11, S13 (excluding S14)
    subjects = ["S10", "S11", "S13"]
    base_path = r"c:\Notre Dame\Machine Learning for Embedded Systems\ML Testing\WESAD\WESAD"
    
    subjects_data = []
    for subject in subjects:
        subject_path = os.path.join(base_path, subject)
        if os.path.exists(subject_path):
            eda_data, labels, subject_name = load_subject_data(subject_path)
            subjects_data.append((eda_data, labels, subject_name))
        else:
            print(f"Subject {subject} not found")
    
    # Filter out subjects with no data
    valid_subjects = [(eda, labels, name) for eda, labels, name in subjects_data if eda is not None]
    
    if len(valid_subjects) == 0:
        print("No valid subjects found!")
        return
    
    print(f"\nUsing {len(valid_subjects)} subjects for training")
    
    # Prepare classification data
    features, labels, subject_ids = prepare_classification_data(valid_subjects)
    
    if features is None:
        print("Failed to prepare classification data!")
        return
    
    # Train classifier
    results = train_stress_classifier(features, labels, subject_ids)
    
    # Plot results
    plot_results(results)
    
    print(f"\n=== Summary ===")
    print(f"Successfully trained stress classifier on {len(set(subject_ids))} subjects")
    print(f"Model can distinguish baseline vs stress with {results['accuracy']*100:.1f}% accuracy")
    print(f"Cross-validation shows {results['cv_scores'].mean()*100:.1f}% ± {results['cv_scores'].std()*100:.1f}% reliability")

if __name__ == "__main__":
    main()