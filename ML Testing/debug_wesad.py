import pickle
import numpy as np
import os

def debug_wesad_labels(subject_path):
    """Debug the label structure in WESAD data"""
    subject_name = os.path.basename(subject_path)
    pkl_file = os.path.join(subject_path, f"{subject_name}.pkl")
    
    print(f"\n=== Debugging {subject_name} ===")
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        print("Main keys:", list(data.keys()))
        
        # Look at label structure in detail
        for key in data.keys():
            if 'label' in key.lower():
                labels = data[key]
                print(f"\nLabel key: '{key}'")
                print(f"Label type: {type(labels)}")
                print(f"Label shape: {labels.shape if hasattr(labels, 'shape') else 'No shape'}")
                
                if hasattr(labels, 'flatten'):
                    flat_labels = labels.flatten()
                    unique, counts = np.unique(flat_labels, return_counts=True)
                    print(f"Unique labels and counts:")
                    for label, count in zip(unique, counts):
                        print(f"  {label}: {count} ({count/len(flat_labels)*100:.1f}%)")
                    
                    # Check first 100 labels to see pattern
                    print(f"First 20 labels: {flat_labels[:20]}")
                    print(f"Last 20 labels: {flat_labels[-20:]}")
        
        # Look at signal structure
        if 'signal' in data:
            signal_data = data['signal']
            print(f"\nSignal keys: {list(signal_data.keys())}")
            
            for device in signal_data.keys():
                print(f"\n{device} signals:")
                if isinstance(signal_data[device], dict):
                    for sensor, sensor_data in signal_data[device].items():
                        if hasattr(sensor_data, 'shape'):
                            print(f"  {sensor}: {sensor_data.shape}")
                        else:
                            print(f"  {sensor}: {type(sensor_data)}")
        
        return data
        
    except Exception as e:
        print(f"Error loading {subject_name}: {e}")
        return None

def main():
    base_path = r"c:\Notre Dame\Machine Learning for Embedded Systems\ML Testing\WESAD\WESAD"
    subjects = ["S10", "S11", "S13", "S14"]
    
    for subject in subjects:
        subject_path = os.path.join(base_path, subject)
        if os.path.exists(subject_path):
            debug_wesad_labels(subject_path)
        else:
            print(f"{subject} not found")

if __name__ == "__main__":
    main()