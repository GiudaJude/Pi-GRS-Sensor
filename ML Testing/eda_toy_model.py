
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR
import warnings
warnings.filterwarnings('ignore')

def load_eda_data(file_path):
    """Load EDA data from CSV file"""
    data = pd.read_csv(file_path, header=None, names=['eda'])
    
    # The first row appears to be a timestamp, let's handle it
    if data.iloc[0]['eda'] > 1000000000:  # Likely a timestamp
        timestamp = data.iloc[0]['eda']
        data = data.iloc[1:]  # Remove timestamp row
        data['eda'] = pd.to_numeric(data['eda'], errors='coerce')
        data = data.dropna()
        print(f"Timestamp: {timestamp}")
    
    print(f"Data shape: {data.shape}")
    print(f"EDA range: {data['eda'].min():.6f} to {data['eda'].max():.6f}")
    return data

def create_features(data, window_size=10):
    """Create features for pattern learning"""
    features = []
    targets = []
    
    eda_values = data['eda'].values
    
    for i in range(window_size, len(eda_values)):
        # Use previous window_size values as features
        feature_window = eda_values[i-window_size:i]
        target = eda_values[i]
        
        # Create statistical features from the window
        window_features = [
            np.mean(feature_window),           # Mean
            np.std(feature_window),            # Standard deviation
            np.max(feature_window),            # Maximum
            np.min(feature_window),            # Minimum
            np.median(feature_window),         # Median
            feature_window[-1] - feature_window[0],  # Change from start to end
            np.sum(np.diff(feature_window) > 0),     # Number of increases
            np.sum(np.diff(feature_window) < 0),     # Number of decreases
        ]
        
        # Add the raw window values as well
        combined_features = list(feature_window) + window_features
        
        features.append(combined_features)
        targets.append(target)
    
    return np.array(features), np.array(targets)

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance"""
    models = {
        'Linear Regression': LinearRegression()
    }
    
    results = {}
    
    # Scale the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'scaler': scaler,
            'mse': mse,
            'r2': r2,
            'predictions': y_pred
        }
        
        print(f"MSE: {mse:.6f}")
        print(f"R²: {r2:.6f}")
    
    return results

def plot_results(y_test, results, sample_size=500):
    """Plot actual vs predicted values for visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('EDA Pattern Learning Results', fontsize=16)
    
    # Plot actual vs predicted for each model
    for i, (name, result) in enumerate(results.items()):
        row = i // 2
        col = i % 2
        
        if i < 3:  # Only plot first 3 models
            ax = axes[row, col]
            
            # Sample data for visualization if too large
            if len(y_test) > sample_size:
                indices = np.random.choice(len(y_test), sample_size, replace=False)
                y_test_sample = y_test[indices]
                y_pred_sample = result['predictions'][indices]
            else:
                y_test_sample = y_test
                y_pred_sample = result['predictions']
            
            ax.scatter(y_test_sample, y_pred_sample, alpha=0.5, s=1)
            ax.plot([y_test_sample.min(), y_test_sample.max()], 
                   [y_test_sample.min(), y_test_sample.max()], 'r--', lw=2)
            ax.set_xlabel('Actual EDA')
            ax.set_ylabel('Predicted EDA')
            ax.set_title(f'{name}\nR² = {result["r2"]:.4f}')
            ax.grid(True, alpha=0.3)
    
    # Plot time series comparison for best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    ax = axes[1, 1]
    
    # Show a small section of the time series
    start_idx = 0
    end_idx = min(200, len(y_test))
    
    ax.plot(range(start_idx, end_idx), y_test[start_idx:end_idx], 
            label='Actual', linewidth=1)
    ax.plot(range(start_idx, end_idx), 
            results[best_model_name]['predictions'][start_idx:end_idx], 
            label='Predicted', linewidth=1)
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('EDA Value')
    ax.set_title(f'Time Series: {best_model_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('eda_model_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_importance(results):
    """Analyze feature importance for Random Forest model"""
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        feature_names = [f'EDA_lag_{i+1}' for i in range(10)] + [
            'Mean', 'Std', 'Max', 'Min', 'Median', 'Change', 'Increases', 'Decreases'
        ]
        
        importance = rf_model.feature_importances_
        
        # Sort features by importance
        indices = np.argsort(importance)[::-1]
        
        print("\nFeature Importance (Random Forest):")
        print("-" * 40)
        for i in range(min(10, len(indices))):
            idx = indices[i]
            print(f"{feature_names[idx]:<15}: {importance[idx]:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.bar(range(len(importance)), importance[indices])
        plt.xticks(range(len(importance)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importance in EDA Pattern Learning')
        plt.ylabel('Importance')
        plt.tight_layout()
        plt.savefig('eda_feature_importance.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    # Load the EDA data
    file_path = r"c:\Notre Dame\Machine Learning for Embedded Systems\ML Testing\WESAD\WESAD\S10\S10_E4_Data\EDA.csv"
    
    print("Loading EDA data...")
    data = load_eda_data(file_path)
    
    # Create features for pattern learning
    print("\nCreating features...")
    window_size = 10  # Use 10 previous values to predict next value
    X, y = create_features(data, window_size)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False  # Don't shuffle to maintain temporal order
    )
    
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['r2'])
    print(f"\nBest performing model: {best_model_name}")
    print(f"Best R² score: {results[best_model_name]['r2']:.6f}")
    
    # Plot results
    plot_results(y_test, results)
    
    # Analyze feature importance
    analyze_feature_importance(results)
    
    # Show basic statistics about the learned patterns
    print("\nPattern Learning Summary:")
    print("-" * 50)
    print(f"Successfully learned patterns from {len(data)} EDA measurements")
    print(f"Using {window_size} previous values to predict next value")
    print(f"Created {len(X)} training examples with {X.shape[1]} features each")
    
    for name, result in results.items():
        print(f"\n{name}:")
        print(f"  - Mean Squared Error: {result['mse']:.6f}")
        print(f"  - R² Score: {result['r2']:.6f}")
        print(f"  - Explained Variance: {result['r2']*100:.2f}%")

if __name__ == "__main__":
    main()