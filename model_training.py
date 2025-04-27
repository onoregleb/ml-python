import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import os

def generate_sample_data(n_samples=1000):
    """Generate sample data for investment returns prediction"""
    np.random.seed(42)
    
    # Generate features
    feature_0 = np.random.normal(0, 1, n_samples)  # Market volatility
    feature_1 = np.random.normal(0, 1, n_samples)  # Company growth
    feature_2 = np.random.normal(0, 1, n_samples)  # Industry trend
    feature_3 = np.random.normal(0, 1, n_samples)  # Economic indicators
    feature_4 = np.random.normal(0, 1, n_samples)  # Risk factors
    
    # Generate target (investment returns)
    target = (0.5 * feature_0 + 0.3 * feature_1 + 0.2 * feature_2 + 
              0.1 * feature_3 - 0.2 * feature_4 + np.random.normal(0, 0.1, n_samples))
    
    # Create DataFrame
    data = pd.DataFrame({
        'feature_0': feature_0,
        'feature_1': feature_1,
        'feature_2': feature_2,
        'feature_3': feature_3,
        'feature_4': feature_4,
        'target': target
    })
    
    return data

def train_models(data):
    """Train three different models on the data"""
    X = data.drop('target', axis=1)
    y = data['target']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train models
    models = {
        'linear_regression': LinearRegression(),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'return_prediction': RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_scaled, y)
        trained_models[name] = model
    
    # Save models and scaler
    os.makedirs('models', exist_ok=True)
    for name, model in trained_models.items():
        joblib.dump(model, f'models/{name}.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    return trained_models, scaler

def generate_sample_csv():
    """Generate a sample CSV file for users to download"""
    data = generate_sample_data(100)
    data.to_csv('sample_data.csv', index=False)
    return 'sample_data.csv'

if __name__ == "__main__":
    # Generate and save sample data
    data = generate_sample_data()
    data.to_csv('sample_data.csv', index=False)
    
    # Train and save models
    train_models(data)
    
    print("Models trained and saved successfully!")
    print("Sample data saved to 'sample_data.csv'") 