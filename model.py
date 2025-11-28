import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# ============================================================================
# TRAINING SCRIPT - RUN THIS LOCALLY WITH YOUR DATASET
# ============================================================================

def load_and_preprocess_data(file_path):
    """Load and preprocess the house data"""
    
    # Read CSV file
    df = pd.read_csv(file_path, on_bad_lines='skip')

    # Select relevant features
    feature_columns = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
        'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ]

    target_column = 'price'

    # Create model dataset
    df_model = df[feature_columns + [target_column]].copy()
    df_model = df_model.fillna(0)

    # Feature engineering
    df_model['house_age'] = 2024 - df_model['yr_built']
    df_model['renovated'] = (df_model['yr_renovated'] > 0).astype(int)
    df_model['years_since_renovation'] = 2024 - df_model['yr_renovated']
    df_model.loc[df_model['yr_renovated'] == 0, 'years_since_renovation'] = df_model['house_age']
    
    df_model['total_sqft'] = df_model['sqft_living'] + df_model['sqft_lot']
    df_model['sqft_ratio'] = df_model['sqft_living'] / (df_model['sqft_lot'] + 1)
    df_model['bath_bed_ratio'] = df_model['bathrooms'] / (df_model['bedrooms'] + 1)

    return df_model


def train_and_save_model(df_model, model_path='trained_model.pkl'):
    """Train the machine learning model and save it"""
    
    print("Starting model training...")
    
    # Separate features and target
    X = df_model.drop('price', axis=1)
    y = df_model['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("Training Random Forest...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    
    # Train Gradient Boosting model
    print("Training Gradient Boosting...")
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb_model.fit(X_train_scaled, y_train)
    
    # Predictions
    rf_pred = rf_model.predict(X_test_scaled)
    gb_pred = gb_model.predict(X_test_scaled)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    # Calculate metrics
    metrics = {
        'rf_r2': r2_score(y_test, rf_pred),
        'rf_rmse': np.sqrt(mean_squared_error(y_test, rf_pred)),
        'rf_mae': mean_absolute_error(y_test, rf_pred),
        'gb_r2': r2_score(y_test, gb_pred),
        'gb_rmse': np.sqrt(mean_squared_error(y_test, gb_pred)),
        'gb_mae': mean_absolute_error(y_test, gb_pred),
        'ensemble_r2': r2_score(y_test, ensemble_pred),
        'ensemble_rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
        'ensemble_mae': mean_absolute_error(y_test, ensemble_pred)
    }
    
    print("\n=== Model Performance ===")
    print(f"Random Forest R²: {metrics['rf_r2']:.4f}")
    print(f"Gradient Boosting R²: {metrics['gb_r2']:.4f}")
    print(f"Ensemble R²: {metrics['ensemble_r2']:.4f}")
    print(f"\nEnsemble RMSE: ${metrics['ensemble_rmse']:,.2f}")
    print(f"Ensemble MAE: ${metrics['ensemble_mae']:,.2f}")
    
    # Save model components
    model_data = {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'metrics': metrics
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"\n✅ Model saved successfully to '{model_path}'")
    print(f"Feature count: {len(model_data['feature_names'])}")
    
    return model_data


if __name__ == "__main__":
    # CHANGE THIS PATH TO YOUR TRAINING DATA FILE
    TRAINING_DATA_PATH = "C:/Users/Kavishnaa Sri/Downloads/House prediction (1)/House prediction/kc_house_data.csv"  # <-- Update this
    MODEL_SAVE_PATH = "trained_model.pkl"
    
    print("Loading data...")
    df_model = load_and_preprocess_data(TRAINING_DATA_PATH)
    print(f"Data loaded: {df_model.shape[0]} rows, {df_model.shape[1]} columns")
    
    # Train and save model
    model_data = train_and_save_model(df_model, MODEL_SAVE_PATH)
    
    print("\n🎉 Training complete! You can now use 'app.py' for predictions.")