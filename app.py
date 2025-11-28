import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PART 1: DATA LOADING AND PREPROCESSING
# ============================================================================


@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Load and preprocess the house data"""

    # Stop early if file not uploaded
    if uploaded_file is None:
        return None

    # CRITICAL FIX: Reset file pointer to beginning before reading
    uploaded_file.seek(0)
    
    # Read uploaded CSV file
    df = pd.read_csv(uploaded_file, on_bad_lines='skip')

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

# ============================================================================
# PART 2: MODEL TRAINING
# ============================================================================

def train_model(df_model):
    """Train the machine learning model"""
    
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
    
    # Ensemble prediction (average)
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
    
    return {
        'rf_model': rf_model,
        'gb_model': gb_model,
        'scaler': scaler,
        'feature_names': X.columns.tolist(),
        'metrics': metrics,
        'X_test': X_test,
        'y_test': y_test,
        'predictions': {
            'rf': rf_pred,
            'gb': gb_pred,
            'ensemble': ensemble_pred
        }
    }

# ============================================================================
# PART 3: STREAMLIT APP
# ============================================================================

def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    
    st.title("🏠 House Price Prediction System")
    st.markdown("### AI-Powered Real Estate Valuation")
    
    # Sidebar
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Select Page", 
                            ["Upload & Train", "Model Performance", "Make Predictions"])
    
    # Initialize session state
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    
    # ========================================================================
    # PAGE 1: UPLOAD AND TRAIN
    # ========================================================================
    if page == "Upload & Train":
        st.header("📊 Data Upload & Model Training")
        
        uploaded_file = st.file_uploader("Upload Your CSV File", type=['csv'])
        
        if uploaded_file is not None:
            # CRITICAL FIX: Reset file pointer before reading for preview
            uploaded_file.seek(0)
            
            # Load data for preview
            df = pd.read_csv(uploaded_file)
            
            st.success(f"✅ Data loaded successfully! Shape: {df.shape}")
            
            # Show data preview
            with st.expander("View Data Preview"):
                st.dataframe(df.head(10))
            
            # Show data statistics
            with st.expander("View Data Statistics"):
                st.write(df.describe())
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training models... This may take a minute..."):
                    # Preprocess data (file pointer will be reset inside the function)
                    df_model = load_and_preprocess_data(uploaded_file)
                    
                    if df_model is not None:
                        # Train model
                        results = train_model(df_model)
                        
                        # Save to session state
                        st.session_state.model_results = results
                        st.session_state.model_trained = True
                        
                        st.success("✅ Models trained successfully!")
                        st.balloons()
                        
                        # Show quick metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Random Forest R²", 
                                     f"{results['metrics']['rf_r2']:.4f}")
                        with col2:
                            st.metric("Gradient Boosting R²", 
                                     f"{results['metrics']['gb_r2']:.4f}")
                        with col3:
                            st.metric("Ensemble R²", 
                                     f"{results['metrics']['ensemble_r2']:.4f}")
    
    # ========================================================================
    # PAGE 2: MODEL PERFORMANCE
    # ========================================================================
    elif page == "Model Performance":
        st.header("📈 Model Performance Analysis")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Upload & Train' page!")
            return
        
        results = st.session_state.model_results
        metrics = results['metrics']
        
        # Display metrics
        st.subheader("Model Comparison")
        
        metrics_df = pd.DataFrame({
            'Model': ['Random Forest', 'Gradient Boosting', 'Ensemble'],
            'R² Score': [metrics['rf_r2'], metrics['gb_r2'], metrics['ensemble_r2']],
            'RMSE': [metrics['rf_rmse'], metrics['gb_rmse'], metrics['ensemble_rmse']],
            'MAE': [metrics['rf_mae'], metrics['gb_mae'], metrics['ensemble_mae']]
        })
        
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['R² Score'])
                                      .highlight_min(axis=0, subset=['RMSE', 'MAE'])
                                      .format({'R² Score': '{:.4f}', 
                                              'RMSE': '{:.2f}', 
                                              'MAE': '{:.2f}'}))
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # R² Score comparison
            fig_r2 = px.bar(metrics_df, x='Model', y='R² Score',
                           title='R² Score Comparison',
                           color='R² Score',
                           color_continuous_scale='Viridis')
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col2:
            # Error metrics comparison
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Bar(name='RMSE', x=metrics_df['Model'], 
                                       y=metrics_df['RMSE']))
            fig_errors.add_trace(go.Bar(name='MAE', x=metrics_df['Model'], 
                                       y=metrics_df['MAE']))
            fig_errors.update_layout(title='Error Metrics Comparison',
                                    barmode='group')
            st.plotly_chart(fig_errors, use_container_width=True)
        
        # Actual vs Predicted
        st.subheader("Actual vs Predicted Prices")
        
        model_choice = st.selectbox("Select Model", 
                                    ['Random Forest', 'Gradient Boosting', 'Ensemble'])
        
        pred_map = {
            'Random Forest': 'rf',
            'Gradient Boosting': 'gb',
            'Ensemble': 'ensemble'
        }
        
        predictions = results['predictions'][pred_map[model_choice]]
        y_test = results['y_test']
        
        # Scatter plot
        fig_scatter = px.scatter(x=y_test, y=predictions,
                                labels={'x': 'Actual Price', 'y': 'Predicted Price'},
                                title=f'{model_choice} - Actual vs Predicted',
                                opacity=0.6)
        fig_scatter.add_trace(go.Scatter(x=[y_test.min(), y_test.max()],
                                        y=[y_test.min(), y_test.max()],
                                        mode='lines',
                                        name='Perfect Prediction',
                                        line=dict(color='red', dash='dash')))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Feature importance
        st.subheader("Feature Importance (Random Forest)")
        
        importances = results['rf_model'].feature_importances_
        feature_names = results['feature_names']
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(15)
        
        fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                               orientation='h',
                               title='Top 15 Most Important Features',
                               color='Importance',
                               color_continuous_scale='Blues')
        st.plotly_chart(fig_importance, use_container_width=True)
    
    # ========================================================================
    # PAGE 3: MAKE PREDICTIONS
    # ========================================================================
    elif page == "Make Predictions":
        st.header("🔮 Make House Price Predictions")
        
        if not st.session_state.model_trained:
            st.warning("⚠️ Please train the model first in the 'Upload & Train' page!")
            return
        
        results = st.session_state.model_results
        
        st.markdown("### Enter House Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
            bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=8.0, 
                                       value=2.0, step=0.5)
            floors = st.number_input("Floors", min_value=1.0, max_value=3.5, 
                                    value=1.0, step=0.5)
            condition = st.slider("Condition (1-5)", 1, 5, 3)
            grade = st.slider("Grade (1-13)", 1, 13, 7)
        
        with col2:
            sqft_living = st.number_input("Living Area (sqft)", min_value=500, 
                                         max_value=10000, value=2000)
            sqft_lot = st.number_input("Lot Size (sqft)", min_value=500, 
                                       max_value=100000, value=5000)
            sqft_above = st.number_input("Above Ground (sqft)", min_value=500, 
                                        max_value=10000, value=1500)
            sqft_basement = st.number_input("Basement (sqft)", min_value=0, 
                                           max_value=5000, value=0)
        
        with col3:
            yr_built = st.number_input("Year Built", min_value=1900, 
                                      max_value=2024, value=2000)
            yr_renovated = st.number_input("Year Renovated (0 if never)", 
                                          min_value=0, max_value=2024, value=0)
            waterfront = st.selectbox("Waterfront", [0, 1], 
                                      format_func=lambda x: "Yes" if x == 1 else "No")
            view = st.slider("View Rating (0-4)", 0, 4, 0)
            zipcode = st.number_input("Zipcode", min_value=98000, max_value=98999, value=98103)
        
        col4, col5 = st.columns(2)
        with col4:
            lat = st.number_input("Latitude", min_value=47.0, max_value=48.0, 
                                 value=47.6, format="%.6f")
            sqft_living15 = st.number_input("Avg Living Area of 15 Nearest (sqft)", 
                                           min_value=500, max_value=10000, value=2000)
        
        with col5:
            long = st.number_input("Longitude", min_value=-123.0, max_value=-121.0, 
                                  value=-122.3, format="%.6f")
            sqft_lot15 = st.number_input("Avg Lot Size of 15 Nearest (sqft)", 
                                        min_value=500, max_value=100000, value=5000)
        
        if st.button("💰 Predict Price", type="primary"):
            # Create feature dictionary
            input_data = {
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'sqft_living': sqft_living,
                'sqft_lot': sqft_lot,
                'floors': floors,
                'waterfront': waterfront,
                'view': view,
                'condition': condition,
                'grade': grade,
                'sqft_above': sqft_above,
                'sqft_basement': sqft_basement,
                'yr_built': yr_built,
                'yr_renovated': yr_renovated,
                'zipcode': zipcode,
                'lat': lat,
                'long': long,
                'sqft_living15': sqft_living15,
                'sqft_lot15': sqft_lot15
            }
            
            # Feature engineering
            input_data['house_age'] = 2024 - input_data['yr_built']
            input_data['renovated'] = 1 if input_data['yr_renovated'] > 0 else 0
            input_data['years_since_renovation'] = (
                2024 - input_data['yr_renovated'] if input_data['yr_renovated'] > 0 
                else input_data['house_age']
            )
            input_data['total_sqft'] = input_data['sqft_living'] + input_data['sqft_lot']
            input_data['sqft_ratio'] = input_data['sqft_living'] / (input_data['sqft_lot'] + 1)
            input_data['bath_bed_ratio'] = input_data['bathrooms'] / (input_data['bedrooms'] + 1)
            
            # Create DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Ensure correct column order
            input_df = input_df[results['feature_names']]
            
            # Scale features
            input_scaled = results['scaler'].transform(input_df)
            
            # Make predictions
            rf_pred = results['rf_model'].predict(input_scaled)[0]
            gb_pred = results['gb_model'].predict(input_scaled)[0]
            ensemble_pred = (rf_pred + gb_pred) / 2
            
            # Display results
            st.success("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Random Forest", f"${rf_pred:,.2f}")
            with col2:
                st.metric("Gradient Boosting", f"${gb_pred:,.2f}")
            with col3:
                st.metric("Ensemble (Recommended)", f"${ensemble_pred:,.2f}", 
                         delta=f"±${abs(rf_pred - gb_pred)/2:,.2f}")
            
            # Prediction range
            st.info(f"**Estimated Price Range:** ${min(rf_pred, gb_pred):,.2f} - ${max(rf_pred, gb_pred):,.2f}")
            
            # Additional insights
            st.markdown("### 📊 Property Insights")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**Price per sqft:** ${ensemble_pred/sqft_living:,.2f}")
                st.write(f"**House Age:** {input_data['house_age']} years")
                st.write(f"**Renovation Status:** {'Renovated' if input_data['renovated'] else 'Not Renovated'}")
            
            with col2:
                st.write(f"**Total Area:** {input_data['total_sqft']:,} sqft")
                st.write(f"**Living to Lot Ratio:** {input_data['sqft_ratio']:.2%}")
                st.write(f"**Bath/Bed Ratio:** {input_data['bath_bed_ratio']:.2f}")

if __name__ == "__main__":
    main(