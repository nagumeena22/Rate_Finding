import pandas as pd
import numpy as np
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# PREDICTION APP - USERS UPLOAD INPUT AND GET PREDICTIONS
# ============================================================================

@st.cache_resource
def load_trained_model(model_path='trained_model.pkl'):
    """Load the pre-trained model"""
    try:
        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)
        return model_data
    except FileNotFoundError:
        st.error(f"❌ Model file '{model_path}' not found! Please train the model first using 'train_model.py'")
        return None


def preprocess_input(input_data, feature_names):
    """Preprocess user input to match training format"""
    
    # Feature engineering (same as training)
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
    input_df = input_df[feature_names]
    
    return input_df


def predict_from_csv(uploaded_file, model_data):
    """Predict prices for multiple houses from CSV"""
    
    # Read CSV
    df = pd.read_csv(uploaded_file)
    
    # Required columns
    required_cols = [
        'bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors',
        'waterfront', 'view', 'condition', 'grade', 'sqft_above',
        'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode',
        'lat', 'long', 'sqft_living15', 'sqft_lot15'
    ]
    
    # Check for missing columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        return None
    
    # Fill missing values
    df = df.fillna(0)
    
    # Feature engineering
    df['house_age'] = 2024 - df['yr_built']
    df['renovated'] = (df['yr_renovated'] > 0).astype(int)
    df['years_since_renovation'] = 2024 - df['yr_renovated']
    df.loc[df['yr_renovated'] == 0, 'years_since_renovation'] = df['house_age']
    df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
    df['sqft_ratio'] = df['sqft_living'] / (df['sqft_lot'] + 1)
    df['bath_bed_ratio'] = df['bathrooms'] / (df['bedrooms'] + 1)
    
    # Select features in correct order
    X = df[model_data['feature_names']]
    
    # Scale features
    X_scaled = model_data['scaler'].transform(X)
    
    # Make predictions
    rf_pred = model_data['rf_model'].predict(X_scaled)
    gb_pred = model_data['gb_model'].predict(X_scaled)
    ensemble_pred = (rf_pred + gb_pred) / 2
    
    # Add predictions to dataframe
    df['predicted_price_rf'] = rf_pred
    df['predicted_price_gb'] = gb_pred
    df['predicted_price_ensemble'] = ensemble_pred
    
    return df


def main():
    st.set_page_config(page_title="House Price Predictor", layout="wide")
    
    st.title("🏠 House Price Prediction System")
    st.markdown("### Upload Your Data and Get Instant Price Predictions")
    
    # Load pre-trained model
    model_data = load_trained_model()
    
    if model_data is None:
        st.stop()
    
    # Display model info
    with st.expander("📊 Model Information"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Random Forest R²", f"{model_data['metrics']['rf_r2']:.4f}")
        with col2:
            st.metric("Gradient Boosting R²", f"{model_data['metrics']['gb_r2']:.4f}")
        with col3:
            st.metric("Ensemble R²", f"{model_data['metrics']['ensemble_r2']:.4f}")
    
    # Sidebar
    st.sidebar.header("Prediction Mode")
    mode = st.sidebar.radio("Choose Input Method", 
                            ["Single House (Manual Input)", "Multiple Houses (CSV Upload)"])
    
    # ========================================================================
    # MODE 1: SINGLE HOUSE PREDICTION
    # ========================================================================
    if mode == "Single House (Manual Input)":
        st.header("🔮 Single House Price Prediction")
        st.markdown("Enter the house details below:")
        
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
            # Create input dictionary
            input_data = {
                'bedrooms': bedrooms, 'bathrooms': bathrooms,
                'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
                'floors': floors, 'waterfront': waterfront,
                'view': view, 'condition': condition, 'grade': grade,
                'sqft_above': sqft_above, 'sqft_basement': sqft_basement,
                'yr_built': yr_built, 'yr_renovated': yr_renovated,
                'zipcode': zipcode, 'lat': lat, 'long': long,
                'sqft_living15': sqft_living15, 'sqft_lot15': sqft_lot15
            }
            
            # Preprocess input
            input_df = preprocess_input(input_data, model_data['feature_names'])
            
            # Scale features
            input_scaled = model_data['scaler'].transform(input_df)
            
            # Make predictions
            rf_pred = model_data['rf_model'].predict(input_scaled)[0]
            gb_pred = model_data['gb_model'].predict(input_scaled)[0]
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
    
    # ========================================================================
    # MODE 2: BATCH PREDICTION FROM CSV
    # ========================================================================
    else:
        st.header("📁 Batch Price Prediction from CSV")
        
        st.markdown("""
        Upload a CSV file with house data. Required columns:
        - bedrooms, bathrooms, sqft_living, sqft_lot, floors
        - waterfront, view, condition, grade
        - sqft_above, sqft_basement, yr_built, yr_renovated
        - zipcode, lat, long, sqft_living15, sqft_lot15
        """)
        
        # File uploader
        uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        
        if uploaded_file is not None:
            with st.spinner("Processing predictions..."):
                result_df = predict_from_csv(uploaded_file, model_data)
            
            if result_df is not None:
                st.success(f"✅ Predictions complete for {len(result_df)} houses!")
                
                # Summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Houses", len(result_df))
                with col2:
                    st.metric("Avg Predicted Price", 
                             f"${result_df['predicted_price_ensemble'].mean():,.2f}")
                with col3:
                    st.metric("Min Price", 
                             f"${result_df['predicted_price_ensemble'].min():,.2f}")
                with col4:
                    st.metric("Max Price", 
                             f"${result_df['predicted_price_ensemble'].max():,.2f}")
                
                # Show results
                st.subheader("Prediction Results")
                
                # Select columns to display
                display_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'grade', 
                               'predicted_price_ensemble', 'predicted_price_rf', 'predicted_price_gb']
                
                display_df = result_df[display_cols].copy()
                display_df.columns = ['Bedrooms', 'Bathrooms', 'Sqft Living', 'Grade',
                                     'Ensemble Price', 'RF Price', 'GB Price']
                
                st.dataframe(display_df.style.format({
                    'Ensemble Price': '${:,.2f}',
                    'RF Price': '${:,.2f}',
                    'GB Price': '${:,.2f}'
                }))
                
                # Visualization
                st.subheader("Price Distribution")
                
                fig = px.histogram(result_df, x='predicted_price_ensemble',
                                  nbins=30,
                                  title='Distribution of Predicted Prices',
                                  labels={'predicted_price_ensemble': 'Predicted Price'},
                                  color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig, use_container_width=True)
                
                # Download results
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="house_price_predictions.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()