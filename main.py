import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import joblib
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
import altair as alt
import shap
import base64
from io import BytesIO
from PIL import Image
import threading
from scipy.signal import savgol_filter

# Import local modules
from train_model import load_data, preprocess_data, evaluate_model, hypertune_models, blend_predict
from predict_glucose import (
    AS726xSensor, 
    GlucosePredictor, 
    simulate_nir_reading, 
    categorize_glucose,
    create_trend_data,
    plot_glucose_trend
)

# App configuration
st.set_page_config(
    page_title="NIR Glucose Monitor",
    page_icon="images/page_icon.jpeg" if os.path.exists("images/page_icon.jpeg") else "🩸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'glucose_history' not in st.session_state:
    st.session_state.glucose_history = []
if 'last_reading_time' not in st.session_state:
    st.session_state.last_reading_time = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = GlucosePredictor()
if 'sensor' not in st.session_state:
    st.session_state.sensor = None
if 'use_real_sensor' not in st.session_state:
    st.session_state.use_real_sensor = False

# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .glucose-reading {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .footer {
        text-align: center;
        margin-top: 3rem;
        padding: 1rem;
        font-size: 0.8rem;
        color: #6c757d;
    }
    .card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .header-icon {
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

def init_sensor():
    """Initialize the NIR sensor."""
    try:
        st.session_state.sensor = AS726xSensor()
        return st.session_state.sensor.connected
    except Exception as e:
        st.error(f"Error initializing sensor: {e}")
        return False

def plot_to_base64(fig):
    """Convert a matplotlib figure to base64 encoded string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    return img_str

def get_nir_reading():
    """Get NIR reading from sensor or simulation."""
    if st.session_state.use_real_sensor and st.session_state.sensor and st.session_state.sensor.connected:
        return st.session_state.sensor.read_nir_value()
    else:
        # Simulate reading based on time pattern (for demo)
        hour = datetime.now().hour
        base_glucose = 90  # Normal fasting
        
        # Simulate daily pattern
        if 7 <= hour < 9:  # Breakfast
            base_glucose = 130
        elif 12 <= hour < 14:  # Lunch
            base_glucose = 140
        elif 18 <= hour < 20:  # Dinner
            base_glucose = 150
        elif 22 <= hour < 24:  # Late night
            base_glucose = 110
            
        # Add some randomness
        glucose = base_glucose + np.random.normal(0, 10)
        return simulate_nir_reading(glucose)

def make_prediction():
    """Make a glucose prediction from NIR reading."""
    nir_value = get_nir_reading()
    
    if nir_value is None:
        return None
    
    predictor = st.session_state.predictor
    glucose = predictor.predict_glucose(nir_value)
    
    if glucose is not None:
        lower, upper = predictor.get_prediction_interval(nir_value)
        category, color = categorize_glucose(glucose)
        
        # Update history
        st.session_state.glucose_history = create_trend_data(
            glucose, 
            st.session_state.glucose_history
        )
        st.session_state.last_reading_time = datetime.now()
        
        # Plot trend
        trend_path = plot_glucose_trend(st.session_state.glucose_history)
        
        return {
            'nir_value': nir_value,
            'glucose': glucose,
            'lower': lower,
            'upper': upper,
            'category': category,
            'color': color,
            'trend_path': trend_path
        }
    
    return None

def train_page():
    """Training page for the model."""
    st.header("🧪 Train Glucose Prediction Model")
    
    with st.expander("Dataset Information", expanded=False):
        st.write("""
        This application uses Near-Infrared (NIR) spectroscopy data to predict blood glucose levels. 
        The dataset contains NIR readings paired with reference glucose measurements.
        """)
        
        # Load and display sample data
        try:
            df = load_data()
            st.dataframe(df.head(10))
            
            # Show basic statistics
            st.subheader("Dataset Statistics")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"Number of samples: {df.shape[0]}")
                st.write(f"Glucose range: {df['GLUCOSE_LEVEL'].min():.1f} - {df['GLUCOSE_LEVEL'].max():.1f} mg/dL")
            with col2:
                st.write(f"NIR range: {df['NIR_Reading'].min():.1f} - {df['NIR_Reading'].max():.1f}")
                st.write(f"Missing values: {df.isna().sum().sum()}")
            
            # Plot the relationship
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(df['NIR_Reading'], df['GLUCOSE_LEVEL'], alpha=0.5)
            ax.set_xlabel('NIR Reading')
            ax.set_ylabel('Glucose Level (mg/dL)')
            ax.set_title('Relationship between NIR Reading and Glucose Level')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error loading dataset: {e}")
    
    # Model training section
    st.subheader("Train New Model")
    
    col1, col2 = st.columns(2)
    with col1:
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
    with col2:
        n_estimators = st.slider("Number of Trees", 50, 200, 100)
    
    if st.button("Train Model"):
        with st.spinner("Training model..."):
            try:
                # Load and preprocess data
                df = load_data()
                X_train, X_test, y_train, y_test, scaler, pca = preprocess_data(df)
                
                # Override default parameters
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(
                    n_estimators=n_estimators,
                    max_depth=10,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42
                )
                
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                mae, rmse, r2, y_pred = evaluate_model(model, X_test, y_test)
                
                # Save model
                os.makedirs('models', exist_ok=True)
                joblib.dump(model, 'models/model.pkl')
                joblib.dump(scaler, 'models/scaler.pkl')
                joblib.dump(pca, 'models/pca.pkl')
                
                # Reset predictor to use new model
                st.session_state.predictor = GlucosePredictor()
                
                # Show results
                st.success("Model trained successfully!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("MAE", f"{mae:.2f} mg/dL")
                col2.metric("RMSE", f"{rmse:.2f} mg/dL")
                col3.metric("R² Score", f"{r2:.4f}")
                
                # Plot results
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.scatter(y_test, y_pred, alpha=0.5)
                
                # Add perfect prediction line
                p1 = max(max(y_pred), max(y_test))
                p2 = min(min(y_pred), min(y_test))
                ax.plot([p2, p1], [p2, p1], 'r--')
                
                ax.set_xlabel('Actual Glucose (mg/dL)')
                ax.set_ylabel('Predicted Glucose (mg/dL)')
                ax.set_title('Actual vs Predicted Glucose Levels')
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                
                # Feature importance
                st.subheader("Feature Importance")
                st.write("""
                For this simple model with only one feature (NIR Reading), 
                feature importance doesn't provide much insight. In a multi-spectral setup 
                with multiple wavelengths, this would show which wavelengths are most 
                important for glucose prediction.
                """)
                
            except Exception as e:
                st.error(f"Error during training: {e}")

def monitor_page():
    """Real-time glucose monitoring page."""
    st.header("📊 Glucose Monitor")
    if st.button("Take Glucose Reading"):
        result = make_prediction()  # Your prediction function
        if result:
            st.metric("Predicted Glucose", f"{result['glucose']:.1f} mg/dL")
            st.metric("Raw NIR Value", f"{result['nir_value']:.2f}")
            if result['trend_path']:
                st.image(result['trend_path'])
        else:
            st.error("Failed to take reading. Please try again.")

def about_page():
    """About page with information about the project."""
    st.header("ℹ️ About NIR Glucose Monitor")
    
    st.markdown("""
    ## How It Works
    
    This application uses Near-Infrared (NIR) spectroscopy to estimate blood glucose levels non-invasively.
    
    ### The Science Behind It
    
    NIR spectroscopy works by measuring how near-infrared light is absorbed or reflected by glucose molecules
    in the blood. Different molecules absorb light at different wavelengths, creating a unique "spectral fingerprint."
    
    The AS726x sensor used in this project captures this spectral information, which our machine learning model
    translates into a glucose estimate.
    
    ### Hardware Setup
    
    - **Raspberry Pi 4**: Serves as the computing platform for the application
    - **AS726x Sensor**: Measures NIR spectral data through the skin
    
    ### Accuracy Considerations
    
    Non-invasive glucose monitoring is still an evolving technology with several challenges:
    
    1. **Interference**: Other molecules in the tissue can affect readings
    2. **Calibration**: Individual differences require personalized calibration
    3. **Environmental factors**: Temperature, humidity, and pressure can influence readings
    
    This application should be considered an experimental tool and not a replacement for medical-grade
    glucose meters.
    """)
    
    st.subheader("Clinical Glucose Ranges")
    
    # Create a dataframe for glucose ranges
    ranges_df = pd.DataFrame({
        'Category': ['Hypoglycemia', 'Normal', 'Prediabetes', 'Diabetes', 'Severe Hyperglycemia'],
        'Range (mg/dL)': ['Below 70', '70-99', '100-125', '126-199', '200 and above'],
        'Description': [
            'Low blood sugar that may require immediate attention',
            'Healthy fasting glucose level',
            'Higher than normal, but not yet diabetes',
            'Elevated glucose indicating diabetes',
            'Very high glucose requiring immediate medical attention'
        ]
    })
    
    st.table(ranges_df)
    
    st.subheader("Project Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### Technical Stack
        
        - **Frontend**: Streamlit
        - **Backend**: Python
        - **ML Framework**: Scikit-learn
        - **Hardware**: Raspberry Pi 4, AS726x NIR Sensor
        - **Data Processing**: Pandas, NumPy
        - **Visualization**: Matplotlib, Seaborn
        """)
    
    with col2:
        st.markdown("""
        ### References
        
        1. Uwadaira, Y., et al. (2018). "Blood glucose measurement by infrared spectroscopy."
        2. Zeng, B., et al. (2020). "Near-infrared spectroscopy for non-invasive blood glucose detection."
        3. Vashist, S.K. (2022). "Non-invasive glucose monitoring technology in diabetes management."
        """)

def advanced_page():
    """Advanced analysis and calibration page."""
    st.header("🔬 Advanced Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Model Analysis", "Calibration", "System Diagnostics"])
    
    with tab1:
        st.subheader("Model Performance Analysis")
        
        if os.path.exists('models/model.pkl') and os.path.exists('models/scaler.pkl'):
            # Load model and data
            model = joblib.load('models/model.pkl')
            scaler = joblib.load('models/scaler.pkl')
            
            try:
                df = load_data()
                df['NIR_Reading_Smooth'] = savgol_filter(df['NIR_Reading'], window_length=7, polyorder=2)
                X = df[['NIR_Reading_Smooth']]
                y = df['GLUCOSE_LEVEL']
                
                # Calculate SHAP values
                st.write("### Feature Importance Analysis")
                
                st.write("""
                SHAP (SHapley Additive exPlanations) values show how each feature
                affects the model prediction for each sample. In a single-feature model
                like this one, it shows how different NIR values impact the glucose prediction.
                """)
                
                with st.spinner("Calculating SHAP values..."):
                    # For large datasets, sample to speed up calculation
                    if len(X) > 100:
                        sampled_indices = np.random.choice(len(X), 100, replace=False)
                        X_sample = X.iloc[sampled_indices]
                    else:
                        X_sample = X
                    
                    X_sample = X_sample[['NIR_Reading_Smooth']] if 'NIR_Reading_Smooth' in X_sample else X_sample
                    X_sample_scaled = scaler.transform(X_sample)
                    
                    # Create explainer and calculate SHAP values
                    explainer = shap.Explainer(model.predict, X_sample_scaled)
                    shap_values = explainer(X_sample_scaled)
                    
                    # Create and display SHAP plots
                    plt.figure()
                    shap_plot = shap.plots.beeswarm(shap_values, show=False)
                    st.pyplot(plt.gcf())
                    plt.close()
                
                # Error analysis
                st.write("### Error Analysis")
                
                # Make predictions on the entire dataset
                X_scaled = scaler.transform(X)
                y_pred = model.predict(X_scaled)
                
                # Calculate errors
                errors = y_pred - y
                abs_errors = np.abs(errors)
                
                # Create a DataFrame for analysis
                error_df = pd.DataFrame({
                    'Actual': y,
                    'Predicted': y_pred,
                    'Error': errors,
                    'AbsError': abs_errors
                })
                
                # Group by glucose ranges
                error_df['Range'] = pd.cut(
                    error_df['Actual'], 
                    bins=[0, 70, 100, 126, 200, float('inf')],
                    labels=['Hypoglycemia', 'Normal', 'Prediabetes', 'Diabetes', 'Severe']
                )
                
                # Calculate metrics per range
                range_errors = error_df.groupby('Range')['AbsError'].agg(['mean', 'std', 'count'])
                range_errors.columns = ['MAE', 'STD', 'Count']
                
                # Show range errors
                st.write("Error by Glucose Range:")
                st.dataframe(range_errors)
                
                # Plot error distribution
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Error histogram
                ax1.hist(errors, bins=30, alpha=0.7)
                ax1.set_xlabel('Error (mg/dL)')
                ax1.set_ylabel('Frequency')
                ax1.set_title('Error Distribution')
                ax1.axvline(x=0, color='r', linestyle='--')
                
                # Error vs Actual
                ax2.scatter(error_df['Actual'], error_df['Error'], alpha=0.5)
                ax2.axhline(y=0, color='r', linestyle='--')
                ax2.set_xlabel('Actual Glucose (mg/dL)')
                ax2.set_ylabel('Error (mg/dL)')
                ax2.set_title('Error vs Actual Glucose')
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close()
                
                # Error metrics
                col1, col2, col3 = st.columns(3)
                col1.metric("Mean Absolute Error", f"{np.mean(abs_errors):.2f} mg/dL")
                col2.metric("Root Mean Squared Error", f"{np.sqrt(np.mean(errors**2)):.2f} mg/dL")
                col3.metric("% within ±15 mg/dL", f"{100 * np.mean(abs_errors < 15):.1f}%")
                
            except Exception as e:
                st.error(f"Error analyzing model: {e}")
        else:
            st.warning("No trained model found. Please go to the Training page to train a model first.")
    
    with tab2:
        st.subheader("Personal Calibration")
        
        st.write("""
        Individual variations in skin, tissue, and blood properties can affect NIR readings.
        Personal calibration improves accuracy by adjusting the model to your specific characteristics.
        """)
        
        st.info("""
        **How to calibrate:**
        1. Take a glucose reading with a traditional glucometer
        2. Take an NIR reading with this device
        3. Enter both values below to create a calibration point
        4. Collect at least 5-7 points across different glucose levels and times of day
        """)
        
        # Calibration form
        col1, col2 = st.columns(2)
        
        with col1:
            reference_glucose = st.number_input(
                "Reference Glucose (mg/dL)", 
                min_value=40, 
                max_value=600,
                value=100
            )
        
        with col2:
            if st.session_state.use_real_sensor and st.session_state.sensor and st.session_state.sensor.connected:
                if st.button("Take NIR Reading for Calibration"):
                    with st.spinner("Taking reading..."):
                        nir_value = st.session_state.sensor.read_nir_value()
                        if nir_value:
                            st.session_state.calibration_nir = nir_value
                            st.success(f"NIR Reading: {nir_value:.2f}")
                        else:
                            st.error("Failed to take NIR reading")
            else:
                st.warning("Connect to real sensor for calibration")
                nir_value = st.number_input("Simulated NIR Value", min_value=100, max_value=1000, value=400)
                st.session_state.calibration_nir = nir_value
        
        # Initialize calibration points in session state if not exist
        if 'calibration_points' not in st.session_state:
            st.session_state.calibration_points = []
            
        # Add calibration point
        if st.button("Add Calibration Point"):
            if hasattr(st.session_state, 'calibration_nir'):
                # Add point to calibration data
                point = {
                    'timestamp': datetime.now(),
                    'nir': st.session_state.calibration_nir,
                    'reference': reference_glucose
                }
                st.session_state.calibration_points.append(point)
                st.success("Calibration point added!")
            else:
                st.error("Please take an NIR reading first")
        
        # Display calibration points
        if st.session_state.calibration_points:
            st.subheader("Calibration Data")
            
            # Convert to DataFrame
            cal_df = pd.DataFrame(st.session_state.calibration_points)
            cal_df['Time'] = cal_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            
            # Display table
            st.dataframe(cal_df[['Time', 'nir', 'reference']].sort_values('timestamp', ascending=False))
            
            # Plot calibration points
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.scatter(cal_df['nir'], cal_df['reference'], alpha=0.7)
            ax.set_xlabel('NIR Reading')
            ax.set_ylabel('Reference Glucose (mg/dL)')
            ax.set_title('Calibration Points')
            ax.grid(True, alpha=0.3)
            
            # Add best fit line if we have enough points
            if len(cal_df) >= 3:
                from scipy import stats
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    cal_df['nir'], cal_df['reference']
                )
                x = np.array([min(cal_df['nir']), max(cal_df['nir'])])
                ax.plot(x, intercept + slope*x, 'r--', 
                        label=f'y = {slope:.2f}x + {intercept:.2f} (R² = {r_value**2:.2f})')
                ax.legend()
            
            st.pyplot(fig)
            
            # Create calibration model
            if len(cal_df) >= 5 and st.button("Create Personal Calibration Model"):
                with st.spinner("Creating calibration model..."):
                    try:
                        from sklearn.linear_model import LinearRegression
                        
                        # Prepare data
                        X_cal = cal_df['nir'].values.reshape(-1, 1)
                        y_cal = cal_df['reference'].values
                        
                        # Fit model
                        cal_model = LinearRegression()
                        cal_model.fit(X_cal, y_cal)
                        
                        # Save calibration model
                        os.makedirs('models', exist_ok=True)
                        joblib.dump(cal_model, 'models/calibration_model.pkl')
                        
                        # Calculate metrics
                        y_cal_pred = cal_model.predict(X_cal)
                        mae = mean_absolute_error(y_cal, y_cal_pred)
                        r2 = r2_score(y_cal, y_cal_pred)
                        
                        st.success(f"Calibration model created! (MAE: {mae:.2f} mg/dL, R²: {r2:.3f})")
                    except Exception as e:
                        st.error(f"Error creating calibration model: {e}")
    
    with tab3:
        st.subheader("System Diagnostics")
        
        # System information
        st.write("### System Information")
        
        import platform
        
        col1, col2 = st.columns(2)
        
        try:
            with col1:
                st.write("**Operating System:**", platform.platform())
                st.write("**Processor:**", platform.processor())
                st.write("**Python Version:**", platform.python_version())
            
            with col2:
                # Raspberry Pi specific info if available
                try:
                    with open('/proc/device-tree/model', 'r') as f:
                        pi_model = f.read().strip()
                    st.write("**Raspberry Pi Model:**", pi_model)
                except:
                    st.write("**Device:**", "Not running on Raspberry Pi")
                
                # Check sensor connection
                if st.session_state.sensor and st.session_state.sensor.connected:
                    st.write("**Sensor Status:**", "✅ Connected")
                else:
                    st.write("**Sensor Status:**", "❌ Not connected")
                
                # Check model availability
                if os.path.exists('models/model.pkl'):
                    st.write("**ML Model:**", "✅ Available")
                else:
                    st.write("**ML Model:**", "❌ Not available")
        except Exception as e:
            st.error(f"Error retrieving system information: {e}")
        
        # Hardware test button
        if st.button("Test Sensor"):
            if st.session_state.use_real_sensor:
                if st.session_state.sensor is None:
                    with st.spinner("Initializing sensor..."):
                        st.session_state.sensor = AS726xSensor()
                
                if st.session_state.sensor.connected:
                    with st.spinner("Taking test reading..."):
                        reading = st.session_state.sensor.read_nir_value()
                        if reading is not None:
                            st.success(f"Sensor test successful! NIR Reading: {reading:.2f}")
                        else:
                            st.error("Sensor test failed. Could not get reading.")
                else:
                    st.error("Sensor not connected. Check connections and try again.")
            else:
                st.warning("Using simulated mode. Switch to real sensor mode to test hardware.")

def main():
    """Main function for the Streamlit app."""
    # App title
    st.title("🩸 NIR Glucose Monitor")
    
    # Navigation
    if os.path.exists("images/page_icon.jpeg"):
        st.sidebar.image("images/page_icon.jpeg", width=100)
    st.sidebar.title("Navigation")
    
    # Page selection
    page = st.sidebar.radio(
        "Select Page",
        ["Glucose Monitor", "Training", "Advanced Analysis", "About"]
    )
    
    # Display selected page
    if page == "Glucose Monitor":
        monitor_page()
    elif page == "Training":
        train_page()
    elif page == "Advanced Analysis":
        advanced_page()
    elif page == "About":
        about_page()
    
    # Footer
    st.markdown("""
    <div class="footer">
        <p>NIR Glucose Monitor | Raspberry Pi-based non-invasive glucose monitoring with AS726x sensor</p>
        <p>Disclaimer: This is an experimental system and should not replace medical-grade glucose monitors.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()