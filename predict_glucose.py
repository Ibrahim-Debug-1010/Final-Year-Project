import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import board
import busio
import smbus2 as smbus  # For SparkFun AS726x I2C communication
import time
import sys

class AS726xSensor:
    """Class to interface with the SparkFun AS726x NIR sensor."""
    
    # SparkFun AS726x I2C Address
    AS726X_ADDR = 0x49
    
    # Register addresses
    DEVICE_TYPE = 0x00
    HW_VERSION = 0x01
    CONTROL_SETUP = 0x04
    INT_T = 0x05
    DEVICE_TEMP = 0x06
    LED_CONTROL = 0x07
    
    # Data registers (AS7263 NIR sensor)
    R_G = 0x08
    S_H = 0x0A
    T_I = 0x0C
    U_J = 0x0E
    V_K = 0x10
    W_L = 0x12
    
    # Control register bits
    DATA_RDY = 0x02
    GAIN_1X = 0x00
    GAIN_3X = 0x01
    GAIN_6X = 0x02
    GAIN_9X = 0x03
    GAIN_18X = 0x04
    
    # Integration time options (in ms)
    INTEG_TIME = {
        0: 1,      # 1ms
        1: 2.8,    # 2.8ms
        2: 5.6,    # 5.6ms
        3: 11.2,   # 11.2ms
        4: 22.4,   # 22.4ms
        5: 44.8,   # 44.8ms
        6: 89.6,   # 89.6ms
        7: 179.2,  # 179.2ms
        8: 358.4,  # 358.4ms
        9: 716.8,  # 716.8ms
        10: 1433.6 # 1433.6ms
    }
    
    def __init__(self):
        """Initialize the AS726x sensor."""
        try:
            # Initialize I2C bus
            self.bus = smbus.SMBus(1)  # Using I2C bus 1 on Raspberry Pi
            
            # Check if sensor is connected
            try:
                device_type = self.read_register(self.DEVICE_TYPE)
                if device_type != 0x63:  # AS7263 type should be 0x63
                    print(f"Warning: Unexpected device type: {device_type}")
            except Exception as e:
                print(f"Error reading device type: {e}")
                raise
            
            # Configure the sensor
            self.set_gain(self.GAIN_18X)  # Higher gain for better signal
            self.set_integration_time(7)  # 179.2ms integration time
            
            print("SparkFun AS726x sensor initialized successfully")
            self.connected = True
        except Exception as e:
            print(f"Failed to initialize SparkFun AS726x sensor: {e}")
            self.connected = False
    
    def read_register(self, reg_addr):
        """Read a register from the AS726x."""
        try:
            # Write the register address to the virtual register
            self.bus.write_byte_data(self.AS726X_ADDR, 0x00, reg_addr)
            # Read the data from the virtual register
            return self.bus.read_byte_data(self.AS726X_ADDR, 0x00)
        except Exception as e:
            print(f"Error reading register {reg_addr}: {e}")
            return None
    
    def write_register(self, reg_addr, data):
        """Write to a register on the AS726x."""
        try:
            # Write the register address to the virtual register
            self.bus.write_byte_data(self.AS726X_ADDR, 0x01, reg_addr)
            # Write the data to the virtual register
            self.bus.write_byte_data(self.AS726X_ADDR, 0x01, data)
            return True
        except Exception as e:
            print(f"Error writing to register {reg_addr}: {e}")
            return False
    
    def set_gain(self, gain):
        """Set the gain of the sensor."""
        # Read current control setup
        control = self.read_register(self.CONTROL_SETUP)
        # Clear gain bits (bits 4:2)
        control &= 0xE3
        # Set new gain
        control |= (gain << 2)
        # Write back to control setup
        self.write_register(self.CONTROL_SETUP, control)
    
    def set_integration_time(self, time_setting):
        """Set the integration time of the sensor."""
        if time_setting < 0 or time_setting > 10:
            print("Integration time setting must be between 0 and 10")
            return False
        
        # Write integration time
        self.write_register(self.INT_T, time_setting)
        return True
    
    def data_ready(self):
        """Check if sensor data is ready."""
        control = self.read_register(self.CONTROL_SETUP)
        return (control & self.DATA_RDY) != 0
    
    def read_channels(self):
        """Read all NIR channel values.
        
        Returns:
            dict: Dictionary with keys 'R', 'S', 'T', 'U', 'V', 'W' containing
                 the 16-bit values for each channel.
        """
        channels = {}
        
        # Read R channel (760nm - closest to glucose absorption)
        r_low = self.read_register(self.R_G)
        r_high = self.read_register(self.R_G + 1)
        channels['R'] = (r_high << 8) | r_low
        
        # Read S channel (810nm)
        s_low = self.read_register(self.S_H)
        s_high = self.read_register(self.S_H + 1)
        channels['S'] = (s_high << 8) | s_low
        
        # Read T channel (860nm)
        t_low = self.read_register(self.T_I)
        t_high = self.read_register(self.T_I + 1)
        channels['T'] = (t_high << 8) | t_low
        
        # Read U channel (910nm)
        u_low = self.read_register(self.U_J)
        u_high = self.read_register(self.U_J + 1)
        channels['U'] = (u_high << 8) | u_low
        
        # Read V channel (940nm)
        v_low = self.read_register(self.V_K)
        v_high = self.read_register(self.V_K + 1)
        channels['V'] = (v_high << 8) | v_low
        
        # Read W channel (990nm)
        w_low = self.read_register(self.W_L)
        w_high = self.read_register(self.W_L + 1)
        channels['W'] = (w_high << 8) | w_low
        
        return channels
    
    def read_nir_value(self, average_readings=5, delay=0.5):
        """Read the NIR value from the sensor.
        
        Args:
            average_readings (int): Number of readings to average
            delay (float): Delay between readings in seconds
            
        Returns:
            float: Averaged NIR reading
        """
        if not self.connected:
            return None
            
        readings = []
        
        for _ in range(average_readings):
            # Wait for data to be ready
            for _ in range(10):  # Timeout after 10 attempts
                if self.data_ready():
                    break
                time.sleep(0.1)
            
            # Read all channels
            channels = self.read_channels()
            
            # Use R channel (760nm) which is closest to glucose absorption features
            nir_value = channels['R']
            readings.append(nir_value)
                
            time.sleep(delay)
        
        # Average the readings
        if not readings:
            return None
            
        avg_reading = sum(readings) / len(readings)
        return avg_reading
        
class GlucosePredictor:
    """Class for making glucose predictions from NIR readings using blended ensemble models."""
    def __init__(self, model_paths=None, scaler_path='models/scaler.pkl'):
        # Default model paths for ensemble
        if model_paths is None:
            model_paths = {
                'rf': 'models/rf.pkl',
                'knn': 'models/knn.pkl',
                'svr': 'models/svr.pkl',
                'lgbm': 'models/lgbm.pkl'
            }
        self.models = {}
        for name, path in model_paths.items():
            try:
                self.models[name] = joblib.load(path)
            except Exception as e:
                print(f"Error loading {name} model: {e}")
        try:
            self.scaler = joblib.load(scaler_path)
            print("Scaler loaded successfully")
        except Exception as e:
            print(f"Error loading scaler: {e}")
            self.scaler = None

    def predict_glucose(self, nir_value):
        if not self.models or self.scaler is None:
            return None
        if nir_value is None or not np.isfinite(nir_value):
            return None
        # Load PCA
        try:
            pca = joblib.load('models/pca.pkl')
        except Exception as e:
            print(f"Error loading PCA: {e}")
            return None
        X_features = preprocess_single_nir(nir_value, self.scaler, pca)
        preds = []
        for model in self.models.values():
            preds.append(model.predict(X_features)[0])
        if preds:
            return float(np.mean(preds))
        return None

    def get_prediction_interval(self, nir_value, percentile=95):
        # Not supported for ensemble; return None
        return None, None

def preprocess_single_nir(nir_value, scaler, pca):
    # Apply the same preprocessing as in train_model.py
    # 1. Smooth (not needed for single value)
    # 2. Scale
    nir_scaled = scaler.transform([[nir_value]])
    # 3. PCA
    X_pca = pca.transform(nir_scaled)
    # 4. Statistical features (use the same logic as in extract_features)
    stats = np.array([
        nir_value,  # mean (single value)
        0.0,        # skew (not defined for single value)
        0.0         # kurtosis (not defined for single value)
    ]).reshape(1, -1)
    X_features = np.hstack([X_pca, stats])
    return X_features

def simulate_nir_reading(glucose_level):
    """Simulate a NIR reading for testing without hardware.
    
    This function simulates the relationship between glucose level and NIR reading
    based on the provided dataset's pattern, with some random noise.
    """
    base = 300 + glucose_level * 0.8
    noise = np.random.normal(0, 30)  # Add random noise
    return max(100, base + noise)  # Ensure positive NIR value

def categorize_glucose(glucose_level):
    """Categorize glucose level into clinical ranges."""
    if glucose_level < 70:
        return "Hypoglycemia (Low)", "red"
    elif glucose_level < 100:
        return "Normal", "green"
    elif glucose_level < 126:
        return "Prediabetes", "orange"
    elif glucose_level < 200:
        return "Diabetes", "red"
    else:
        return "Severe Hyperglycemia", "darkred"

def create_trend_data(new_reading, history=None):
    """Add new reading to history and return updated history."""
    if history is None:
        history = []
    
    # Add timestamp and reading to history
    timestamp = pd.Timestamp.now()
    history.append((timestamp, new_reading))
    
    # Keep only the last 24 hours of data
    cutoff = pd.Timestamp.now() - pd.Timedelta(hours=24)
    history = [(ts, val) for ts, val in history if ts > cutoff]
    
    return history

def plot_glucose_trend(history, save_path='images/trend.png'):
    """Plot glucose trend over time."""
    if not history:
        return None
    
    # Create dataframe from history
    df = pd.DataFrame(history, columns=['Timestamp', 'Glucose'])
    
    # Plot trend
    plt.figure(figsize=(10, 5))
    plt.plot(df['Timestamp'], df['Glucose'], 'b-o')
    
    # Add reference lines for glucose ranges
    plt.axhspan(0, 70, color='red', alpha=0.1, label='Hypoglycemia')
    plt.axhspan(70, 100, color='green', alpha=0.1, label='Normal')
    plt.axhspan(100, 126, color='orange', alpha=0.1, label='Prediabetes')
    plt.axhspan(126, 200, color='red', alpha=0.1, label='Diabetes')
    plt.axhspan(200, 400, color='darkred', alpha=0.1, label='Severe Hyperglycemia')
    
    plt.xlabel('Time')
    plt.ylabel('Glucose Level (mg/dL)')
    plt.title('Glucose Trend Over Time')
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Format time axis
    plt.gcf().autofmt_xdate()
    
    # Save plot
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def moving_average(predictions, window=3):
    return np.convolve(predictions, np.ones(window)/window, mode='valid')

def main(use_real_sensor=False):
    """Main function for prediction."""
    # Initialize the sensor interface
    if use_real_sensor:
        sensor = AS726xSensor()
        if not sensor.connected:
            print("Using simulated data because sensor is not connected")
            use_real_sensor = False
    
    # Initialize the predictor
    predictor = GlucosePredictor()
    
    # Get NIR reading (either from sensor or simulation)
    if use_real_sensor:
        nir_value = sensor.read_nir_value()
    else:
        # For simulation, we'll assume a glucose level and generate NIR
        true_glucose = 120  # mg/dL (for simulation only)
        nir_value = simulate_nir_reading(true_glucose)
    
    print(f"NIR Reading: {nir_value}")
    
    # Make prediction
    glucose_prediction = predictor.predict_glucose(nir_value)
    
    if glucose_prediction is not None:
        lower, upper = predictor.get_prediction_interval(nir_value)
        category, color = categorize_glucose(glucose_prediction)
        
        print(f"Predicted Glucose: {glucose_prediction:.1f} mg/dL")
        print(f"Prediction Interval: [{lower}, {upper}] mg/dL")
        print(f"Category: {category}")
    else:
        print("Unable to make prediction")
        lower, upper, category, color = None, None, None, None
    
    return {
        'nir_value': nir_value,
        'glucose_prediction': glucose_prediction,
        'lower_bound': lower,
        'upper_bound': upper,
        'category': category,
        'color': color
    }

if __name__ == "__main__":
    result = main(use_real_sensor=False)
    print(result)