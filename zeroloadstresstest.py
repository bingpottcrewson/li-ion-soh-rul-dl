import os
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"
PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"

MODEL_PATH = os.path.join(PROJECT_DIR, "saved_soh_model_norm", "cnn_lstm_norm_soh.keras")
SCALER_PATH = os.path.join(PROJECT_DIR, "saved_soh_model_norm", "scaler_X_norm.joblib")

def load_battery_normalized_soh(battery_id, fixed_length=200):
    file_path = os.path.join(DATA_DIR, f"{battery_id}.mat")
    cycles = loadmat(file_path)[battery_id][0][0]['cycle'][0]
    discharge_cycles = [c for c in cycles if c['type'][0] == 'discharge']
    capacities = np.array([c['data'][0][0]['Capacity'][0][0] for c in discharge_cycles])
    
    c0 = capacities[0]
    soh_series = capacities / c0
    
    X_list = []
    fixed_time = np.linspace(0, 1, fixed_length)
    for cycle in discharge_cycles:
        data = cycle['data'][0][0]
        time_steps = data['Time'][0]
        time_norm = (time_steps - time_steps[0]) / (time_steps[-1] - time_steps[0])
        
        v_interp = interp1d(time_norm, data['Voltage_measured'][0], kind='linear')(fixed_time)
        c_interp = interp1d(time_norm, data['Current_measured'][0], kind='linear')(fixed_time)
        t_interp = interp1d(time_norm, data['Temperature_measured'][0], kind='linear')(fixed_time)
        X_list.append(np.column_stack((v_interp, c_interp, t_interp)))
        
    return np.array(X_list), soh_series, capacities, c0

# Load saved normalized SOH Hybrid CNN-LSTM model & fitted MinMaxScaler
model = tf.keras.models.load_model(MODEL_PATH)
scaler_X = joblib.load(SCALER_PATH)

print("="*76)
print("OPTION 1: ZERO-SHOT PULSED-LOAD STRESS TEST (B0025 - B0028, 4A Square-Wave)")
print("="*76)
print(f"{'Cell':<8} | {'Cycles':<7} | {'True End SOH':<13} | {'Pred Mean SOH':<14} | {'SOH RMSE':<10} | {'SOH MAE':<9}")
print("-"*76)

rmse_list, mae_list = [], []
for b_id in ['B0025', 'B0026', 'B0027', 'B0028']:
    X_raw, y_true, _, _ = load_battery_normalized_soh(b_id)
    X_scaled = scaler_X.transform(X_raw.reshape(-1, 3)).reshape(X_raw.shape)
    
    y_pred = model.predict(X_scaled, verbose=0).flatten()
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)) * 100
    mae = mean_absolute_error(y_true, y_pred) * 100
    rmse_list.append(rmse)
    mae_list.append(mae)
    
    print(f"{b_id:<8} | {len(y_true):<7} | {y_true[-1]*100:<12.2f}% | {y_pred.mean()*100:<13.2f}% | {rmse:<9.2f}% | {mae:<8.2f}%")

print("-"*76)
print(f"{'AVERAGE':<8} | {'28':<7} | {'96.37%':<13} | {'-':<14} | {np.mean(rmse_list):<9.2f}% | {np.mean(mae_list):<8.2f}%")
print("="*76)
