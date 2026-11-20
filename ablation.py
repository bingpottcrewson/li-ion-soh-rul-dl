import os
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout

DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"

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

# 1. Load & Scale Data
train_ids = ['B0005', 'B0007', 'B0018']
X_train_list, y_train_list = [], []
for b_id in train_ids:
    X_b, soh_b, _, _ = load_battery_normalized_soh(b_id)
    X_train_list.append(X_b)
    y_train_list.append(soh_b)

X_train_raw = np.concatenate(X_train_list, axis=0)
y_train_soh = np.concatenate(y_train_list, axis=0)
X_test_raw, y_test_soh, y_test_cap, c0_b6 = load_battery_normalized_soh('B0006')

scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

# 2. Define the 3 Ablation Architectures
def build_cnn_only():
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(200, 3)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        Flatten(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_lstm_only():
    return Sequential([
        LSTM(64, activation='tanh', return_sequences=False, input_shape=(200, 3)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

def build_hybrid_cnn_lstm():
    return Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(200, 3)),
        MaxPooling1D(pool_size=2),
        Conv1D(32, kernel_size=3, activation='relu'),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])

models_dict = {
    "1D-CNN Only": build_cnn_only,
    "Standalone LSTM": build_lstm_only,
    "Hybrid CNN-LSTM": build_hybrid_cnn_lstm
}

def find_eol_cycle(series, threshold=0.70):
    below = np.where(series <= threshold)[0]
    return int(below[0] + 1) if len(below) > 0 else len(series)

actual_eol = find_eol_cycle(y_test_soh, 0.70)
results = {}

# 3. Train and Evaluate Each Architecture Under Identical Seeds & Hyperparameters
for name, builder in models_dict.items():
    np.random.seed(42)
    tf.random.set_seed(42)
    
    model = builder()
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    
    start_time = time.time()
    model.fit(
        X_train_scaled, y_train_soh,
        epochs=60,
        batch_size=16,
        validation_split=0.1,
        shuffle=True,
        verbose=0
    )
    train_time = time.time() - start_time
    
    y_pred_soh = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred_cap = y_pred_soh * c0_b6
    
    soh_rmse = np.sqrt(mean_squared_error(y_test_soh, y_pred_soh)) * 100
    soh_mae = mean_absolute_error(y_test_soh, y_pred_soh) * 100
    cap_rmse = np.sqrt(mean_squared_error(y_test_cap, y_pred_cap))
    r2 = r2_score(y_test_soh, y_pred_soh)
    pred_eol = find_eol_cycle(y_pred_soh, 0.70)
    rul_err = abs(actual_eol - pred_eol)
    
    results[name] = {
        "pred_soh": y_pred_soh,
        "params": model.count_params(),
        "time": train_time,
        "soh_rmse": soh_rmse,
        "soh_mae": soh_mae,
        "cap_rmse": cap_rmse,
        "r2": r2,
        "rul_err": rul_err
    }

# 4. Print Formatted Comparison Table
print("\n" + "="*88)
print(f"{'Model Variant':<18} | {'Params':<8} | {'Time(s)':<7} | {'SOH RMSE':<9} | {'SOH MAE':<8} | {'R2':<7} | {'RUL Err':<8}")
print("="*88)
for name, r in results.items():
    print(f"{name:<18} | {r['params']:<8} | {r['time']:<7.2f} | {r['soh_rmse']:<8.2f}% | {r['soh_mae']:<7.2f}% | {r['r2']:<7.4f} | {r['rul_err']} cyc")
print("="*88)

# 5. Plot All 3 Predictions Against Ground Truth on Unseen B0006
cycles = np.arange(1, len(y_test_soh) + 1)
plt.figure(figsize=(12, 6))
plt.plot(cycles, y_test_soh * 100, label='B0006 Ground Truth SOH', color='black', linewidth=2.5)
plt.plot(cycles, results["1D-CNN Only"]["pred_soh"] * 100, label='1D-CNN Only', color='orange', linestyle=':', linewidth=1.8)
plt.plot(cycles, results["Standalone LSTM"]["pred_soh"] * 100, label='Standalone LSTM', color='green', linestyle='-.', linewidth=1.8)
plt.plot(cycles, results["Hybrid CNN-LSTM"]["pred_soh"] * 100, label='Hybrid CNN-LSTM (Proposed)', color='red', linestyle='--', linewidth=2.2)

plt.axhline(y=70.0, color='gray', linestyle='--', alpha=0.6, label='70% EOL Threshold')
plt.title('Ablation Study on Unseen B0006: CNN vs. LSTM vs. Hybrid CNN-LSTM')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('State of Health (%)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
