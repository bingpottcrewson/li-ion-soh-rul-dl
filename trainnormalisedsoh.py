import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# Fix random seeds for reproducible training
np.random.seed(42)
tf.random.set_seed(42)

DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"

def load_battery_normalized_soh(battery_id, fixed_length=200):
    file_path = os.path.join(DATA_DIR, f"{battery_id}.mat")
    mat_contents = loadmat(file_path)
    cycles = mat_contents[battery_id][0][0]['cycle'][0]
    
    discharge_cycles = [c for c in cycles if c['type'][0] == 'discharge']
    capacities = np.array([c['data'][0][0]['Capacity'][0][0] for c in discharge_cycles])
    
    # Normalize target by this specific cell's Cycle 1 capacity (C_0)
    c0 = capacities[0]
    soh_series = capacities / c0
    
    X_list = []
    fixed_time = np.linspace(0, 1, fixed_length)
    
    for cycle in discharge_cycles:
        data = cycle['data'][0][0]
        v = data['Voltage_measured'][0]
        i = data['Current_measured'][0]
        t = data['Temperature_measured'][0]
        time_steps = data['Time'][0]
        
        time_norm = (time_steps - time_steps[0]) / (time_steps[-1] - time_steps[0])
        
        v_interp = interp1d(time_norm, v, kind='linear')(fixed_time)
        c_interp = interp1d(time_norm, i, kind='linear')(fixed_time)
        t_interp = interp1d(time_norm, t, kind='linear')(fixed_time)
        
        X_list.append(np.column_stack((v_interp, c_interp, t_interp)))
        
    return np.array(X_list), soh_series, capacities, c0

# 1. Load Training Cells (B0005, B0007, B0018) and Unseen Test Cell (B0006)
train_ids = ['B0005', 'B0007', 'B0018']
X_train_list, y_train_list = [], []

for b_id in train_ids:
    X_b, soh_b, _, c0_b = load_battery_normalized_soh(b_id)
    X_train_list.append(X_b)
    y_train_list.append(soh_b)
    print(f"Loaded {b_id}: {len(soh_b)} cycles | C0 = {c0_b:.4f} Ah | Min SOH = {soh_b.min():.4f}")

X_train_raw = np.concatenate(X_train_list, axis=0)
y_train_soh = np.concatenate(y_train_list, axis=0)

X_test_raw, y_test_soh, y_test_cap, c0_b6 = load_battery_normalized_soh('B0006')
print(f"Loaded B0006 (Test): {len(y_test_soh)} cycles | C0 = {c0_b6:.4f} Ah | Min SOH = {y_test_soh.min():.4f}")

# 2. Scale Input Features (SOH target is already naturally bounded in [0, 1])
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

# 3. Build and Compile CNN-LSTM Model
norm_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(200, 3)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

norm_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 4. Train on Multi-Battery Normalized SOH
print("\nStarting Normalized SOH multi-battery training...")
norm_model.fit(
    X_train_scaled, y_train_soh,
    epochs=60,
    batch_size=16,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# 5. Predict Normalized SOH & Reconstruct Capacity (Ah) on Unseen B0006
y_pred_soh = norm_model.predict(X_test_scaled).flatten()
y_pred_cap = y_pred_soh * c0_b6

# Evaluation Metrics (Normalized SOH % and Reconstructed Capacity Ah)
rmse_soh = np.sqrt(mean_squared_error(y_test_soh, y_pred_soh)) * 100
mae_soh = mean_absolute_error(y_test_soh, y_pred_soh) * 100
r2_soh = r2_score(y_test_soh, y_pred_soh)
rmse_cap = np.sqrt(mean_squared_error(y_test_cap, y_pred_cap))

print("\n--- Normalized SOH Evaluation on Unseen B0006 ---")
print(f"SOH RMSE (%):                   {rmse_soh:.2f}%")
print(f"SOH MAE (%):                    {mae_soh:.2f}%")
print(f"Reconstructed Capacity RMSE:    {rmse_cap:.4f} Ah (vs 0.0938 Ah raw)")
print(f"R-squared (R2) Score:           {r2_soh:.4f} (vs 0.8606 raw)")

# 6. Remaining Useful Life (RUL) at 70% SOH Threshold (SOH = 0.70)
SOH_EOL_THRESHOLD = 0.70
cycles = np.arange(1, len(y_test_soh) + 1)

def find_eol_cycle(series, threshold):
    below = np.where(series <= threshold)[0]
    return int(below[0] + 1) if len(below) > 0 else len(series)

actual_eol = find_eol_cycle(y_test_soh, SOH_EOL_THRESHOLD)
pred_eol = find_eol_cycle(y_pred_soh, SOH_EOL_THRESHOLD)
rul_error = abs(actual_eol - pred_eol)

print("\n--- Normalized SOH RUL Prognostics on B0006 ---")
print(f"EOL Threshold:                  70.0% SOH ({SOH_EOL_THRESHOLD * c0_b6:.4f} Ah)")
print(f"Actual EOL Cycle:               Cycle {actual_eol}")
print(f"Predicted EOL Cycle:            Cycle {pred_eol}")
print(f"Absolute RUL Cycle Error:       {rul_error} cycles (vs 39 cycles raw)")
print(f"Relative RUL Error:             {(rul_error / actual_eol) * 100:.2f}%")

# 7. Save Normalized SOH Model & Scaler
save_dir = "saved_soh_model_norm"
os.makedirs(save_dir, exist_ok=True)
norm_model.save(os.path.join(save_dir, "cnn_lstm_norm_soh.keras"))
joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X_norm.joblib"))

# 8. Plot Normalized SOH & RUL Comparison
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

ax1.plot(cycles, y_test_soh * 100, label='B0006 Actual SOH (%)', color='blue', linewidth=2)
ax1.plot(cycles, y_pred_soh * 100, label='Predicted SOH (%) [Normalized C/C0]', color='red', linestyle='--', linewidth=2)
ax1.axhline(y=70.0, color='black', linestyle=':', linewidth=1.5, label='EOL Threshold (70% SOH)')
ax1.axvline(x=actual_eol, color='blue', linestyle='-.', alpha=0.7, label=f'Actual EOL (Cycle {actual_eol})')
ax1.axvline(x=pred_eol, color='red', linestyle='-.', alpha=0.7, label=f'Predicted EOL (Cycle {pred_eol})')
ax1.set_title('Normalized SOH Prognostics & RUL Detection on Unseen B0006')
ax1.set_ylabel('State of Health (%)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

actual_rul_curve = np.maximum(0, actual_eol - cycles)
pred_rul_curve = np.maximum(0, pred_eol - cycles)
ax2.plot(cycles, actual_rul_curve, label='Actual RUL (Cycles)', color='blue', linewidth=2)
ax2.plot(cycles, pred_rul_curve, label='Predicted RUL (Cycles)', color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Discharge Cycle Number')
ax2.set_ylabel('Remaining Useful Life (Cycles)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
