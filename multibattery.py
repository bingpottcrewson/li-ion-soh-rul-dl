import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout

# 1. Helper function to load and interpolate any NASA PCoE battery file
DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"

def load_and_preprocess_battery(battery_id, fixed_length=200):
    file_path = os.path.join(DATA_DIR, f"{battery_id}.mat")
    mat_contents = loadmat(file_path)
    cycles = mat_contents[battery_id][0][0]['cycle'][0]
    
    discharge_cycles = [c for c in cycles if c['type'][0] == 'discharge']
    capacities = np.array([c['data'][0][0]['Capacity'][0][0] for c in discharge_cycles])
    
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
        
    return np.array(X_list), capacities

# 2. Load Training Batteries (B0005, B0007, B0018) and Unseen Test Battery (B0006)
train_ids = ['B0005', 'B0007', 'B0018']
X_train_list, y_train_list = [], []

for b_id in train_ids:
    X_b, y_b = load_and_preprocess_battery(b_id)
    X_train_list.append(X_b)
    y_train_list.append(y_b)
    print(f"Loaded {b_id}: {X_b.shape[0]} discharge cycles")

# Stack all training batteries into unified arrays (168 + 168 + 132 = 468 cycles)
X_train_raw = np.concatenate(X_train_list, axis=0)
y_train_raw = np.concatenate(y_train_list, axis=0)

# Load B0006 completely held out for unseen testing
X_test_raw, y_test_actual = load_and_preprocess_battery('B0006')
print(f"Loaded B0006 (Unseen Test): {X_test_raw.shape[0]} discharge cycles")
print(f"Combined Training Tensor Shape: {X_train_raw.shape}")

# 3. Fit Scalers on Combined Training Data, Transform Test Data
scaler_X = MinMaxScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
X_test_scaled = scaler_X.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train_raw.reshape(-1, 1))

# 4. Build the Hybrid CNN-LSTM Architecture
multi_model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(200, 3)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.2), # Helps prevent overfitting to any single battery's curve
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

multi_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 5. Train on Multi-Battery Dataset (shuffle=True mixes cycles across the 3 training cells)
print("\nStarting multi-battery training...")
history = multi_model.fit(
    X_train_scaled, y_train_scaled,
    epochs=60,
    batch_size=16,
    validation_split=0.1,
    shuffle=True,
    verbose=1
)

# 6. Evaluate on Unseen B0006 Battery
y_pred_scaled = multi_model.predict(X_test_scaled)
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)

rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n--- Multi-Battery Model Evaluation on Unseen B0006 ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} Ah (vs 0.0962 Ah single-cell)")
print(f"Mean Absolute Error (MAE):      {mae:.4f} Ah (vs 0.0635 Ah single-cell)")
print(f"R-squared (R2) Score:           {r2:.4f} (vs 0.8534 single-cell)")

# 7. Save Upgraded Multi-Battery Model & Scalers
save_dir = "saved_soh_model_multi"
os.makedirs(save_dir, exist_ok=True)
multi_model.save(os.path.join(save_dir, "cnn_lstm_multi.keras"))
joblib.dump(scaler_X, os.path.join(save_dir, "scaler_X_multi.joblib"))
joblib.dump(scaler_y, os.path.join(save_dir, "scaler_y_multi.joblib"))
print(f"Saved multi-battery artifacts to '{save_dir}/'")

# 8. Plot Actual vs. Predicted SOH for Unseen B0006
plt.figure(figsize=(12, 6))
plt.plot(range(1, len(y_test_actual) + 1), y_test_actual, label='B0006 Actual Capacity (Ground Truth)', color='blue', linewidth=2)
plt.plot(range(1, len(y_test_actual) + 1), y_pred_actual, label='Multi-Cell CNN-LSTM Predicted Capacity', color='red', linestyle='--', linewidth=2)
plt.title('Multi-Battery Training Generalization: Unseen B0006 SOH Prediction')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('Capacity (Ampere-hours)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# 1. Define the 70% End-of-Life (EOL) failure threshold (1.40 Ah for 2.0 Ah rated cell)
RATED_CAPACITY = 2.0
EOL_THRESHOLD = 0.70 * RATED_CAPACITY  # 1.40 Ah

# Flatten arrays to 1D for clean indexing
actual_cap = y_test_actual.flatten()
pred_cap = y_pred_actual.flatten()
cycles = np.arange(1, len(actual_cap) + 1)

# 2. Helper function to find the first cycle where capacity drops below EOL_THRESHOLD
def find_eol_cycle(capacity_series, threshold):
    below_threshold = np.where(capacity_series <= threshold)[0]
    if len(below_threshold) > 0:
        return int(below_threshold[0] + 1)  # +1 for 1-based cycle numbering
    return len(capacity_series)

actual_eol_cycle = find_eol_cycle(actual_cap, EOL_THRESHOLD)
pred_eol_cycle = find_eol_cycle(pred_cap, EOL_THRESHOLD)

# 3. Calculate RUL Error (in cycles) and Relative RUL Error
rul_cycle_error = abs(actual_eol_cycle - pred_eol_cycle)
relative_rul_error = (rul_cycle_error / actual_eol_cycle) * 100

print("\n--- Remaining Useful Life (RUL) Prognostics on B0006 ---")
print(f"End-of-Life (EOL) Threshold (70%): {EOL_THRESHOLD:.2f} Ah")
print(f"Actual EOL Cycle:                  Cycle {actual_eol_cycle}")
print(f"Predicted EOL Cycle:               Cycle {pred_eol_cycle}")
print(f"Absolute RUL Cycle Error:          {rul_cycle_error} cycles")
print(f"Relative RUL Error:                {relative_rul_error:.2f}%")

# 4. Compute cycle-by-cycle RUL countdown up to EOL
actual_rul_curve = np.maximum(0, actual_eol_cycle - cycles)
pred_rul_curve = np.maximum(0, pred_eol_cycle - cycles)

# 5. Plot SOH Threshold Crossing and RUL Countdown
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

# Top Plot: Capacity Curve with 1.40 Ah EOL Threshold
ax1.plot(cycles, actual_cap, label='B0006 Actual Capacity', color='blue', linewidth=2)
ax1.plot(cycles, pred_cap, label='Multi-Cell CNN-LSTM Predicted Capacity', color='red', linestyle='--', linewidth=2)
ax1.axhline(y=EOL_THRESHOLD, color='black', linestyle=':', linewidth=1.5, label=f'EOL Threshold ({EOL_THRESHOLD:.2f} Ah)')
ax1.axvline(x=actual_eol_cycle, color='blue', linestyle='-.', alpha=0.7, label=f'Actual EOL (Cycle {actual_eol_cycle})')
ax1.axvline(x=pred_eol_cycle, color='red', linestyle='-.', alpha=0.7, label=f'Predicted EOL (Cycle {pred_eol_cycle})')
ax1.set_title('B0006 End-of-Life (EOL) Threshold Detection & Remaining Useful Life (RUL)')
ax1.set_ylabel('Capacity (Ah)')
ax1.legend()
ax1.grid(True, linestyle='--', alpha=0.7)

# Bottom Plot: Remaining Useful Life (RUL) Countdown
ax2.plot(cycles, actual_rul_curve, label='Actual RUL (Cycles Remaining)', color='blue', linewidth=2)
ax2.plot(cycles, pred_rul_curve, label='Predicted RUL (Cycles Remaining)', color='red', linestyle='--', linewidth=2)
ax2.set_xlabel('Discharge Cycle Number')
ax2.set_ylabel('Remaining Useful Life (Cycles)')
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
