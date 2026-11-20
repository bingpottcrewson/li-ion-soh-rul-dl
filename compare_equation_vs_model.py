import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"
PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"

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

# 1. Load training cells (B0005, B0007, B0018) and unseen test cell (B0006)
train_ids = ['B0005', 'B0007', 'B0018']
X_tr_list, y_tr_list, k_tr_list = [], [], []
for cid in train_ids:
    Xb, yb, _, _ = load_battery_normalized_soh(cid)
    X_tr_list.append(Xb)
    y_tr_list.append(yb)
    k_tr_list.append(np.arange(1, len(yb) + 1))

X_train_raw = np.concatenate(X_tr_list, axis=0)
y_train = np.concatenate(y_tr_list, axis=0)
k_train = np.concatenate(k_tr_list, axis=0)

X_test_raw, y_test, _, _ = load_battery_normalized_soh('B0006')
k_test = np.arange(1, len(y_test) + 1)

# 2. Fit Empirical Power-Law SOH Equation: SOH(k) = 1 - a * (k ^ b)
def empirical_soh_eq(k, a, b):
    return 1.0 - a * np.power(k, b)

popt, _ = curve_fit(empirical_soh_eq, k_train, y_train, p0=[0.001, 1.0], bounds=([0, 0.1], [1.0, 3.0]))
a_fit, b_fit = popt
y_pred_eq = empirical_soh_eq(k_test, a_fit, b_fit)

# 3. Train Hybrid CNN-LSTM Model on the exact same training cells
tf.keras.utils.set_random_seed(42)
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
X_test = scaler.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

model = Sequential([
    Input(shape=(200, 3)),
    Conv1D(64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Conv1D(32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    LSTM(50),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.fit(X_train, y_train, validation_split=0.15, epochs=120, batch_size=32, callbacks=[es], verbose=0)
y_pred_dl = model(X_test, training=False).numpy().flatten()

# 4. Print Overall & Cycle-by-Cycle Comparison Table
rmse_eq = np.sqrt(mean_squared_error(y_test, y_pred_eq)) * 100
mae_eq = mean_absolute_error(y_test, y_pred_eq) * 100
r2_eq = r2_score(y_test, y_pred_eq)

rmse_dl = np.sqrt(mean_squared_error(y_test, y_pred_dl)) * 100
mae_dl = mean_absolute_error(y_test, y_pred_dl) * 100
r2_dl = r2_score(y_test, y_pred_dl)

print("="*78)
print(f"FITTED EMPIRICAL EQUATION: SOH(k) = 1 - {a_fit:.6f} * k^({b_fit:.4f})")
print("="*78)
print(f"{'Method':<28} | {'SOH RMSE (%)':<13} | {'SOH MAE (%)':<12} | {'R^2 Score':<10}")
print("-"*78)
print(f"{'Empirical Fade Equation':<28} | {rmse_eq:<13.2f} | {mae_eq:<12.2f} | {r2_eq:<10.4f}")
print(f"{'Hybrid CNN-LSTM (Proposed)':<28} | {rmse_dl:<13.2f} | {mae_dl:<12.2f} | {r2_dl:<10.4f}")
print("="*78)

print("\nCYCLE-BY-CYCLE VALUE COMPARISON ON UNSEEN B0006:")
print(f"{'Cycle':<7} | {'Ground Truth (C_k/C_0)':<22} | {'Empirical Eq.':<15} | {'Hybrid CNN-LSTM':<17} | {'Eq. Err':<9} | {'DL Err':<9}")
print("-"*86)
for idx in [0, 24, 49, 74, 99, 124, 149, 167]:
    gt = y_test[idx] * 100
    eq_v = y_pred_eq[idx] * 100
    dl_v = y_pred_dl[idx] * 100
    print(f"{idx+1:<7} | {gt:<21.2f}% | {eq_v:<14.2f}% | {dl_v:<16.2f}% | {abs(gt-eq_v):<8.2f}% | {abs(gt-dl_v):<8.2f}%")

# 5. Save Comparison Plot
plt.figure(figsize=(10, 5.2))
plt.plot(k_test, y_test * 100, label="Ground Truth SOH (C_k / C_0)", color="#111111", linewidth=2.3)
plt.plot(k_test, y_pred_eq * 100, label=f"Empirical Power-Law Eq. (RMSE: {rmse_eq:.2f}%)", color="#0072B2", linestyle=":", linewidth=2.1)
plt.plot(k_test, y_pred_dl * 100, label=f"Hybrid CNN-LSTM (RMSE: {rmse_dl:.2f}%)", color="#D55E00", linestyle="--", linewidth=2.1)
plt.axhline(70.0, color="#555555", linestyle="-.", linewidth=1.1, label="70% EOL Threshold")
plt.title("Unseen Cell B0006: Empirical Degradation Equation vs. Hybrid CNN-LSTM", fontweight="bold")
plt.xlabel("Discharge Cycle Number (k)")
plt.ylabel("State of Health (%)")
plt.grid(True, linestyle="--", alpha=0.45)
plt.legend(loc="lower left")
plt.tight_layout()
out_path = os.path.join(PROJECT_DIR, "fig6_equation_vs_cnnlstm.png")
plt.savefig(out_path)
plt.close()
print("\nSaved comparison plot to:", out_path)
