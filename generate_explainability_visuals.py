import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.dpi": 300
})

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

# ==============================================================================
# FIGURE 4: INTRA-CYCLE AGING PHYSICS (CYCLE 1 vs. CYCLE 80 vs. CYCLE 160)
# ==============================================================================
X_b6, soh_b6, _, _ = load_battery_normalized_soh('B0006', fixed_length=200)
norm_steps = np.linspace(0, 1, 200)

selected_indices = [0, 79, 159]  # Cycle 1, Cycle 80, Cycle 160
cycle_labels = [
    f"Cycle 1 (Fresh: {soh_b6[0]*100:.1f}% SOH)",
    f"Cycle 80 (Mid-Life: {soh_b6[79]*100:.1f}% SOH)",
    f"Cycle 160 (Aged: {soh_b6[159]*100:.1f}% SOH)"
]
curve_colors = ["#0072B2", "#E69F00", "#D55E00"]
curve_styles = ["-", "--", "-."]

fig4, axes = plt.subplots(1, 3, figsize=(13.5, 4.3))
channel_titles = ["(a) Terminal Voltage Sag", "(b) Discharge Current Profile", "(c) Surface Temperature Rise"]
y_labels = ["Voltage (V)", "Current (A)", "Temperature (°C)"]

for ch in range(3):
    ax = axes[ch]
    for idx, lbl, col, ls in zip(selected_indices, cycle_labels, curve_colors, curve_styles):
        ax.plot(norm_steps, X_b6[idx, :, ch], label=lbl, color=col, linestyle=ls, linewidth=2.0)
    ax.set_title(channel_titles[ch], fontweight="bold")
    ax.set_xlabel("Normalized Discharge Time (0 to 1)")
    ax.set_ylabel(y_labels[ch])
    ax.grid(True, linestyle="--", alpha=0.45)
    if ch == 0:
        ax.legend(loc="lower left", frameon=True, framealpha=0.92)

fig4.suptitle("Electrochemical Degradation Signatures Captured by 200-Step Resampling (Cell B0006)", fontsize=12.5, fontweight="bold")
fig4.tight_layout(rect=[0, 0, 1, 0.94])
fig4_path = os.path.join(PROJECT_DIR, "fig4_intracycle_aging_physics.png")
fig4.savefig(fig4_path)
plt.close(fig4)
print("Saved:", fig4_path)

# ==============================================================================
# FIGURE 5: 4-FOLD PARITY PLOT (+-5% TOLERANCE CONE) & BMS EDGE BENCHMARK
# ==============================================================================
def build_hybrid_model():
    tf.keras.utils.set_random_seed(42)
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
    return model

ALL_CELLS = ['B0005', 'B0006', 'B0007', 'B0018']
cell_colors = {"B0005": "#0072B2", "B0006": "#D55E00", "B0007": "#009E73", "B0018": "#CC79A7"}
cell_markers = {"B0005": "o", "B0006": "s", "B0007": "^", "B0018": "D"}

data_cache = {cid: load_battery_normalized_soh(cid) for cid in ALL_CELLS}

fig5, ax_p = plt.subplots(figsize=(6.8, 6.2))
ideal_line = np.linspace(55, 105, 200)

# Shaded +-5% SOH error cone and ideal diagonal
ax_p.plot(ideal_line, ideal_line, color="#111111", linestyle="-", linewidth=1.8, label="Ideal Prediction (y = x)")
ax_p.fill_between(ideal_line, ideal_line - 5.0, ideal_line + 5.0, color="#56B4E9", alpha=0.20, label="±5% SOH Tolerance Band")

all_true, all_pred = [], []
for test_cid in ALL_CELLS:
    train_cids = [c for c in ALL_CELLS if c != test_cid]
    X_train_raw = np.concatenate([data_cache[c][0] for c in train_cids], axis=0)
    y_train = np.concatenate([data_cache[c][1] for c in train_cids], axis=0)
    X_test_raw, y_test, _, _ = data_cache[test_cid]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

    model = build_hybrid_model()
    es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    model.fit(X_train, y_train, validation_split=0.15, epochs=120, batch_size=32, callbacks=[es], verbose=0)
    y_pred = model(X_test, training=False).numpy().flatten()

    all_true.extend(y_test * 100)
    all_pred.extend(y_pred * 100)

    ax_p.scatter(
        y_test * 100, y_pred * 100,
        label=f"Unseen {test_cid} (n={len(y_test)})",
        color=cell_colors[test_cid], marker=cell_markers[test_cid],
        s=32, alpha=0.80, edgecolors="white", linewidth=0.4
    )

all_true = np.array(all_true)
all_pred = np.array(all_pred)
within_5 = np.mean(np.abs(all_true - all_pred) <= 5.0) * 100

ax_p.set_xlim(55, 105)
ax_p.set_ylim(55, 105)
ax_p.set_xlabel("Actual State of Health (%)")
ax_p.set_ylabel("Predicted State of Health (%)")
ax_p.set_title(f"4-Fold Unseen SOH Parity Plot ({within_5:.1f}% Cycles Within ±5% Band)", fontweight="bold")
ax_p.grid(True, linestyle="--", alpha=0.45)
ax_p.legend(loc="upper left", frameon=True, framealpha=0.92)

fig5.tight_layout()
fig5_path = os.path.join(PROJECT_DIR, "fig5_parity_error_cone.png")
fig5.savefig(fig5_path)
plt.close(fig5)
print("Saved:", fig5_path)

# ==============================================================================
# EMBEDDED BMS FEASIBILITY METRICS (FOR SLIDE CALLOUT BOX)
# ==============================================================================
total_params = model.count_params()
fp32_kb = (total_params * 4) / 1024.0
int8_kb = total_params / 1024.0

sample_cycle = X_test[:1]
_ = model(sample_cycle, training=False)  # Warm-up pass
t0 = time.perf_counter()
for _ in range(200):
    _ = model(sample_cycle, training=False)
latency_ms = ((time.perf_counter() - t0) / 200.0) * 1000.0

print("\n" + "="*62)
print("EMBEDDED BMS FEASIBILITY METRICS (HYBRID CNN-LSTM)")
print("="*62)
print(f"Total Trainable Parameters : {total_params:,}")
print(f"Model Footprint (FP32)     : {fp32_kb:.2f} KB")
print(f"Model Footprint (INT8)     : {int8_kb:.2f} KB")
print(f"Single-Cycle Latency       : {latency_ms:.3f} ms")
print("="*62)
