import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping

# Enforce strict deterministic execution across CPU/GPU ops
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = SCRIPT_DIR
DATA_DIR = os.getenv("NASA_DATA_DIR", os.path.join(SCRIPT_DIR, "data"))
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

def build_model(variant):
    tf.keras.utils.set_random_seed(42)
    if variant == "1D-CNN Only":
        model = Sequential([
            Input(shape=(200, 3)),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(32, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
    elif variant == "Standalone LSTM":
        model = Sequential([
            Input(shape=(200, 3)),
            LSTM(64, return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dense(32, activation='relu'),
            Dense(1)
        ])
    elif variant == "Hybrid CNN-LSTM":
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

# ==============================================================================
# STEP 1: SINGLE DETERMINISTIC 4-FOLD LOOCV RUN (SOURCE OF TRUTH)
# ==============================================================================
ALL_CELLS = ['B0005', 'B0006', 'B0007', 'B0018']
variants = ["1D-CNN Only", "Standalone LSTM", "Hybrid CNN-LSTM"]

data_cache = {cid: load_battery_normalized_soh(cid) for cid in ALL_CELLS}
predictions_cache = {cid: {} for cid in ALL_CELLS}
metrics_summary = {v: {"rmse": [], "mae": [], "r2": []} for v in variants}
hybrid_models = {}

for idx, test_cid in enumerate(ALL_CELLS):
    print(f"[{idx+1}/4] Running deterministic fold for unseen {test_cid}...")
    train_cids = [c for c in ALL_CELLS if c != test_cid]
    X_train_raw = np.concatenate([data_cache[c][0] for c in train_cids], axis=0)
    y_train = np.concatenate([data_cache[c][1] for c in train_cids], axis=0)
    X_test_raw, y_test, _, _ = data_cache[test_cid]

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train_raw.reshape(-1, 3)).reshape(X_train_raw.shape)
    X_test = scaler.transform(X_test_raw.reshape(-1, 3)).reshape(X_test_raw.shape)

    for v in variants:
        model = build_model(v)
        es = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model.fit(X_train, y_train, validation_split=0.15, epochs=120, batch_size=32, callbacks=[es], verbose=0)
        y_pred = model(X_test, training=False).numpy().flatten()

        predictions_cache[test_cid][v] = y_pred
        rmse = np.sqrt(mean_squared_error(y_test, y_pred)) * 100
        mae = mean_absolute_error(y_test, y_pred) * 100
        r2 = r2_score(y_test, y_pred)

        metrics_summary[v]["rmse"].append(rmse)
        metrics_summary[v]["mae"].append(mae)
        metrics_summary[v]["r2"].append(r2)

        if v == "Hybrid CNN-LSTM":
            hybrid_models[test_cid] = (model, X_test)

# ==============================================================================
# FIGURE 1: 4-FOLD SOH TRAJECTORY GRID (MATCHES TABLE 2 EXACTLY)
# ==============================================================================
colors = {"1D-CNN Only": "#E69F00", "Standalone LSTM": "#009E73", "Hybrid CNN-LSTM": "#D55E00"}
styles = {"1D-CNN Only": ":", "Standalone LSTM": "-.", "Hybrid CNN-LSTM": "--"}

fig1, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for idx, test_cid in enumerate(ALL_CELLS):
    _, y_test, _, _ = data_cache[test_cid]
    ax = axes[idx]
    cycles = np.arange(1, len(y_test) + 1)
    ax.plot(cycles, y_test * 100, label="Ground Truth SOH", color="#111111", linewidth=2.2)

    for v in variants:
        y_pred = predictions_cache[test_cid][v]
        rmse = metrics_summary[v]["rmse"][idx]
        ax.plot(cycles, y_pred * 100, label=f"{v} (RMSE: {rmse:.2f}%)",
                color=colors[v], linestyle=styles[v], linewidth=1.8)

    ax.axhline(70.0, color="#555555", linestyle="--", linewidth=1.0, alpha=0.7, label="70% EOL Threshold")
    ax.set_title(f"Unseen Test Cell: {test_cid}", fontweight="bold")
    ax.set_ylabel("State of Health (%)")
    if idx >= 2:
        ax.set_xlabel("Discharge Cycle Number")
    ax.set_ylim(55, 105)
    ax.grid(True, linestyle="--", alpha=0.45)
    ax.legend(loc="lower left", frameon=True, framealpha=0.9)

fig1.suptitle("4-Fold Leave-One-Battery-Out Cross-Validation: SOH Trajectory Comparison", fontsize=13, fontweight="bold")
fig1.tight_layout(rect=[0, 0, 1, 0.96])
fig1.savefig(os.path.join(PROJECT_DIR, "fig1_4fold_soh_grid.png"))
plt.close(fig1)

# ==============================================================================
# FIGURE 2: 4-FOLD ABLATION BARS & ZERO-SHOT PULSED STRESS TEST
# ==============================================================================
fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.8))
x_pos = np.arange(len(variants))
width = 0.34
avg_rmses = [np.mean(metrics_summary[v]["rmse"]) for v in variants]
avg_maes = [np.mean(metrics_summary[v]["mae"]) for v in variants]

bars1 = ax_left.bar(x_pos - width/2, avg_rmses, width, label="4-Fold Avg RMSE (%)", color="#0072B2")
bars2 = ax_left.bar(x_pos + width/2, avg_maes, width, label="4-Fold Avg MAE (%)", color="#56B4E9")
for bar in bars1:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8.5, fontweight='bold')
for bar in bars2:
    ax_left.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15, f"{bar.get_height():.2f}%", ha='center', va='bottom', fontsize=8.5)

ax_left.set_xticks(x_pos)
ax_left.set_xticklabels(["1D-CNN\nOnly", "Standalone\nLSTM", "Hybrid CNN-LSTM\n(Proposed)"])
ax_left.set_ylabel("Prediction Error (%)")
ax_left.set_title("(a) 4-Fold Cross-Battery Error Comparison", fontweight="bold")
ax_left.set_ylim(0, max(avg_rmses) + 1.8)
ax_left.grid(axis="y", linestyle="--", alpha=0.45)
ax_left.legend()

norm_model = load_model(os.path.join(PROJECT_DIR, "saved_soh_model_norm", "cnn_lstm_norm_soh.keras"))
norm_scaler = joblib.load(os.path.join(PROJECT_DIR, "saved_soh_model_norm", "scaler_X_norm.joblib"))

stress_cells = ['B0025', 'B0026', 'B0027', 'B0028']
true_means, pred_means, stress_rmses = [], [], []
for bid in stress_cells:
    Xr, yt, _, _ = load_battery_normalized_soh(bid)
    Xs = norm_scaler.transform(Xr.reshape(-1, 3)).reshape(Xr.shape)
    yp = norm_model(Xs, training=False).numpy().flatten()
    true_means.append(yt.mean() * 100)
    pred_means.append(yp.mean() * 100)
    stress_rmses.append(np.sqrt(mean_squared_error(yt, yp)) * 100)

xs = np.arange(len(stress_cells))
b_t = ax_right.bar(xs - width/2, true_means, width, label="True Mean SOH (%)", color="#009E73")
b_p = ax_right.bar(xs + width/2, pred_means, width, label="Zero-Shot Pred SOH (%)", color="#CC79A7")
for bar in b_t:
    ax_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8)
for bar in b_p:
    ax_right.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.0, f"{bar.get_height():.1f}%", ha='center', va='bottom', fontsize=8, fontweight='bold')

ax_right.set_xticks(xs)
ax_right.set_xticklabels(stress_cells)
ax_right.set_ylabel("Mean State of Health (%)")
ax_right.set_title("(b) Zero-Shot Pulsed-Load Domain Shift (4A Square-Wave)", fontweight="bold")
ax_right.set_ylim(0, 112)
ax_right.grid(axis="y", linestyle="--", alpha=0.45)
ax_right.legend(loc="lower right")

fig2.tight_layout()
fig2.savefig(os.path.join(PROJECT_DIR, "fig2_ablation_and_stress.png"))
plt.close(fig2)

# ==============================================================================
# FIGURE 3: UNSEEN B0006 RUL COUNTDOWN (USES EXACT B0006 CACHE)
# ==============================================================================
_, y_b6_true, _, _ = data_cache["B0006"]
y_b6_pred = predictions_cache["B0006"]["Hybrid CNN-LSTM"]
rmse_b6_hybrid = metrics_summary["Hybrid CNN-LSTM"]["rmse"][1]
c_b6 = np.arange(1, len(y_b6_true) + 1)

def get_eol(arr, thresh=0.70):
    idx = np.where(arr <= thresh)[0]
    return int(idx[0] + 1) if len(idx) > 0 else len(arr)

act_eol = get_eol(y_b6_true, 0.70)
prd_eol = get_eol(y_b6_pred, 0.70)

fig3, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
ax_top.plot(c_b6, y_b6_true * 100, label="Actual SOH (%)", color="#111111", linewidth=2.2)
ax_top.plot(c_b6, y_b6_pred * 100, label=f"Hybrid CNN-LSTM (RMSE: {rmse_b6_hybrid:.2f}%)", color="#D55E00", linestyle="--", linewidth=2.0)
ax_top.fill_between(c_b6, y_b6_true * 100, y_b6_pred * 100, color="#D55E00", alpha=0.15, label="Prediction Error Band")
ax_top.axhline(70.0, color="#444444", linestyle=":", linewidth=1.3, label="70% EOL Threshold")
ax_top.axvline(act_eol, color="#111111", linestyle="-.", alpha=0.75, label=f"Actual EOL (Cycle {act_eol})")
ax_top.axvline(prd_eol, color="#D55E00", linestyle="-.", alpha=0.75, label=f"Predicted EOL (Cycle {prd_eol})")
ax_top.set_ylabel("State of Health (%)")
ax_top.set_title("Unseen Cell B0006: Hybrid CNN-LSTM SOH Prognostics & RUL Countdown", fontweight="bold")
ax_top.grid(True, linestyle="--", alpha=0.45)
ax_top.legend(loc="lower left", ncol=2)

act_rul = np.maximum(0, act_eol - c_b6)
prd_rul = np.maximum(0, prd_eol - c_b6)
ax_bot.plot(c_b6, act_rul, label="Actual RUL (Cycles)", color="#111111", linewidth=2.2)
ax_bot.plot(c_b6, prd_rul, label="Predicted RUL (Cycles)", color="#D55E00", linestyle="--", linewidth=2.0)
ax_bot.fill_between(c_b6, act_rul, prd_rul, color="#D55E00", alpha=0.15, label=f"EOL Horizon Offset ({abs(act_eol - prd_eol)} cycles)")
ax_bot.set_xlabel("Discharge Cycle Number")
ax_bot.set_ylabel("Remaining Useful Life (Cycles)")
ax_bot.grid(True, linestyle="--", alpha=0.45)
ax_bot.legend(loc="upper right")

fig3.tight_layout()
fig3.savefig(os.path.join(PROJECT_DIR, "fig3_b0006_rul_countdown.png"))
plt.close(fig3)

# ==============================================================================
# FIGURE 4: INTRA-CYCLE AGING PHYSICS (CELL B0006)
# ==============================================================================
X_b6 = data_cache["B0006"][0]
norm_steps = np.linspace(0, 1, 200)
selected_indices = [0, 79, 159]
cycle_labels = [
    f"Cycle 1 (Fresh: {y_b6_true[0]*100:.1f}% SOH)",
    f"Cycle 80 (Mid-Life: {y_b6_true[79]*100:.1f}% SOH)",
    f"Cycle 160 (Aged: {y_b6_true[159]*100:.1f}% SOH)"
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
fig4.savefig(os.path.join(PROJECT_DIR, "fig4_intracycle_aging_physics.png"))
plt.close(fig4)

# ==============================================================================
# FIGURE 5: 4-FOLD PARITY PLOT (USES EXACT SAME PREDICTIONS CACHE)
# ==============================================================================
cell_colors = {"B0005": "#0072B2", "B0006": "#D55E00", "B0007": "#009E73", "B0018": "#CC79A7"}
cell_markers = {"B0005": "o", "B0006": "s", "B0007": "^", "B0018": "D"}

fig5, ax_p = plt.subplots(figsize=(6.8, 6.2))
ideal_line = np.linspace(55, 105, 200)
ax_p.plot(ideal_line, ideal_line, color="#111111", linestyle="-", linewidth=1.8, label="Ideal Prediction (y = x)")
ax_p.fill_between(ideal_line, ideal_line - 5.0, ideal_line + 5.0, color="#56B4E9", alpha=0.20, label="±5% SOH Tolerance Band")

all_true, all_pred = [], []
for idx, test_cid in enumerate(ALL_CELLS):
    y_test = data_cache[test_cid][1]
    y_pred = predictions_cache[test_cid]["Hybrid CNN-LSTM"]
    rmse_c = metrics_summary["Hybrid CNN-LSTM"]["rmse"][idx]
    all_true.extend(y_test * 100)
    all_pred.extend(y_pred * 100)

    ax_p.scatter(
        y_test * 100, y_pred * 100,
        label=f"Unseen {test_cid} (RMSE: {rmse_c:.2f}%)",
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
fig5.savefig(os.path.join(PROJECT_DIR, "fig5_parity_error_cone.png"))
plt.close(fig5)

# ==============================================================================
# FIGURE 6: EMPIRICAL EQUATION VS. HYBRID CNN-LSTM (USES EXACT B0006 CACHE)
# ==============================================================================
y_tr_eq = np.concatenate([data_cache[c][1] for c in ['B0005', 'B0007', 'B0018']], axis=0)
k_tr_eq = np.concatenate([np.arange(1, len(data_cache[c][1]) + 1) for c in ['B0005', 'B0007', 'B0018']], axis=0)

def empirical_soh_eq(k, a, b):
    return 1.0 - a * np.power(k, b)

popt, _ = curve_fit(empirical_soh_eq, k_tr_eq, y_tr_eq, p0=[0.001, 1.0], bounds=([0, 0.1], [1.0, 3.0]))
y_pred_eq = empirical_soh_eq(c_b6, popt[0], popt[1])
rmse_eq = np.sqrt(mean_squared_error(y_b6_true, y_pred_eq)) * 100
r2_eq = r2_score(y_b6_true, y_pred_eq)

plt.figure(figsize=(10, 5.2))
plt.plot(c_b6, y_b6_true * 100, label="Ground Truth SOH (C_k / C_0)", color="#111111", linewidth=2.3)
plt.plot(c_b6, y_pred_eq * 100, label=f"Empirical Power-Law Eq. (RMSE: {rmse_eq:.2f}%)", color="#0072B2", linestyle=":", linewidth=2.1)
plt.plot(c_b6, y_b6_pred * 100, label=f"Hybrid CNN-LSTM (RMSE: {rmse_b6_hybrid:.2f}%)", color="#D55E00", linestyle="--", linewidth=2.1)
plt.axhline(70.0, color="#555555", linestyle="-.", linewidth=1.1, label="70% EOL Threshold")
plt.title("Unseen Cell B0006: Empirical Degradation Equation vs. Hybrid CNN-LSTM", fontweight="bold")
plt.xlabel("Discharge Cycle Number (k)")
plt.ylabel("State of Health (%)")
plt.grid(True, linestyle="--", alpha=0.45)
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig(os.path.join(PROJECT_DIR, "fig6_equation_vs_cnnlstm.png"))
plt.close()

# ==============================================================================
# PRINT SYNCHRONIZED LATEX TABLE 2 & PAPER SUMMARY NUMBERS
# ==============================================================================
print("\n" + "="*80)
print("COPY THESE EXACT NUMBERS INTO TABLE 2 & ABSTRACT OF main.tex:")
print("="*80)
for v in variants:
    print(f"\n--- {v} ---")
    for idx, cid in enumerate(ALL_CELLS):
        r = metrics_summary[v]["rmse"][idx]
        m = metrics_summary[v]["mae"][idx]
        r2 = metrics_summary[v]["r2"][idx]
        print(f" & \\texttt{{{cid}}} & {r:.2f} & {m:.2f} & {r2:.4f} \\\\")
    print(f" & \\textbf{{4-Fold Mean}} & \\textbf{{{np.mean(metrics_summary[v]['rmse']):.2f}}} & \\textbf{{{np.mean(metrics_summary[v]['mae']):.2f}}} & \\textbf{{{np.mean(metrics_summary[v]['r2']):.4f}}} \\\\")

print(f"\nEmpirical Eq on B0006: SOH(k) = 1 - {popt[0]:.6f} * k^({popt[1]:.4f}) | RMSE: {rmse_eq:.2f}% | R^2: {r2_eq:.4f}")
print(f"Zero-Shot Pulsed Stress Mean RMSE (B0025-B0028): {np.mean(stress_rmses):.2f}%")
print("All 6 figures regenerated deterministically from the exact same predictions!")
