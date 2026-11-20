import os

# Use r"..." so Python treats Windows backslashes literally
DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"

# 1. In the inspection loop:
for bid in ['B0025', 'B0026', 'B0027', 'B0028']:
    file_path = os.path.join(DATA_DIR, f"{bid}.mat")
    X_b, soh_b, caps_b = load_nasa_battery(file_path)
    eol_cycles = np.where(soh_b <= 0.70)[0]
    eol_status = f"Hits 70% EOL at cycle {eol_cycles[0] + 1}" if len(eol_cycles) > 0 else "Does NOT reach 70% EOL"
    print(f"{bid}: {len(soh_b)} cycles | C0 = {caps_b[0]:.3f} Ah | End SOH = {soh_b[-1]*100:.2f}% | {eol_status}")

# 2. In the multi-protocol loader loop:
all_cells = {}
for cid in train_ids + test_ids:
    file_path = os.path.join(DATA_DIR, f"{cid}.mat")
    all_cells[cid] = load_nasa_battery(file_path)
