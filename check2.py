import os
import glob
import scipy.io
import numpy as np

DATA_DIR = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet"

def find_mat_file(base_dir, cell_id):
    matches = glob.glob(os.path.join(base_dir, "**", f"{cell_id}.mat"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"{cell_id}.mat not found anywhere under {base_dir}")
    return matches[0]

def load_nasa_battery(mat_path, num_steps=200):
    mat = scipy.io.loadmat(mat_path)
    cell_name = os.path.splitext(os.path.basename(mat_path))[0]
    cycles = mat[cell_name]['cycle'][0, 0][0]
    
    X_list, cap_list = [], []
    for c in cycles:
        if str(c['type'][0]) == 'discharge':
            data = c['data'][0, 0]
            t = data['Time'][0].astype(np.float64)
            v = data['Voltage_measured'][0].astype(np.float64)
            i = data['Current_measured'][0].astype(np.float64)
            temp = data['Temperature_measured'][0].astype(np.float64)
            cap = float(data['Capacity'][0][0])
            
            t_uniform = np.linspace(t[0], t[-1], num_steps)
            v_interp = np.interp(t_uniform, t, v)
            i_interp = np.interp(t_uniform, t, i)
            temp_interp = np.interp(t_uniform, t, temp)
            
            X_list.append(np.stack([v_interp, i_interp, temp_interp], axis=-1))
            cap_list.append(cap)
            
    X = np.array(X_list, dtype=np.float32)
    caps = np.array(cap_list, dtype=np.float32)
    soh = caps / caps[0]
    return X, soh, caps

for bid in ['B0025', 'B0026', 'B0027', 'B0028']:
    file_path = find_mat_file(DATA_DIR, bid)
    X_b, soh_b, caps_b = load_nasa_battery(file_path)
    eol_cycles = np.where(soh_b <= 0.70)[0]
    eol_status = f"Hits 70% EOL at cycle {eol_cycles[0] + 1}" if len(eol_cycles) > 0 else "Does NOT reach 70% EOL"
    print(f"{bid}: {len(soh_b)} cycles | C0 = {caps_b[0]:.3f} Ah | End SOH = {soh_b[-1]*100:.2f}% | {eol_status}")
