import os

PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"

for name in ['saved_soh_model', 'saved_soh_model_multi', 'saved_soh_model_norm']:
    p = os.path.join(PROJECT_DIR, name)
    print(name, "-> is_dir:", os.path.isdir(p), "| contents:", os.listdir(p) if os.path.isdir(p) else os.path.getsize(p))

# Also print the last 25 lines of trainnormalisedsoh.py and multibattery.py to see how they saved & scaled!
for script in ['trainnormalisedsoh.py', 'multibattery.py', 'ablation.py']:
    print(f"\n--- {script} (tail) ---")
    with open(os.path.join(PROJECT_DIR, script), 'r', encoding='utf-8', errors='ignore') as f:
        print("".join(f.readlines()[-25:]))
