import os

PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"

for script in ['trainnormalisedsoh.py', 'ablation.py']:
    print(f"\n=== {script} (top 60 lines) ===")
    with open(os.path.join(PROJECT_DIR, script), 'r', encoding='utf-8', errors='ignore') as f:
        print("".join(f.readlines()[:60]))
