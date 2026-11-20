import os, glob

PROJECT_DIR = r"C:\Users\krish\OneDrive\Documents\study\s5\deep learning\miniproject"
print("PNG files found:", glob.glob(os.path.join(PROJECT_DIR, "*.png")))
