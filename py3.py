import kagglehub

# Download latest version
path = kagglehub.dataset_download("ckskaggle/li-ion-battery-dataset-from-nasa-pcoe")

print("Path to dataset files:", path)
