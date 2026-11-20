import pandas as pd
import matplotlib.pyplot as plt

# 1. Load the metadata index
metadata = pd.read_csv('metadata.csv')

# 2. Filter for Battery B0005 and only 'discharge' cycles 
# (Capacity is typically recorded during discharge in this dataset)
b0005_discharge = metadata[(metadata['battery_id'] == 'B0005') & (metadata['type'] == 'discharge')].copy()

# 3. Create a sequential cycle number based on the filtered rows
b0005_discharge['cycle'] = range(1, len(b0005_discharge) + 1)

# 4. Ensure the Capacity column is read as numeric data
b0005_discharge['Capacity'] = pd.to_numeric(b0005_discharge['Capacity'], errors='coerce')

# 5. Plot the cycle-to-cycle Capacity Fade
plt.figure(figsize=(10, 5))
plt.plot(b0005_discharge['cycle'], b0005_discharge['Capacity'], marker='o', linestyle='-', color='b', markersize=4)
plt.axhline(y=1.4, color='r', linestyle='--', label='End of Life (EOL) Threshold (1.4 Ah)')

plt.title('Battery B0005 Capacity Fade Over Cycles')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('Capacity (Ah)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
