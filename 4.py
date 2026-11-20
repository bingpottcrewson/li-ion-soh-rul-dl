from scipy.io import loadmat

# 1. Define the path and load the file into 'mat_contents'
file_path = r"C:\Users\krish\.cache\kagglehub\datasets\ckskaggle\li-ion-battery-dataset-from-nasa-pcoe\versions\5\Battery_DataSet\Battery_DataSet\B0005.mat"
mat_contents = loadmat(file_path)

# 2. Extract the core B0005 data payload
battery_data = mat_contents['B0005']

# 3. Access the nested 'cycle' array
cycles = battery_data[0][0]['cycle'][0]

print(f"Total cycles recorded for B0005: {len(cycles)}")

# Initialize empty lists to hold the separated data
charge_data = []
discharge_data = []
impedance_data = []

# Loop through all 616 cycles
for i in range(len(cycles)):
    # Extract the string representing the cycle type
    cycle_type = cycles[i]['type'][0]
    
    # Sort the data into the correct list
    if cycle_type == 'charge':
        charge_data.append(cycles[i])
    elif cycle_type == 'discharge':
        discharge_data.append(cycles[i])
    elif cycle_type == 'impedance':
        impedance_data.append(cycles[i])

# Print the breakdown
print(f"Total Charge operations: {len(charge_data)}")
print(f"Total Discharge operations: {len(discharge_data)}")
print(f"Total Impedance operations: {len(impedance_data)}")

import matplotlib.pyplot as plt

# 1. Initialize lists for cycle index and capacity values
cycle_numbers = []
capacities = []

# 2. Loop through the 168 discharge operations
for i, cycle in enumerate(discharge_data):
    # Dig into the nested 'data' array to extract the Capacity value
    capacity = cycle['data'][0][0]['Capacity'][0][0]
    
    cycle_numbers.append(i + 1)
    capacities.append(capacity)

# 3. Plot the battery capacity degradation curve
plt.figure(figsize=(10, 6))
plt.plot(cycle_numbers, capacities, linestyle='-', marker='o', markersize=3, color='b')
plt.title('B0005 Battery Capacity Degradation (State of Health)')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('Capacity (Ampere-hours)')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Extract the data payload for the very first discharge cycle (Index 0)
first_discharge_cycle = discharge_data[0]['data'][0][0]

# Isolate the specific sensor arrays
voltage_measured = first_discharge_cycle['Voltage_measured'][0]
current_measured = first_discharge_cycle['Current_measured'][0]
temperature_measured = first_discharge_cycle['Temperature_measured'][0]
time_steps = first_discharge_cycle['Time'][0]

# Print the shapes to understand what the CNN will process
print(f"Data points in Cycle 1:")
print(f"Voltage array shape: {voltage_measured.shape}")
print(f"Current array shape: {current_measured.shape}")
print(f"Temperature array shape: {temperature_measured.shape}")
print(f"Time array shape: {time_steps.shape}")

import numpy as np
from scipy.interpolate import interp1d

# 1. Define the uniform sequence length for the neural network
FIXED_LENGTH = 200
X_features = []

# 2. Loop through all 168 discharge cycles
for cycle in discharge_data:
    # Extract the raw 1D sensor arrays
    voltage = cycle['data'][0][0]['Voltage_measured'][0]
    current = cycle['data'][0][0]['Current_measured'][0]
    temp = cycle['data'][0][0]['Temperature_measured'][0]
    time_steps = cycle['data'][0][0]['Time'][0]
    
    # Normalize the time array to a 0.0 - 1.0 scale to create a uniform X-axis
    time_norm = (time_steps - time_steps[0]) / (time_steps[-1] - time_steps[0])
    fixed_time = np.linspace(0, 1, FIXED_LENGTH)
    
    # 3. Interpolate the sensor data to exactly 200 steps
    v_interp = interp1d(time_norm, voltage, kind='linear')(fixed_time)
    c_interp = interp1d(time_norm, current, kind='linear')(fixed_time)
    t_interp = interp1d(time_norm, temp, kind='linear')(fixed_time)
    
    # Stack the three variables into a (200, 3) matrix for this specific cycle
    cycle_matrix = np.column_stack((v_interp, c_interp, t_interp))
    X_features.append(cycle_matrix)

# 4. Convert the lists to final NumPy arrays
X_features = np.array(X_features)
y_target = np.array(capacities)

print(f"Input Feature Tensor (X) Shape: {X_features.shape}")
print(f"Target Array (Y) Shape: {y_target.shape}")

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Normalize the Input Features (X)
# MinMaxScaler requires 2D arrays, so we flatten X, scale it, and reshape it back to 3D
scaler_X = MinMaxScaler()
X_reshaped = X_features.reshape(-1, 3)
X_scaled_reshaped = scaler_X.fit_transform(X_reshaped)
X_scaled = X_scaled_reshaped.reshape(X_features.shape)

# 2. Normalize the Target Variable (Y)
scaler_y = MinMaxScaler()
y_scaled = scaler_y.fit_transform(y_target.reshape(-1, 1))

# 3. Split into Training (80%) and Testing (20%) Sets
# shuffle=False is critical here so the model trains on early cycles and tests on late cycles
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense

# 1. Initialize the Sequential model
model = Sequential([
    # CNN Block: Extract spatial features from Voltage, Current, and Temp
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(200, 3)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=32, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    
    # RNN Block: Track the temporal evolution of the capacity fade
    LSTM(64, activation='tanh', return_sequences=False),
    
    # Dense Block: Output the final predicted capacity value
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# 2. Compile the model for regression
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# 3. Train the network
print("Starting model training...")
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=16,
    validation_split=0.1, # Uses 10% of training data to monitor overfitting
    verbose=1
)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 1. Generate predictions on the unseen test data (final 34 cycles)
y_pred_scaled = model.predict(X_test)

# 2. Inverse-transform the predictions and ground truth back to actual Ampere-hours (Ah)
y_pred_actual = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test)

# 3. Calculate exact real-world evaluation metrics
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))
mae = mean_absolute_error(y_test_actual, y_pred_actual)
r2 = r2_score(y_test_actual, y_pred_actual)

print("\n--- Final Model Evaluation Metrics (Test Set) ---")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f} Ah")
print(f"Mean Absolute Error (MAE):      {mae:.4f} Ah")
print(f"R-squared (R2) Score:           {r2:.4f}")

# 4. Plot the Actual vs. Predicted SOH Degradation Curve
plt.figure(figsize=(12, 6))

# Plot the entire actual degradation history
plt.plot(range(1, 169), y_target, label='Actual Capacity (Ground Truth)', color='blue', linewidth=2)

# Plot the predicted end-of-life trajectory (Cycles 135 to 168)
test_cycle_indices = range(135, 169)
plt.plot(test_cycle_indices, y_pred_actual, label='CNN-LSTM Predicted Capacity', color='red', linestyle='--', marker='o', markersize=4)

# Highlight the train/test split boundary
plt.axvline(x=134, color='green', linestyle=':', linewidth=2, label='Train/Test Split (Cycle 134)')

plt.title('B0005 Battery State of Health (SOH) Prognostics: Actual vs. Predicted')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('Capacity (Ampere-hours)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()
