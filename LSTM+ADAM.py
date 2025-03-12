import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import tensorflow as tf
import os
import pickle

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Load dataset
df = pd.read_excel("/content/202 GL (1).xlsx", sheet_name="Sheet1")
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Define features and targets
features = ['WHTP', 'DSP', 'THP', 'Pressure DS Choke', 'Temperature DS Choke', 'CHP', 'TAP', 'BCP']
targets = ['OIL (BOPD)', 'GAS (MCF)', 'WATER (BWPD)']

# Scale data
scaler = MinMaxScaler()
df[features + targets] = scaler.fit_transform(df[features + targets])

# Sequence length (increase to capture longer trends)
seq_length = 48  # Use last 48 time steps instead of 24

# Function to create sequences
def create_sequences(data, seq_length, X_cols, y_cols):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][X_cols].values)
        y.append(data.iloc[i+seq_length][y_cols].values)
    return np.array(X), np.array(y)

# Split dataset (Train: 70%, Validation: 15%, Test: 15%)
train_size = int(len(df) * 0.7)
val_size = int(len(df) * 0.15)

df_train = df.iloc[:train_size]
df_val = df.iloc[train_size:train_size+val_size]
df_test = df.iloc[train_size+val_size:]

X_train, y_train = create_sequences(df_train, seq_length, features, targets)
X_val, y_val = create_sequences(df_val, seq_length, features, targets)
X_test, y_test = create_sequences(df_test, seq_length, features, targets)

# Define improved LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(seq_length, len(features)), dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),

    LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),

    LSTM(32, dropout=0.2, recurrent_dropout=0.2),
    Dense(16, activation='relu'),
    Dense(len(targets))
])

# Compile model with Adam optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_path = "./best_model.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,  # Increase epochs for better learning
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Load best model weights
model.load_weights(checkpoint_path)

# --- Improved Future Predictions ---
future_steps = 1080  # Assuming 2-hour intervals, 3 months = 1080 steps
last_seq = df[features].values[-seq_length:].copy()
future_predictions = []

for _ in range(future_steps):
    pred = model.predict(last_seq[np.newaxis, :, :])[0]  # Predict next step

    # Add randomness to prevent identical outputs
    pred += np.random.normal(0, 0.01, size=pred.shape)  # Small random noise

    # Create new row with past feature values + predicted target values
    new_row = last_seq[-1].copy()
    new_row[-len(targets):] = pred  # Replace last columns with new predictions

    future_predictions.append(pred)
    last_seq = np.roll(last_seq, -1, axis=0)  # Shift left
    last_seq[-1] = new_row  # Update with new prediction

# Convert predictions back to original scale
future_predictions = scaler.inverse_transform(np.concatenate([np.zeros((future_steps, len(features))), future_predictions], axis=1))[:, -len(targets):]

# Save predictions
future_df = pd.DataFrame(future_predictions, columns=targets)
future_df.to_csv("Future_3Months_Predictions.csv", index=False)
print("Predictions saved! Check 'Future_3Months_Predictions.csv' for results.")

# --- Plot Training History ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("LSTM Training Loss")
plt.savefig("training_loss.png")

# --- Plot Predictions ---
for i, target in enumerate(targets):
    plt.figure(figsize=(12, 6))
    plt.plot(future_predictions[:, i], label='Predicted')
    plt.title(f'Next 3 Months Forecast: {target}')
    plt.legend()
    plt.savefig(f"{target}_forecast.png")

print("Plots saved!")
