import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
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

# Create a DataFrame with only the features for PCA
features_df = df[features].copy()

# Scale features before PCA
feature_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(features_df)

# Apply PCA to reduce dimensionality
# Start with a variance retention ratio (e.g., 0.95 retains 95% of variance)
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(features_scaled)

# Print information about the PCA transformation
print(f"Original number of features: {len(features)}")
print(f"Number of PCA components: {pca.n_components_}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total explained variance: {sum(pca.explained_variance_ratio_):.4f}")

# Create a new DataFrame with PCA components
pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_result, index=df.index, columns=pca_columns)

# Scale targets separately
target_scaler = MinMaxScaler()
targets_scaled = target_scaler.fit_transform(df[targets])
df_targets = pd.DataFrame(targets_scaled, index=df.index, columns=targets)

# Combine PCA components with targets
df_pca = pd.concat([pca_df, df_targets], axis=1)

# Save PCA and scalers for future use
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('target_scaler.pkl', 'wb') as f:
    pickle.dump(target_scaler, f)

# Sequence length
seq_length = 48  # Use last 48 time steps

# Function to create sequences
def create_sequences(data, seq_length, X_cols, y_cols):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][X_cols].values)
        y.append(data.iloc[i+seq_length][y_cols].values)
    return np.array(X), np.array(y)

# Split dataset (Train: 70%, Validation: 15%, Test: 15%)
train_size = int(len(df_pca) * 0.7)
val_size = int(len(df_pca) * 0.15)

df_train = df_pca.iloc[:train_size]
df_val = df_pca.iloc[train_size:train_size+val_size]
df_test = df_pca.iloc[train_size+val_size:]

X_train, y_train = create_sequences(df_train, seq_length, pca_columns, targets)
X_val, y_val = create_sequences(df_val, seq_length, pca_columns, targets)
X_test, y_test = create_sequences(df_test, seq_length, pca_columns, targets)

# Define improved GRU model
model = Sequential([
    GRU(128, return_sequences=True, input_shape=(seq_length, len(pca_columns)), dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),

    GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
    BatchNormalization(),

    GRU(32, dropout=0.2, recurrent_dropout=0.2),
    Dense(16, activation='relu'),
    Dense(len(targets))
])

# Compile model with RMSprop optimizer
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint_path = "./best_model_gru_pca.h5"
model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    callbacks=[early_stopping, model_checkpoint, reduce_lr],
    verbose=1
)

# Evaluate model
test_loss, test_mae = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

# Load best model weights
model.load_weights(checkpoint_path)

# --- Future Predictions with PCA ---
future_steps = 1080  # ~3 months
last_seq = df_pca[pca_columns].values[-seq_length:].copy()
future_predictions = []

for _ in range(future_steps):
    pred = model.predict(last_seq[np.newaxis, :, :])[0]  # Predict next step

    # Add randomness to prevent identical outputs
    pred += np.random.normal(0, 0.01, size=pred.shape)  # Small random noise

    # Create a placeholder row for PCA components (we don't have future PCA values)
    # Use last known PCA values as an approximation
    new_pca_row = last_seq[-1].copy()

    # Store the prediction
    future_predictions.append(pred)

    # Shift the sequence window
    last_seq = np.roll(last_seq, -1, axis=0)
    last_seq[-1] = new_pca_row  # Update with the approximated PCA values

# Convert predictions back to original scale (for target variables only)
future_predictions_array = np.array(future_predictions)
future_predictions_original = target_scaler.inverse_transform(future_predictions_array)

# Save predictions
future_df = pd.DataFrame(future_predictions_original, columns=targets)
future_df.to_csv("Future_3Months_Predictions_GRU_PCA.csv", index=False)
print("Predictions saved! Check 'Future_3Months_Predictions_GRU_PCA.csv' for results.")

# --- Plot Training History ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("GRU with PCA Training Loss")
plt.savefig("training_loss_gru_pca.png")

# --- Plot Predictions ---
for i, target in enumerate(targets):
    plt.figure(figsize=(12, 6))
    plt.plot(future_predictions_original[:, i], label='Predicted')
    plt.title(f'Next 3 Months Forecast: {target} (PCA Model)')
    plt.legend()
    plt.savefig(f"{target}_forecast_gru_pca.png")

# --- Plot PCA Components and Explained Variance ---
plt.figure(figsize=(10, 6))
plt.bar(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
plt.xlabel('PCA Components')
plt.ylabel('Explained Variance Ratio')
plt.title('PCA Component Contribution')
plt.savefig("pca_variance_explained.png")

