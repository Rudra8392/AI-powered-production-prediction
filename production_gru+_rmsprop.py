import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
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

# Scale features before PCA
feature_scaler = MinMaxScaler()
features_scaled = feature_scaler.fit_transform(df[features])

# Apply PCA
pca = PCA(n_components=0.95)
pca_result = pca.fit_transform(features_scaled)
pca_columns = [f'PCA_{i+1}' for i in range(pca.n_components_)]
pca_df = pd.DataFrame(pca_result, index=df.index, columns=pca_columns)

# Scale targets
target_scaler = MinMaxScaler()
targets_scaled = target_scaler.fit_transform(df[targets])
df_targets = pd.DataFrame(targets_scaled, index=df.index, columns=targets)

# Combine PCA components with targets
df_pca = pd.concat([pca_df, df_targets], axis=1)

# Save PCA and scalers
with open('pca_model.pkl', 'wb') as f:
    pickle.dump(pca, f)
with open('feature_scaler.pkl', 'wb') as f:
    pickle.dump(feature_scaler, f)
with open('target_scaler.pkl', 'wb') as f:
    pickle.dump(target_scaler, f)

# Function to create sequences
def create_sequences(data, seq_length, X_cols, y_cols):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length][X_cols].values)
        y.append(data.iloc[i+seq_length][y_cols].values)
    return np.array(X), np.array(y)

seq_length = 48  # Time steps
X, y = create_sequences(df_pca, seq_length, pca_columns, targets)

# OPTIMIZED: Reduced Time Series Cross-Validation folds
ts_splits = 3  # Reduced from 5
tscv = TimeSeriesSplit(n_splits=ts_splits)

# OPTIMIZED: Single checkpoint path instead of per-fold
checkpoint_path = "./best_model_gru_pca.h5"

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"Fold {fold+1}/{ts_splits}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    # Define GRU model with bias regularization
    model = Sequential([
        GRU(128, return_sequences=True, input_shape=(seq_length, len(pca_columns)), dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        GRU(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        BatchNormalization(),
        GRU(32, dropout=0.2, recurrent_dropout=0.2),
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(len(targets), kernel_regularizer=l2(0.01))
    ])

    optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    # OPTIMIZED: Callbacks with more aggressive stopping and LR reduction
    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)  # Reduced from 10
    model_checkpoint = ModelCheckpoint(filepath=checkpoint_path, save_best_only=True, monitor='val_loss')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)  # Reduced from 5

    # OPTIMIZED: Training configuration
    batch_size = 64  # Increased from 32
    verbose = 2  # Reduced verbosity
    
    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,  # Reduced from 100
        batch_size=batch_size,
        callbacks=[early_stopping, model_checkpoint, reduce_lr],
        verbose=verbose
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_val, y_val, verbose=1)
    print(f"Fold {fold+1} - Test Loss: {test_loss}, Test MAE: {test_mae}")

# Predict next 3 months
total_future_steps = 1080
last_seq = df_pca[pca_columns].values[-seq_length:].copy()
future_predictions = []

for _ in range(total_future_steps):
    pred = model.predict(last_seq[np.newaxis, :, :], verbose=0)[0]  # Reduced verbosity
    future_predictions.append(pred)
    last_seq = np.roll(last_seq, -1, axis=0)
    last_seq[-1] = np.append(pred, last_seq[-1][-1])  # Ensure shape consistency

future_predictions_array = np.array(future_predictions)
future_predictions_original = target_scaler.inverse_transform(future_predictions_array)
future_df = pd.DataFrame(future_predictions_original, columns=targets)
future_df.to_csv("Future_3Months_Predictions_GRU_PCA.csv", index=False)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title("GRU with PCA Training Loss")
plt.savefig("training_loss_gru_pca.png")

print("Cross-validation and future predictions completed! Predictions saved in Future_3Months_Predictions_GRU_PCA.csv")
