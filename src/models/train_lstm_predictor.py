"""
F1 Pit Stop Predictor - Phase 2B (LSTM)
Temporal sequence model for pit stop prediction with 30-lap sequences
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score

print("="*60)
print("F1 PIT STOP PREDICTOR - PHASE 2B (LSTM)")
print("="*60)
print(f"TensorFlow version: {tf.__version__}")

# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10
BATCH_SIZE = 64
EPOCHS = 30
LSTM_UNITS_1 = 64
LSTM_UNITS_2 = 32
DENSE_UNITS = 16
DROPOUT_RATE = 0.3
LEARNING_RATE = 0.001

# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================

print("\n[1/10] Loading and preprocessing data...")

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'raw'

csv_files = sorted(data_dir.glob('*.csv'))
race_dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(race_dfs, ignore_index=True)

print(f"  Loaded {len(df_all):,} laps from {len(csv_files)} races")

# ============================================================
# SHIFT FEATURES (Leakage prevention)
# ============================================================

print("\n  Shifting features to previous lap...")

grouped = df_all.groupby(['Year', 'RaceName', 'Driver'])

# Shift tire features
df_all['PrevTyreLife'] = grouped['TyreLife'].shift(1)
df_all['PrevCompound'] = grouped['Compound'].shift(1)
df_all['PrevFreshTyre'] = grouped['FreshTyre'].shift(1)

# Shift telemetry
telemetry_cols = ['AvgSpeed', 'MaxSpeed', 'AvgAcceleration', 'MaxAcceleration', 
                  'AvgThrottle', 'AvgBrake', 'AvgRPM', 'DRS_count', 'GearChanges']
for col in telemetry_cols:
    if col in df_all.columns:
        df_all[f'Prev{col}'] = grouped[col].shift(1)

# Shift performance
perf_cols = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']
for col in perf_cols:
    if col in df_all.columns:
        df_all[f'Prev{col}'] = grouped[col].shift(1)

# Shift position
df_all['PrevPosition'] = grouped['Position'].shift(1)
df_all['PrevGapToAhead'] = grouped['GapToAhead'].shift(1)
df_all['PrevGapToBehind'] = grouped['GapToBehind'].shift(1)

# Create derived features
df_all['LapsSinceLastPit'] = 0  
df_all['RemainingLaps'] = df_all.groupby(['Year', 'RaceName'])['LapNumber'].transform('max') - df_all['LapNumber']

print("  ✓ Shifted features")

# ============================================================
# CLEAN DATA
# ============================================================

print("\n  Cleaning data...")

df_model = df_all[
    (df_all['LapNumber'] > 1) &
    (df_all['RemainingLaps'] > 5)
].copy()

print(f"  Clean dataset: {len(df_model):,} laps")

# ============================================================
# ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n  Encoding categorical features...")

# Encode PrevCompound
df_model['PrevCompound'] = df_model['PrevCompound'].fillna('UNKNOWN')
le_compound = LabelEncoder()
df_model['PrevCompound'] = le_compound.fit_transform(df_model['PrevCompound'].astype(str))

# Encode other categoricals
categorical_cols = ['TrackType', 'OvertakingDifficulty']
for col in categorical_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))

print(f"  ✓ Encoded categorical features")

# ============================================================
# FILL ALL NaN VALUES
# ============================================================

print("\n  Filling missing values...")

# Count NaN before
total_nans_before = df_model.isna().sum().sum()
print(f"    NaN values before: {total_nans_before:,}")

# Fill numeric columns with median (or 0 if all NaN)
numeric_cols = df_model.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_cols:
    if df_model[col].isna().sum() > 0:
        median_val = df_model[col].median()
        if pd.isna(median_val):
            median_val = 0
        df_model[col] = df_model[col].fillna(median_val)

# Fill boolean columns
bool_cols = ['PrevFreshTyre', 'IsSafetyCar', 'IsVSC', 'IsYellowFlag', 'Rainfall']
for col in bool_cols:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna(False).astype(int)

# Safety net: fill any remaining NaN
df_model = df_model.fillna(0)

# Replace infinity
df_model = df_model.replace([np.inf, -np.inf], 0)

# Verify
total_nans_after = df_model.isna().sum().sum()
print(f"    NaN values after:  {total_nans_after:,}")

if total_nans_after == 0:
    print(f"    ✓ No NaN values remaining!")
else:
    print(f"    ⚠️  WARNING: Still have {total_nans_after} NaN values!")

print("  ✓ Data preprocessing complete")

# ============================================================
# 2. DEFINE FEATURES
# ============================================================

print("\n[2/10] Defining features...")

feature_columns = [
    'PrevTyreLife', 'PrevCompound', 'PrevFreshTyre',
    'PrevLapTime', 'PrevSector1Time', 'PrevSector2Time', 'PrevSector3Time',
    'PrevAvgSpeed', 'PrevMaxSpeed', 
    'PrevAvgAcceleration', 'PrevMaxAcceleration',
    'PrevAvgThrottle', 'PrevAvgBrake', 'PrevAvgRPM',
    'PrevPosition', 'PrevGapToAhead', 'PrevGapToBehind',
    'LapNumber', 'RemainingLaps', 'LapsSinceLastPit',
    'IsSafetyCar', 'IsVSC', 'IsYellowFlag',
    'AirTemp', 'TrackTemp', 'Humidity',
    'TrackLength_km', 'PitLossTime_sec'
]

# Keep only features that exist
feature_columns = [f for f in feature_columns if f in df_model.columns]

target_column = 'IsPitLap'

print(f"  ✓ Using {len(feature_columns)} features")

# ============================================================
# 3. CREATE SEQUENCES
# ============================================================

print(f"\n[3/10] Creating {SEQUENCE_LENGTH}-lap sequences...")

def create_sequences(df, features, target, sequence_length=30):
    """Create sequences for LSTM training"""
    sequences = []
    labels = []
    metadata_list = []

    grouped = df.groupby(['Year', 'RaceName', 'Driver'])
    total_groups = len(grouped)
    processed = 0

    for (year, race, driver), group in grouped:
        processed += 1
        if processed % 200 == 0:
            print(f"    Processed {processed}/{total_groups} driver-race combinations...")

        group = group.sort_values('LapNumber').reset_index(drop=True)

        if len(group) < sequence_length + 1:
            continue

        feature_values = group[features].values
        target_values = group[target].values

        for i in range(len(group) - sequence_length):
            sequence = feature_values[i:i + sequence_length]
            label = target_values[i + sequence_length]

            sequences.append(sequence)
            labels.append(label)

            metadata_list.append({
                'Year': year,
                'RaceName': race,
                'Driver': driver,
                'SequenceStartLap': group.iloc[i]['LapNumber'],
                'PredictedLap': group.iloc[i + sequence_length]['LapNumber'],
                'ActualPit': label
            })

    X = np.array(sequences)
    y = np.array(labels).astype(int)
    metadata_df = pd.DataFrame(metadata_list)

    return X, y, metadata_df

X_sequences, y_sequences, metadata = create_sequences(
    df=df_model,
    features=feature_columns,
    target=target_column,
    sequence_length=SEQUENCE_LENGTH
)

print(f"\n  Created {len(X_sequences):,} sequences")
print(f"  Sequence shape: {X_sequences.shape}")

# Check class distribution
pit_count = y_sequences.sum()
non_pit_count = len(y_sequences) - pit_count
pit_pct = (pit_count / len(y_sequences)) * 100

print(f"\n  Target distribution:")
print(f"    Non-pit sequences: {non_pit_count:,} ({100-pit_pct:.2f}%)")
print(f"    Pit sequences:     {pit_count:,} ({pit_pct:.2f}%)")

# ============================================================
# CHECK SEQUENCE DATA QUALITY
# ============================================================

print("\n  Checking sequence data quality...")

nan_count = np.isnan(X_sequences).sum()
inf_count = np.isinf(X_sequences).sum()

print(f"    NaN values in sequences: {nan_count:,}")
print(f"    Inf values in sequences: {inf_count:,}")

if nan_count > 0 or inf_count > 0:
    print(f"\n    ⚠️  Cleaning sequences...")
    
    if nan_count > 0:
        # Find which features have NaN
        print(f"\n    Features with NaN:")
        for i, feat in enumerate(feature_columns):
            feat_nan = np.isnan(X_sequences[:, :, i]).sum()
            if feat_nan > 0:
                print(f"      - {feat}: {feat_nan:,} NaN values")
    
    X_sequences = np.nan_to_num(X_sequences, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    ✓ Replaced NaN/Inf with 0")

final_nan = np.isnan(X_sequences).sum()
print(f"\n    Final check: {final_nan} NaN values")
print("  ✓ Sequences are clean")

# ============================================================
# 4. TRAIN/TEST SPLIT
# ============================================================

print("\n[4/10] Train/test split...")

# Race-based split
race_ids = metadata['Year'].astype(str) + '_' + metadata['RaceName']
unique_races = race_ids.unique()

train_races, test_races = train_test_split(
    unique_races,
    test_size=0.2,
    random_state=42
)

train_mask = race_ids.isin(train_races)
test_mask = race_ids.isin(test_races)

X_train = X_sequences[train_mask]
y_train = y_sequences[train_mask]
X_test = X_sequences[test_mask]
y_test = y_sequences[test_mask]

print(f"  ✓ Train sequences: {len(X_train):,}")
print(f"  ✓ Test sequences:  {len(X_test):,}")

train_pit_pct = (y_train.sum() / len(y_train)) * 100
test_pit_pct = (y_test.sum() / len(y_test)) * 100

print(f"\n  Train pit rate: {train_pit_pct:.2f}%")
print(f"  Test pit rate:  {test_pit_pct:.2f}%")

# ============================================================
# 5. CALCULATE CLASS WEIGHTS
# ============================================================

print("\n[5/10] Calculating class weights...")

train_non_pit = (y_train == 0).sum()
train_pit = (y_train == 1).sum()

class_weight_ratio = train_non_pit / train_pit

class_weight = {
    0: 1.0,
    1: float(class_weight_ratio)
}

print(f"  Non-pit sequences: {train_non_pit:,}")
print(f"  Pit sequences:     {train_pit:,}")
print(f"  Class weight ratio: {class_weight_ratio:.1f}")
print(f"\n  ✓ Class weights: {class_weight}")

# ============================================================
# 6. SCALE FEATURES
# ============================================================

print("\n[6/10] Scaling features...")

n_sequences, n_timesteps, n_features = X_train.shape

# Reshape to 2D for scaling
X_train_2d = X_train.reshape(-1, n_features)
X_test_2d = X_test.reshape(-1, n_features)

print(f"  Original shape: {X_train.shape}")
print(f"  Reshaped for scaling: {X_train_2d.shape}")

# Check for NaN/Inf before scaling
print("\n  Checking data before scaling...")
train_nan = np.isnan(X_train_2d).sum()
train_inf = np.isinf(X_train_2d).sum()

print(f"    NaN values: {train_nan:,}")
print(f"    Inf values: {train_inf:,}")

if train_nan > 0 or train_inf > 0:
    print(f"    ⚠️  Cleaning data before scaling...")
    X_train_2d = np.nan_to_num(X_train_2d, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_2d = np.nan_to_num(X_test_2d, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    ✓ Cleaned")

# Fit scaler
scaler = StandardScaler()
X_train_scaled_2d = scaler.fit_transform(X_train_2d)
X_test_scaled_2d = scaler.transform(X_test_2d)

# Reshape back to 3D
X_train_scaled = X_train_scaled_2d.reshape(n_sequences, n_timesteps, n_features)
X_test_scaled = X_test_scaled_2d.reshape(X_test.shape[0], n_timesteps, n_features)

print(f"  ✓ Scaled back to: {X_train_scaled.shape}")

# Sanity check
print(f"\n  Sanity check:")
print(f"    Before scaling - mean: {X_train_2d.mean():.4f}, std: {X_train_2d.std():.4f}")
print(f"    After scaling  - mean: {X_train_scaled_2d.mean():.4f}, std: {X_train_scaled_2d.std():.4f}")

# Final NaN check
scaled_nan = np.isnan(X_train_scaled).sum()
if scaled_nan > 0:
    print(f"\n    ⚠️  WARNING: {scaled_nan} NaN in scaled data!")
    X_train_scaled = np.nan_to_num(X_train_scaled, nan=0.0)
    X_test_scaled = np.nan_to_num(X_test_scaled, nan=0.0)
else:
    print(f"    ✓ No NaN in scaled data")

print("\n  ✓ Features scaled successfully")

# ============================================================
# 7. BUILD LSTM MODEL
# ============================================================

print("\n[7/10] Building LSTM model...")

print(f"  Architecture:")
print(f"    - Input: ({n_timesteps}, {n_features})")
print(f"    - LSTM Layer 1: {LSTM_UNITS_1} units")
print(f"    - LSTM Layer 2: {LSTM_UNITS_2} units")
print(f"    - Dense Layer: {DENSE_UNITS} units")
print(f"    - Output: 1 unit (probability)")
print(f"    - Dropout: {DROPOUT_RATE}")
print(f"    - Learning rate: {LEARNING_RATE}")

model = Sequential([
    LSTM(LSTM_UNITS_1, return_sequences=True, input_shape=(n_timesteps, n_features), name='lstm_1'),
    Dropout(DROPOUT_RATE, name='dropout_1'),
    
    LSTM(LSTM_UNITS_2, return_sequences=False, name='lstm_2'),
    Dropout(DROPOUT_RATE, name='dropout_2'),
    
    Dense(DENSE_UNITS, activation='relu', name='dense'),
    Dropout(DROPOUT_RATE, name='dropout_3'),
    
    Dense(1, activation='sigmoid', name='output')
])

optimizer = Adam(learning_rate=LEARNING_RATE)

model.compile(
    optimizer=optimizer,
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall')
    ]
)

print("\n  ✓ Model built successfully!")
print("\n  Model Summary:")
model.summary()

total_params = model.count_params()
print(f"\n  Total trainable parameters: {total_params:,}")

# ============================================================
# 8. SETUP TRAINING CALLBACKS
# ============================================================

print("\n[8/10] Setting up training callbacks...")

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=3,
    min_lr=0.00001,
    verbose=1
)

callbacks = [early_stopping, reduce_lr]

print("  ✓ Callbacks configured:")
print("    - Early stopping (patience=5)")
print("    - Learning rate reduction (patience=3)")

# ============================================================
# 9. TRAIN MODEL
# ============================================================

print(f"\n[9/10] Training LSTM model...")
print("="*60)
print(f"  Batch size: {BATCH_SIZE}")
print(f"  Max epochs: {EPOCHS}")
print(f"  Class weights: {class_weight}")
print("\n" + "="*60)
print("TRAINING START")
print("="*60 + "\n")

history = model.fit(
    X_train_scaled, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.2,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)

# ============================================================
# 10. EVALUATE MODEL
# ============================================================

print("\n[10/10] Evaluating LSTM model...")
print("="*60)

# Predict on test set
y_pred_proba = model.predict(X_test_scaled, verbose=0)
y_pred = (y_pred_proba > 0.5).astype(int).flatten()

# Classification report
print("\n📊 LSTM RESULTS (30-lap sequences):")
print("-"*60)
print(classification_report(y_test, y_pred, target_names=['Non-Pit', 'Pit']))

# Calculate metrics
lstm_precision = precision_score(y_test, y_pred, zero_division=0)
lstm_recall = recall_score(y_test, y_pred, zero_division=0)
lstm_f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"\n📈 Summary:")
print(f"  Precision: {lstm_precision:.3f}")
print(f"  Recall:    {lstm_recall:.3f}")
print(f"  F1-Score:  {lstm_f1:.3f}")

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print(f"\n📊 Confusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:,} (Correct non-pit predictions)")
print(f"  False Positives: {cm[0,1]:,} (False alarms)")
print(f"  False Negatives: {cm[1,0]:,} (Missed pit stops)")
print(f"  True Positives:  {cm[1,1]:,} (Correct pit predictions)")

# Compare with XGBoost
print("\n" + "="*60)
print("📊 COMPARISON: LSTM vs XGBoost")
print("="*60)

print("\nXGBoost (20-lap sequences, Phase 2A):")
print("  Precision: 0.760")
print("  Recall:    0.840")
print("  F1-Score:  0.800")

print(f"\nLSTM (30-lap sequences, Phase 2B):")
print(f"  Precision: {lstm_precision:.3f}")
print(f"  Recall:    {lstm_recall:.3f}")
print(f"  F1-Score:  {lstm_f1:.3f}")

print("\n📈 Difference:")
print(f"  Precision: {(lstm_precision - 0.76):+.3f} ({(lstm_precision - 0.76)*100:+.1f}%)")
print(f"  Recall:    {(lstm_recall - 0.84):+.3f} ({(lstm_recall - 0.84)*100:+.1f}%)")
print(f"  F1-Score:  {(lstm_f1 - 0.80):+.3f} ({(lstm_f1 - 0.80)*100:+.1f}%)")

if lstm_f1 > 0.80:
    print("\n🎉 LSTM WINS! Better temporal pattern recognition!")
elif lstm_f1 > 0.75:
    print("\n✅ LSTM Competitive! Good temporal learning!")
else:
    print("\n⚠️  XGBoost still better. Consider:")
    print("    - Try 20-lap sequences (more data)")
    print("    - Try 10-lap sequences (less context)")
    print("    - Ensemble LSTM + XGBoost")

print("\n" + "="*60)
print("✅ EVALUATION COMPLETE!")
print("="*60)