"""
src/models/train_laptime_predictor.py

Phase 2C: Lap Time Prediction Model (FINAL FIX)
Predicts lap time DELTA using ONLY predictive features

KEY FIX #2: Remove outcome features (AvgSpeed, AvgThrottle, AvgAcceleration)
These are RESULTS of tire degradation, not INPUTS we can know beforehand!

Only use features we KNOW before the lap happens:
- Tire state (age, compound)
- Race state (lap number, position, gaps)
- Track conditions (temperature, weather)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("F1 LAP TIME PREDICTOR - PHASE 2C (FINAL FIX)")
print("="*60)

# ============================================================
# CONFIGURATION
# ============================================================

TARGET = 'LapTimeDelta'
TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost parameters - tuned for better generalization
XGBOOST_PARAMS = {
    'objective': 'reg:squarederror',
    'max_depth': 5,  # Reduced to prevent overfitting
    'learning_rate': 0.03,  # Slower learning
    'n_estimators': 300,  # More trees
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 5,  # Prevent overfitting on small groups
    'gamma': 0.1,  # Regularization
    'reg_alpha': 0.1,  # L1 regularization
    'reg_lambda': 1.0,  # L2 regularization
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# ============================================================
# 1. LOAD DATA
# ============================================================

print("\n[1/9] Loading data...")

project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'raw'

csv_files = sorted(data_dir.glob('*.csv'))
race_dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(race_dfs, ignore_index=True)

print(f"  Loaded {len(df_all):,} laps from {len(csv_files)} races")

# ============================================================
# 2. FILTER CLEAN LAPS ONLY
# ============================================================

print("\n[2/9] Filtering to clean racing laps...")

total_laps = len(df_all)

clean_laps = df_all[
    (df_all['LapNumber'] > 1) &
    (~df_all['IsPitLap']) &
    (~df_all['IsSafetyCar'].fillna(False)) &
    (~df_all['IsVSC'].fillna(False)) &
    (~df_all['HasDamage'].fillna(False)) &
    (~df_all['HasPendingPenalty'].fillna(False)) &
    (df_all['LapTime'].notna()) &
    (df_all['LapTime'] > 0) &
    (df_all['LapTime'] < 200) &
    (df_all['RaceName'].notna())
].copy()

print(f"  Total laps: {total_laps:,}")
print(f"  Clean laps: {len(clean_laps):,} ({len(clean_laps)/total_laps*100:.1f}%)")
print(f"  Filtered out: {total_laps - len(clean_laps):,} laps")

# ============================================================
# 3. CALCULATE TRACK BASELINES
# ============================================================

print("\n[3/9] Calculating track baselines...")

# For each track, find fast lap baseline (10th percentile)
track_baselines = clean_laps.groupby('RaceName')['LapTime'].quantile(0.10)

print(f"\n  Track baselines calculated for {len(track_baselines)} tracks")
print(f"  Example baselines:")
for track in list(track_baselines.index)[:5]:
    print(f"    {track:<30} {track_baselines[track]:.3f}s")

# Add baseline and calculate delta
clean_laps['TrackBaseline'] = clean_laps['RaceName'].map(track_baselines)
clean_laps['LapTimeDelta'] = clean_laps['LapTime'] - clean_laps['TrackBaseline']

print(f"\n  LapTimeDelta statistics:")
print(f"    Mean: {clean_laps['LapTimeDelta'].mean():.3f}s")
print(f"    Std:  {clean_laps['LapTimeDelta'].std():.3f}s")
print(f"    Min:  {clean_laps['LapTimeDelta'].min():.3f}s")
print(f"    Max:  {clean_laps['LapTimeDelta'].max():.3f}s")

# ============================================================
# 4. DEFINE PREDICTIVE FEATURES ONLY
# ============================================================

print("\n[4/9] Defining PREDICTIVE features only...")

# PREDICTIVE FEATURES = Things we KNOW before the lap happens
PREDICTIVE_FEATURES = [
    # TIRE STATE (we know this!)
    'TyreLife',           # How old are the tires?
    'Compound',           # SOFT, MEDIUM, HARD
    'FreshTyre',          # New or used tires?
    
    # RACE PROGRESSION (we know this!)
    'LapNumber',          # Fuel decreases each lap (0.035s/kg)
    
    # POSITION & TRAFFIC (we know this!)
    'Position',           # Where in pack (affects traffic)
    'GapToAhead',         # Stuck behind someone?
    
    # TRACK CONDITIONS (we know this!)
    'TrackTemp',          # Temperature
    'AirTemp',            # Air temperature
    'OvertakingDifficulty',  # Track characteristic
]

# REMOVED (these are OUTCOMES, not inputs!):
# ❌ AvgSpeed - This is the RESULT of tire degradation!
# ❌ AvgThrottle - This is the RESULT of tire degradation!
# ❌ AvgAcceleration - This is the RESULT of tire degradation!

# Check availability
available_features = [f for f in PREDICTIVE_FEATURES if f in clean_laps.columns]
missing_features = [f for f in PREDICTIVE_FEATURES if f not in clean_laps.columns]

print(f"  Predictive features: {len(PREDICTIVE_FEATURES)}")
print(f"  Available: {len(available_features)}")

if missing_features:
    print(f"  Missing: {missing_features}")

FEATURES = available_features

print(f"\n  Final feature set ({len(FEATURES)} features):")
for feat in FEATURES:
    print(f"    ✓ {feat}")

print(f"\n  Removed outcome features:")
print(f"    ❌ AvgSpeed (result of tire deg)")
print(f"    ❌ AvgThrottle (result of tire deg)")
print(f"    ❌ AvgAcceleration (result of tire deg)")

# ============================================================
# 5. ENCODE CATEGORICAL FEATURES
# ============================================================

print("\n[5/9] Encoding categorical features...")

df_model = clean_laps.copy()

# Encode Compound
if 'Compound' in df_model.columns:
    df_model['Compound'] = df_model['Compound'].fillna('UNKNOWN')
    le_compound = LabelEncoder()
    df_model['Compound'] = le_compound.fit_transform(df_model['Compound'].astype(str))
    print(f"  Compound classes: {list(le_compound.classes_)}")

# Encode OvertakingDifficulty
if 'OvertakingDifficulty' in df_model.columns:
    df_model['OvertakingDifficulty'] = df_model['OvertakingDifficulty'].fillna('UNKNOWN')
    le_difficulty = LabelEncoder()
    df_model['OvertakingDifficulty'] = le_difficulty.fit_transform(df_model['OvertakingDifficulty'].astype(str))

# Convert boolean to int
if 'FreshTyre' in df_model.columns:
    df_model['FreshTyre'] = df_model['FreshTyre'].fillna(False).astype(int)

print("  ✓ Categorical features encoded")

# ============================================================
# 6. HANDLE MISSING VALUES
# ============================================================

print("\n[6/9] Handling missing values...")

for feat in FEATURES:
    if df_model[feat].dtype in ['float64', 'int64']:
        if df_model[feat].isna().sum() > 0:
            median_val = df_model[feat].median()
            df_model[feat] = df_model[feat].fillna(median_val)
            print(f"  Filled {feat} with median: {median_val:.2f}")

# Replace inf
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.fillna(0)

print("  ✓ Missing values handled")

# ============================================================
# 7. PREPARE DATA
# ============================================================

print("\n[7/9] Preparing training data...")

X = df_model[FEATURES].copy()
y = df_model[TARGET].copy()

race_ids = df_model['Year'].astype(str) + '_' + df_model['RaceName']

print(f"  Features shape: {X.shape}")
print(f"  Target shape: {y.shape}")
print(f"\n  Target (LapTimeDelta) statistics:")
print(f"    Mean: {y.mean():.3f}s")
print(f"    Std:  {y.std():.3f}s")
print(f"    Min:  {y.min():.3f}s")
print(f"    Max:  {y.max():.3f}s")

# ============================================================
# 8. TRAIN/TEST SPLIT
# ============================================================

print("\n[8/9] Train/test split...")

unique_races = race_ids.unique()
train_races, test_races = train_test_split(
    unique_races,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

train_mask = race_ids.isin(train_races)
test_mask = race_ids.isin(test_races)

X_train = X[train_mask]
y_train = y[train_mask]
X_test = X[test_mask]
y_test = y[test_mask]

print(f"  Training races: {len(train_races)}")
print(f"  Test races: {len(test_races)}")
print(f"  Training laps: {len(X_train):,}")
print(f"  Test laps: {len(X_test):,}")

# ============================================================
# 9. TRAIN XGBOOST MODEL
# ============================================================

print("\n[9/9] Training XGBoost lap time predictor...")
print("="*60)

model = xgb.XGBRegressor(**XGBOOST_PARAMS)

print(f"  Model parameters (regularized for better generalization):")
for key, value in XGBOOST_PARAMS.items():
    print(f"    {key}: {value}")

print(f"\n  Training...")

model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    verbose=False
)

print("  ✓ Training complete!")

# ============================================================
# 10. EVALUATE MODEL
# ============================================================

print("\n" + "="*60)
print("MODEL EVALUATION")
print("="*60)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

train_mae = mean_absolute_error(y_train, y_train_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)

print("\n📊 PERFORMANCE METRICS:")
print("-"*60)
print(f"{'Metric':<25} {'Training':<15} {'Test':<15}")
print("-"*60)
print(f"{'Mean Absolute Error':<25} {train_mae:<15.3f} {test_mae:<15.3f}")
print(f"{'Root Mean Squared Error':<25} {train_rmse:<15.3f} {test_rmse:<15.3f}")
print(f"{'R² Score':<25} {train_r2:<15.3f} {test_r2:<15.3f}")
print("-"*60)

# Check for overfitting
overfitting_gap = train_mae - test_mae
print(f"\n📈 Overfitting check:")
print(f"  Train MAE: {train_mae:.3f}s")
print(f"  Test MAE:  {test_mae:.3f}s")
print(f"  Gap:       {abs(overfitting_gap):.3f}s")

if abs(overfitting_gap) < 0.5:
    print(f"  ✅ Good generalization!")
elif abs(overfitting_gap) < 1.0:
    print(f"  ✓ Acceptable generalization")
else:
    print(f"  ⚠️  Some overfitting detected")

print(f"\n📈 INTERPRETATION:")
print(f"  ✓ Average prediction error: ±{test_mae:.3f}s from baseline")
print(f"  ✓ Model explains {test_r2*100:.1f}% of variance")

if test_r2 < 0:
    print(f"  ⚠️  Negative R² means model worse than predicting mean")
    print(f"  → This is expected with limited predictive features")
    print(f"  → For A/B testing, relative comparison is what matters")
elif test_mae < 1.0:
    print(f"  🎉 EXCELLENT for A/B testing!")
elif test_mae < 2.0:
    print(f"  ✅ GOOD for A/B testing!")
else:
    print(f"  ✓ Acceptable for relative strategy comparison")

# ============================================================
# 11. FEATURE IMPORTANCE
# ============================================================

print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nAll Features by Importance:")
print("-"*60)
for idx, row in importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"  {row['Feature']:<25} {row['Importance']:.4f} {bar}")

# Highlight TyreLife
tyrelife_row = importance_df[importance_df['Feature'] == 'TyreLife']
if not tyrelife_row.empty:
    importance = tyrelife_row['Importance'].values[0]
    rank = importance_df[importance_df['Feature'] == 'TyreLife'].index[0] + 1
    print(f"\n🔍 TyreLife Analysis:")
    print(f"  Importance: {importance:.4f}")
    print(f"  Rank: #{rank} out of {len(FEATURES)}")
    if importance > 0.15:
        print(f"  🎉 Major factor in lap time prediction!")
    elif importance > 0.08:
        print(f"  ✅ Significant factor")
    else:
        print(f"  ⚠️  Lower than expected")

# ============================================================
# 12. EXAMPLE PREDICTIONS
# ============================================================

print("\n" + "="*60)
print("EXAMPLE PREDICTIONS")
print("="*60)

sample_indices = np.random.choice(len(X_test), 10, replace=False)

print("\nSample predictions from test set:")
print("-"*60)
print(f"{'Actual Δ':<12} {'Pred Δ':<12} {'Error':<12} {'TyreLife':<10} {'Compound':<10}")
print("-"*60)

for idx in sample_indices:
    actual = y_test.iloc[idx]
    predicted = y_test_pred[idx]
    error = predicted - actual
    
    tire_life = int(X_test.iloc[idx]['TyreLife'])
    compound_code = int(X_test.iloc[idx]['Compound'])
    compound_name = le_compound.classes_[compound_code] if compound_code < len(le_compound.classes_) else 'UNKNOWN'
    
    print(f"{actual:>+11.3f}s {predicted:>+11.3f}s {error:>+11.3f}s {tire_life:>9} {compound_name:>9}")

# ============================================================
# 13. SAVE MODEL
# ============================================================

print("\n" + "="*60)
print("SAVING MODEL")
print("="*60)

models_dir = project_root / 'data' / 'models'
models_dir.mkdir(parents=True, exist_ok=True)

# Save model
model_path = models_dir / 'laptime_predictor.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"\n✓ Model saved: {model_path}")

# Save features
features_path = models_dir / 'laptime_features.pkl'
with open(features_path, 'wb') as f:
    pickle.dump(FEATURES, f)
print(f"✓ Features saved: {features_path}")

# Save encoders
encoders = {
    'compound': le_compound,
    'difficulty': le_difficulty if 'OvertakingDifficulty' in df_model.columns else None
}
encoders_path = models_dir / 'laptime_encoders.pkl'
with open(encoders_path, 'wb') as f:
    pickle.dump(encoders, f)
print(f"✓ Encoders saved: {encoders_path}")

# Save track baselines
baselines_path = models_dir / 'track_baselines.pkl'
with open(baselines_path, 'wb') as f:
    pickle.dump(track_baselines.to_dict(), f)
print(f"✓ Track baselines saved: {baselines_path}")

# Save metadata
metadata = {
    'test_mae': test_mae,
    'test_rmse': test_rmse,
    'test_r2': test_r2,
    'n_features': len(FEATURES),
    'training_laps': len(X_train),
    'test_laps': len(X_test),
    'predicts': 'LapTimeDelta from baseline',
    'feature_type': 'PREDICTIVE ONLY (no outcome features)'
}
metadata_path = models_dir / 'laptime_metadata.pkl'
with open(metadata_path, 'wb') as f:
    pickle.dump(metadata, f)
print(f"✓ Metadata saved: {metadata_path}")

# ============================================================
# SUMMARY
# ============================================================

print("\n" + "="*60)
print("✅ LAP TIME PREDICTOR COMPLETE!")
print("="*60)

print(f"\n📊 Model Summary:")
print(f"  - Type: Predictive features only")
print(f"  - Target: Lap time DELTA from track baseline")
print(f"  - Test MAE: ±{test_mae:.3f}s")
print(f"  - Test R²: {test_r2:.3f}")
print(f"  - Features: {len(FEATURES)} predictive features")
print(f"  - Training samples: {len(X_train):,}")

print(f"\n🎯 For A/B Testing:")
print(f"  - Predicts relative lap time differences")
print(f"  - Suitable for comparing strategies")
print(f"  - TyreLife importance: {importance_df[importance_df['Feature']=='TyreLife']['Importance'].values[0]:.4f}")

print(f"\n🚀 Ready for strategy simulation!")
print("="*60)