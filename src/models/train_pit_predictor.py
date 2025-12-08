import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("F1 PIT STOP PREDICTOR - PHASE 2A (LEAKAGE-FREE)")
print("="*60)

print("\n[1/8] Loading data...")
project_root = Path(__file__).parent.parent.parent
data_dir = project_root / 'data' / 'raw'

csv_files = sorted(data_dir.glob('*.csv'))
print(f"Found {len(csv_files)} race files")

race_dfs = [pd.read_csv(f) for f in csv_files]
df_all = pd.concat(race_dfs, ignore_index=True)

print(f"✓ Loaded {len(df_all):,} laps from {df_all['RaceName'].nunique()} races")

print("\n[2/8] Defining features...")

target_variable = 'IsPitLap'

# NOTE: These will be created as "Prev" versions in feature engineering
base_features = [
    # Tire state (from PREVIOUS lap)
    'PrevTyreLife', 'PrevCompound', 'PrevFreshTyre', 'PrevStartingCompound',
    
    # Performance (from PREVIOUS lap)
    'PrevLapTime', 'PrevSector1Time', 'PrevSector2Time', 'PrevSector3Time',
    
    # Telemetry (from PREVIOUS lap)
    'PrevAvgSpeed', 'PrevMaxSpeed', 
    'PrevAvgThrottle', 'PrevThrottleVariance',
    'PrevAvgBrake', 'PrevMaxBrake',
    'PrevAvgAcceleration', 'PrevMaxAcceleration',
    'PrevAvgRPM', 'PrevGearChanges', 'PrevDRS_count',
    
    # Position (from PREVIOUS lap)
    'PrevPosition', 'PrevGapToAhead', 'PrevGapToBehind', 
    
    # Race state (these don't change during pit lap - safe to use)
    'StartingPosition', 'LapNumber', 'PreviousStint',
    'IsSafetyCar', 'IsVSC', 'IsYellowFlag', 'IsGreen',
    
    # Weather (doesn't change during lap)
    'AirTemp', 'TrackTemp', 'Humidity', 'Rainfall',
    
    # Track characteristics (constant)
    'TrackLength_km', 'PitLossTime_sec', 'OvertakingDifficulty',
    'TypicalPitStops', 'TrackType', 'DRS_Zones',
    
    # Traffic (current lap - but safe)
    'InTraffic', 'LappingBackmarker',
    
    # Qualifying (constant)
    'QualifyingSession', 'FreeCompoundChoice',
]

print(f"✓ {len(base_features)} base features defined (will use PREVIOUS lap values)")

print("\n[3/8] Feature engineering (anti-leakage - FINAL)...")

# Group by driver in each race
grouped = df_all.groupby(['Year', 'RaceName', 'Driver'])

print("\n  [3.1] Shifting tire features to previous lap...")

df_all['PrevTyreLife'] = grouped['TyreLife'].shift(1).fillna(df_all['TyreLife'])
df_all['PrevFreshTyre'] = grouped['FreshTyre'].shift(1).fillna(df_all['FreshTyre'])
df_all['PrevCompound'] = grouped['Compound'].shift(1).fillna(df_all['Compound'])

if 'StartingCompound' in df_all.columns:
    df_all['PrevStartingCompound'] = grouped['StartingCompound'].shift(1).fillna(df_all['StartingCompound'])

print("  ✓ Shifted: TyreLife, FreshTyre, Compound, StartingCompound")

print("\n  [3.2] Shifting telemetry features to previous lap...")

telemetry_features = [
    'AvgSpeed', 'MaxSpeed', 
    'AvgThrottle', 'ThrottleVariance',
    'AvgBrake', 'MaxBrake',
    'AvgAcceleration', 'MaxAcceleration',
    'AvgRPM', 'GearChanges', 'DRS_count'
]

for feat in telemetry_features:
    if feat in df_all.columns:
        df_all[f'Prev{feat}'] = grouped[feat].shift(1).fillna(df_all[feat])

print(f"  ✓ Shifted {len([f for f in telemetry_features if f in df_all.columns])} telemetry features")


print("\n  [3.3] Shifting performance features to previous lap...")

performance_features = [
    'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time'
]

for feat in performance_features:
    if feat in df_all.columns:
        df_all[f'Prev{feat}'] = grouped[feat].shift(1).fillna(df_all[feat])

print(f"  ✓ Shifted {len([f for f in performance_features if f in df_all.columns])} performance features")

print("\n  [3.4] Shifting position features to previous lap...")

position_features = ['Position', 'GapToAhead', 'GapToBehind']

for feat in position_features:
    if feat in df_all.columns:
        df_all[f'Prev{feat}'] = grouped[feat].shift(1).fillna(df_all[feat])

print(f"  ✓ Shifted {len([f for f in position_features if f in df_all.columns])} position features")

print("\n  [3.5] Calculating laps since last pit...")

def calc_laps_since_pit_no_leakage(group):
    laps = []
    last_pit = 0
    
    for i, (lap_num, is_pit) in enumerate(zip(group['LapNumber'], group['IsPitLap'])):
        if i > 0:
            prev_is_pit = group['IsPitLap'].iloc[i-1]
            if prev_is_pit and group['LapNumber'].iloc[i-1] > 1:
                last_pit = group['LapNumber'].iloc[i-1]
        
        laps.append(lap_num - last_pit)
    
    return pd.Series(laps, index=group.index)

df_all['LapsSinceLastPit'] = grouped.apply(
    calc_laps_since_pit_no_leakage
).reset_index(level=[0,1,2], drop=True)

print("  ✓ Calculated LapsSinceLastPit")

print("\n  [3.6] Calculating pit stops made so far...")

df_all['PitStopsMadeSoFar'] = grouped['IsPitLap'].apply(
    lambda x: x.shift(1).fillna(0).cumsum()
).reset_index(level=[0,1,2], drop=True).astype(int)

print("  ✓ Calculated PitStopsMadeSoFar")

print("\n  [3.7] Shifting stint to previous lap...")

if 'Stint' in df_all.columns:
    df_all['PreviousStint'] = grouped['Stint'].shift(1).fillna(df_all['Stint'])
    print("  ✓ Created PreviousStint")

print("\n  [3.8] Creating remaining derived features...")

df_all['TotalRaceLaps'] = df_all.groupby(['Year', 'RaceName'])['LapNumber'].transform('max')
df_all['RemainingLaps'] = df_all['TotalRaceLaps'] - df_all['LapNumber']

# Track rules
qatar_mask = df_all['RaceName'].str.contains('Qatar', case=False, na=False)
df_all['HasMandatoryTireLimit'] = qatar_mask
df_all['MandatoryTireLimitLaps'] = 999
df_all.loc[qatar_mask, 'MandatoryTireLimitLaps'] = 25

# Use PREVIOUS tire life for limit check
df_all['ApproachingTireLimit'] = (
    (df_all['PrevTyreLife'] >= df_all['MandatoryTireLimitLaps'] - 3) &
    df_all['HasMandatoryTireLimit']
)

monaco_mask = (
    df_all['RaceName'].str.contains('Monaco', case=False, na=False) &
    (df_all['Year'] >= 2025)
)
df_all['HasMandatory2Stop'] = monaco_mask
df_all['NeedsAnotherPitStop'] = (
    df_all['HasMandatory2Stop'] &
    (df_all['PitStopsMadeSoFar'] < 2)
)

hard_tracks = ['Monaco', 'Singapore', 'Hungary', 'Zandvoort']
df_all['IsHardToOvertake'] = df_all['RaceName'].apply(
    lambda x: any(t.lower() in str(x).lower() for t in hard_tracks)
)

print("  ✓ Created track rules and remaining features")

print("\n✓ Feature engineering complete (100% leakage-free)")

print("\n🔍 Leakage verification:")
pit_laps = df_all[df_all['IsPitLap'] == True]
print(f"  PrevTyreLife on pit laps: min={pit_laps['PrevTyreLife'].min():.0f}, max={pit_laps['PrevTyreLife'].max():.0f}, mean={pit_laps['PrevTyreLife'].mean():.1f}")
print(f"  PrevAvgAcceleration on pit laps: mean={pit_laps['PrevAvgAcceleration'].mean():.2f}")
print(f"  PitStopsMadeSoFar on pit laps: min={pit_laps['PitStopsMadeSoFar'].min()}, max={pit_laps['PitStopsMadeSoFar'].max()}")



print("\n[4/8] Data cleaning...")

# Fill missing gaps
df_all['PrevGapToAhead'] = df_all['PrevGapToAhead'].fillna(0)
df_all['PrevGapToBehind'] = df_all['PrevGapToBehind'].fillna(999)

# Fill missing telemetry with median
for col in df_all.columns:
    if col.startswith('Prev') and df_all[col].dtype in ['float64', 'int64']:
        if df_all[col].isna().any():
            median_val = df_all[col].median()
            df_all[col] = df_all[col].fillna(median_val)

# Fill missing weather
weather_cols = ['AirTemp', 'TrackTemp', 'Humidity', 'Rainfall']
for col in weather_cols:
    if col in df_all.columns and df_all[col].isna().any():
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Fill boolean flags
bool_cols = ['PrevFreshTyre', 'IsSafetyCar', 'IsVSC', 'IsYellowFlag', 'IsGreen',
             'InTraffic', 'LappingBackmarker', 'FreeCompoundChoice']
for col in bool_cols:
    if col in df_all.columns and df_all[col].isna().any():
        df_all[col] = df_all[col].fillna(False)

# Fill track characteristics
track_cols = ['TrackLength_km', 'PitLossTime_sec', 'TypicalPitStops', 'DRS_Zones']
for col in track_cols:
    if col in df_all.columns and df_all[col].isna().any():
        df_all[col] = df_all[col].fillna(df_all[col].median())

# Exclude lap 1 and final 5 laps
df_model = df_all[
    (df_all['LapNumber'] > 1) &
    (df_all['RemainingLaps'] > 5)
].copy()

# Drop missing critical features
critical = ['PrevLapTime', 'PrevTyreLife', 'PrevPosition', 'PrevCompound']
for col in critical:
    if col in df_model.columns:
        df_model = df_model[df_model[col].notna()]

print(f"✓ Clean dataset: {len(df_model):,} laps")



print("\n[5/8] Encoding categorical...")

categorical = ['PrevCompound', 'TrackType', 'OvertakingDifficulty',
               'QualifyingSession', 'PrevStartingCompound']

for col in categorical:
    if col in df_model.columns:
        df_model[col] = df_model[col].fillna('UNKNOWN')
        le = LabelEncoder()
        df_model[col] = le.fit_transform(df_model[col].astype(str))

print("✓ Encoded categorical features")

print("\n[6/8] Preparing final feature set...")

# Add derived features
derived = ['LapsSinceLastPit', 'RemainingLaps', 'PitStopsMadeSoFar']
track_rules = ['HasMandatoryTireLimit', 'MandatoryTireLimitLaps',
               'ApproachingTireLimit', 'HasMandatory2Stop',
               'NeedsAnotherPitStop', 'IsHardToOvertake']

all_features = base_features + derived + track_rules
final_features = [f for f in all_features if f in df_model.columns]

# Fill ALL remaining NaNs (safety check)
df_model[final_features] = df_model[final_features].fillna(0)

print(f"✓ Total features: {len(final_features)}")
print(f"✓ Target: {target_variable}")
print(f"✓ Pit rate: {df_model[target_variable].mean()*100:.2f}%")

print("\n[7/8] Train/test split...")

df_model['RaceID'] = df_model['Year'].astype(str) + '_' + df_model['RaceName']
unique_races = df_model['RaceID'].unique()

race_meta = df_model.groupby('RaceID').agg({'Year': 'first'}).reset_index()
train_races, test_races = train_test_split(
    race_meta['RaceID'],
    test_size=0.2,
    random_state=42,
    stratify=race_meta['Year']
)

train_df = df_model[df_model['RaceID'].isin(train_races)]
test_df = df_model[df_model['RaceID'].isin(test_races)]

X_train = train_df[final_features]
y_train = train_df[target_variable]
X_test = test_df[final_features]
y_test = test_df[target_variable]

scale_pos_weight = (y_train == False).sum() / (y_train == True).sum()

print(f"✓ Train: {len(X_train):,} laps")
print(f"✓ Test: {len(X_test):,} laps")
print(f"✓ Scale pos weight: {scale_pos_weight:.1f}")

print("\n[8/8] Training models...")
print("="*60)

# Model 1: Logistic Regression
print("\n📊 MODEL 1: Logistic Regression")
print("-"*60)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# Model 2: Random Forest
print("\n📊 MODEL 2: Random Forest")
print("-"*60)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# Model 3: XGBoost
print("\n🚀 MODEL 3: XGBoost")
print("-"*60)

xgb = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred_xgb))

print("\n" + "="*60)
print("✅ TRAINING COMPLETE!")
print("="*60)

print("\n" + "="*60)
print("📊 MODEL ANALYSIS")
print("="*60)

# Feature Importance
print("\n[1/5] Feature Importance Analysis...")
print("-"*60)

feature_importance = pd.DataFrame({
    'feature': final_features,
    'importance': xgb.feature_importances_
}).sort_values('importance', ascending=False)

print("\n🔥 TOP 20 MOST IMPORTANT FEATURES:")
print("="*60)
for idx, row in feature_importance.head(20).iterrows():
    bar = "█" * int(row['importance'] * 100)
    print(f"{row['feature']:35s} {bar} {row['importance']:.4f}")

print("\n📉 BOTTOM 10 LEAST IMPORTANT FEATURES:")
print("="*60)
for idx, row in feature_importance.tail(10).iterrows():
    print(f"{row['feature']:35s} {row['importance']:.4f}")

# Error Analysis
print("\n[2/5] Error Analysis...")
print("-"*60)

y_pred_proba = xgb.predict_proba(X_test)[:, 1]
y_pred = xgb.predict(X_test)

test_df_analysis = test_df.copy()
test_df_analysis['Predicted'] = y_pred
test_df_analysis['PredictedProba'] = y_pred_proba
test_df_analysis['Correct'] = (y_pred == y_test)

true_positives = test_df_analysis[(y_test == True) & (y_pred == True)]
false_positives = test_df_analysis[(y_test == False) & (y_pred == True)]
false_negatives = test_df_analysis[(y_test == True) & (y_pred == False)]
true_negatives = test_df_analysis[(y_test == False) & (y_pred == False)]

print(f"\n📊PREDICTION BREAKDOWN:")
print(f"  ✅ True Positives:  {len(true_positives):,}")
print(f"  ❌ False Positives: {len(false_positives):,}")
print(f"  ❌ False Negatives: {len(false_negatives):,}")
print(f"  ✅ True Negatives:  {len(true_negatives):,}")

# Race-by-Race Performance
print("\n[5/5] Race-by-Race Performance...")
print("-"*60)

race_performance = []

for race_id in test_df_analysis['RaceID'].unique():
    race_data = test_df_analysis[test_df_analysis['RaceID'] == race_id]
    
    actual_pits = (race_data[target_variable] == True).sum()
    predicted_pits = (race_data['Predicted'] == True).sum()
    correct_pits = ((race_data[target_variable] == True) & (race_data['Predicted'] == True)).sum()
    
    if actual_pits > 0:
        precision = correct_pits / predicted_pits if predicted_pits > 0 else 0
        recall = correct_pits / actual_pits
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    else:
        precision = recall = f1 = 0
    
    race_performance.append({
        'RaceID': race_id,
        'Year': race_data['Year'].iloc[0],
        'RaceName': race_data['RaceName'].iloc[0],
        'ActualPits': actual_pits,
        'PredictedPits': predicted_pits,
        'CorrectPits': correct_pits,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

race_perf_df = pd.DataFrame(race_performance).sort_values('F1-Score')

print("\n BEST PERFORMING RACES:")
print("="*60)
for idx, row in race_perf_df.tail(5).iterrows():
    print(f"{row['Year']} {row['RaceName']:30s} F1={row['F1-Score']:.2f}")

print("\n WORST PERFORMING RACES:")
print("="*60)
for idx, row in race_perf_df.head(5).iterrows():
    print(f"{row['Year']} {row['RaceName']:30s} F1={row['F1-Score']:.2f} (Actual:{row['ActualPits']}, Pred:{row['PredictedPits']}, Correct:{row['CorrectPits']})")

print("\n" + "="*60)
print("ANALYSIS COMPLETE!")
print("="*60)