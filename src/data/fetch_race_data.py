import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import sys

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from config.track_characteristics import get_track_info

def setup_fastf1_cache():
    cache_dir = Path(__file__).parent.parent.parent / 'data' / 'cache' / 'fastf1'
    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))

def get_lap_data(session):
    # Get all laps from the session
    laps = session.laps
    
    columns_to_keep = [
        'Time',           # Timestamp of lap
        'LapNumber',      # Which lap (1, 2, 3, ...)
        'LapTime',        # Total lap time
        'Compound',       # Tire compound (SOFT, MEDIUM, HARD)
        'TyreLife',       # Age of current tires (laps)
        'Driver',         # Driver 3-letter code (VER, HAM, etc)
        'Team',           # Team name
        'Position',       # Position in race (1st, 2nd, etc)
        'Stint',          # Stint number (changes at pit stops)
        'Sector1Time',    # Sector 1 time
        'Sector2Time',    # Sector 2 time
        'Sector3Time',    # Sector 3 time
        'FreshTyre',      # Boolean: new tire or used?
        'TrackStatus',    # Track condition (green, SC, VSC, etc)
        'PitInTime',      # Time entered pit lane
        'PitOutTime'      # Time exited pit lane
    ]
    
    available_columns = [col for col in columns_to_keep if col in laps.columns]
    
    # Create a copy of the data with selected columns
    lap_data = laps[available_columns].copy()
    
    # Convert Timedelta columns to seconds (easier to work with)
    # Timedelta format: "0 days 00:01:23.456000"
    # Seconds format: 83.456
    time_columns = ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time', 'PitInTime', 'PitOutTime']
    
    for col in time_columns:
        if col in lap_data.columns:
            # .dt.total_seconds() converts Timedelta to float seconds
            lap_data[col] = lap_data[col].dt.total_seconds()
    
    return lap_data

def add_weather_per_lap(lap_data, session):
    """
    Match weather data to each lap.
    
    Why per-lap instead of session average?
    - Rain starts lap 15 → Strategy changes completely
    - Temperature drops 10°C → Tire behavior changes
    - We need to see weather EVOLUTION during race
    
    How it works:
    1. Get weather data (sampled every ~1 minute)
    2. For each lap, find closest weather reading by timestamp
    3. Add weather columns to lap data
    
    Returns:
        lap_data with added columns: AirTemp, TrackTemp, Humidity, Rainfall
    """
    try:
        # Get weather data from session
        weather = session.weather_data
        
        # Check if weather data exists
        if weather.empty:
            print("  ⚠ No weather data available")
            # Add empty columns
            lap_data['AirTemp'] = np.nan
            lap_data['TrackTemp'] = np.nan
            lap_data['Humidity'] = np.nan
            lap_data['Rainfall'] = False
            return lap_data
        
        # We'll store weather for each lap
        weather_features = []
        
        # Loop through each lap
        for idx, lap in lap_data.iterrows():
            # Get the timestamp of this lap
            lap_time = lap['Time']
            
            # Find the closest weather reading to this lap time
            # weather['Time'] - lap_time = time difference for each weather reading
            # abs() = absolute value (we don't care if before/after)
            time_diff = abs(weather['Time'] - lap_time)
            
            # idxmin() = index of minimum value
            # This gives us the weather reading closest in time
            closest_idx = time_diff.idxmin()
            
            # Get that weather reading
            closest_weather = weather.loc[closest_idx]
            
            # Extract the weather values we want
            weather_features.append({
                'AirTemp': closest_weather['AirTemp'],
                'TrackTemp': closest_weather['TrackTemp'],
                'Humidity': closest_weather['Humidity'],
                'Rainfall': closest_weather['Rainfall']
            })
        
        # Convert list of dicts to DataFrame
        weather_df = pd.DataFrame(weather_features)
        
        # Combine lap_data with weather_df (side by side)
        lap_data = pd.concat([
            lap_data.reset_index(drop=True), 
            weather_df.reset_index(drop=True)
        ], axis=1)
        
        
        # Print useful info about weather changes -  for debugging
        if lap_data['Rainfall'].any():
            # .any() returns True if ANY value is True
            rain_laps = lap_data[lap_data['Rainfall'] == True]['LapNumber'].tolist()
            print(f" Rain detected on laps: {rain_laps}")
        
        temp_change = lap_data['TrackTemp'].max() - lap_data['TrackTemp'].min()
        if temp_change > 5:
            print(f"Track temp changed by {temp_change:.1f}°C during race")
        
        return lap_data
        
    except Exception as e:
        print(f"Could not add weather data: {e}")
        # Add empty columns if error
        lap_data['AirTemp'] = np.nan
        lap_data['TrackTemp'] = np.nan
        lap_data['Humidity'] = np.nan
        lap_data['Rainfall'] = False
        return lap_data
    


def aggregate_telemetry_per_lap(session, lap_data):
    """
    Aggregate telemetry data per lap.
    
    **Features extracted:**
    - AvgSpeed, MaxSpeed: Overall pace
    - AvgAcceleration, MaxAcceleration: ERS proxy, aggression indicator
    - AvgThrottle, ThrottleVariance: Driving style (smooth vs aggressive)
    - AvgBrake, MaxBrake: Braking patterns
    - DRS_count: DRS activations (overtaking attempts)
    - AvgRPM: Engine usage
    - GearChanges: Driving activity
    
    Returns:
        DataFrame with telemetry features
    """
    print("  Aggregating telemetry data per lap...")
    telemetry_features = []
    total_laps = len(lap_data)
    
    success_count = 0
    error_count = 0

    for idx in tqdm(range(total_laps), desc="  Processing laps", leave=False):
        try:
            # Get the lap object
            lap_obj = session.laps.iloc[idx]
            
            # Get telemetry (300-500 samples per lap)
            telemetry = lap_obj.get_telemetry()

            # Check if telemetry exists
            if telemetry is None or telemetry.empty:
                telemetry_features.append({
                    'AvgSpeed': np.nan,
                    'MaxSpeed': np.nan,
                    'AvgThrottle': np.nan,
                    'AvgBrake': np.nan,
                    'MaxBrake': np.nan,
                    'DRS_count': 0,
                    'AvgAcceleration': np.nan,
                    'MaxAcceleration': np.nan,
                    'ThrottleVariance': np.nan,
                    'AvgRPM': np.nan,
                    'GearChanges': 0
                })
                error_count += 1
                continue

            # === CALCULATE ACCELERATION ===
            # Acceleration = change in speed / change in time
            speed_diff = telemetry['Speed'].diff()
            time_diff = telemetry['Time'].diff().dt.total_seconds()
            telemetry['Acceleration'] = speed_diff / time_diff
            
            # === AGGREGATE FEATURES ===
            features = {
                # Speed metrics
                'AvgSpeed': telemetry['Speed'].mean(),
                'MaxSpeed': telemetry['Speed'].max(),
                
                # Throttle metrics
                'AvgThrottle': telemetry['Throttle'].mean(),
                'ThrottleVariance': telemetry['Throttle'].std(),
                
                # Brake metrics (Brake is boolean, convert to numeric)
                'AvgBrake': telemetry['Brake'].astype(float).mean(),
                'MaxBrake': telemetry['Brake'].astype(float).max(),
                
                # DRS activations
                'DRS_count': (telemetry['DRS'] > 0).sum(),
                
                # Acceleration (ERS proxy)
                'AvgAcceleration': telemetry['Acceleration'].mean(),
                'MaxAcceleration': telemetry['Acceleration'].max(),
                
                # RPM (engine usage)
                'AvgRPM': telemetry['RPM'].mean() if 'RPM' in telemetry.columns else np.nan,
                
                # Gear changes (driving activity)
                'GearChanges': telemetry['nGear'].diff().abs().sum() if 'nGear' in telemetry.columns else 0
            }
            
            telemetry_features.append(features)
            success_count += 1
            
        except Exception as e:
            # Print first 3 errors for debugging
            if error_count < 3:
                print(f"Error on lap {idx}: {e}")
            
            telemetry_features.append({
                'AvgSpeed': np.nan,
                'MaxSpeed': np.nan,
                'AvgThrottle': np.nan,
                'AvgBrake': np.nan,
                'MaxBrake': np.nan,
                'DRS_count': 0,
                'AvgAcceleration': np.nan,
                'MaxAcceleration': np.nan,
                'ThrottleVariance': np.nan,
                'AvgRPM': np.nan,
                'GearChanges': 0
            })
            error_count += 1

    # Convert to DataFrame
    telemetry_df = pd.DataFrame(telemetry_features)
    
    print(f"  Telemetry aggregation complete.")
    print(f"Success: {success_count}/{total_laps} laps ({success_count/total_laps*100:.1f}%)")
    
    if error_count > 0:
        print(f"Errors: {error_count}/{total_laps} laps")
    
    # Show sample statistics
    if success_count > 0:
        print(f"    Sample stats:")
        print(f"      AvgSpeed: {telemetry_df['AvgSpeed'].mean():.1f} km/h")
        print(f"      AvgThrottle: {telemetry_df['AvgThrottle'].mean():.1f}%")
        print(f"      AvgAcceleration: {telemetry_df['AvgAcceleration'].mean():.2f} km/h/s")
    
    return telemetry_df

def calculate_gaps(lap_data):
    """
    Calculate time gap to car ahead and car behind for each lap.
    
    Why this is CRITICAL for strategy:
    
    Example 1: Large gap behind (15s)
    → Can pit and still maintain position
    → Conservative strategy viable
    
    Example 2: Small gap behind (2s)
    → Pit = lose position
    → Must stay out and push
    
    Example 3: Small gap ahead (3s)
    → Can undercut (pit early to overtake)
    → Aggressive strategy pays off
    
    How it works:
    1. For each lap, look at all drivers on that lap
    2. For each driver, find car ahead (position - 1)
    3. Calculate cumulative time difference
    4. Repeat for car behind (position + 1)
    
    Returns:
        lap_data with GapToAhead and GapToBehind columns added
    """
    print("  → Calculating gaps to cars ahead/behind...")
    
    # Sort data by lap number and position
    lap_data = lap_data.sort_values(['LapNumber', 'Position']).reset_index(drop=True)
    
    # Initialize gap columns with NaN
    lap_data['GapToAhead'] = np.nan
    lap_data['GapToBehind'] = np.nan
    
    # Process each lap number
    for lap_num in tqdm(lap_data['LapNumber'].unique(), desc="  Calculating gaps", leave=False):
        
        # Get all cars on this specific lap
        lap_subset = lap_data[lap_data['LapNumber'] == lap_num].copy()
        
        # Skip if no valid lap times on this lap
        if lap_subset['LapTime'].isna().all():
            continue
        
        # For each driver on this lap
        for idx, row in lap_subset.iterrows():
            
            # Get current driver's position
            current_pos = row['Position']
            
            # Skip if position is missing
            if pd.isna(current_pos):
                continue
            
            current_driver = row['Driver']
            
            
            if current_pos > 1:  
                # Find the car in position ahead (current_pos - 1)
                car_ahead_data = lap_subset[lap_subset['Position'] == current_pos - 1]
                
                if not car_ahead_data.empty:
                    driver_ahead = car_ahead_data.iloc[0]['Driver']
                    
                    # Calculate cumulative time gap
                    gap_ahead = calculate_cumulative_gap(
                        lap_data, lap_num, current_driver, driver_ahead
                    )
                    
                    lap_data.loc[idx, 'GapToAhead'] = gap_ahead
            else:
                lap_data.loc[idx, 'GapToAhead'] = 0.0
            
            # Find car in position behind (current_pos + 1)
            car_behind_data = lap_subset[lap_subset['Position'] == current_pos + 1]
            
            if not car_behind_data.empty:
                driver_behind = car_behind_data.iloc[0]['Driver']
                
                # Calculate cumulative time gap
                gap_behind = calculate_cumulative_gap(
                    lap_data, lap_num, driver_behind, current_driver
                )
                
                lap_data.loc[idx, 'GapToBehind'] = gap_behind
            else:
                lap_data.loc[idx, 'GapToBehind'] = 0.0
    
    print(f"Gaps calculated")
    
    return lap_data


def calculate_cumulative_gap(lap_data, current_lap, driver_behind, driver_ahead):
    """
    Calculate cumulative time gap between two drivers at a specific lap.
    
    How gaps work in F1:
    - It's "total time difference from start of race"
    
    Example:
    Lap 1: VER 75.2s, HAM 75.5s → HAM +0.3s behind
    Lap 2: VER 74.8s, HAM 75.0s → HAM +0.5s behind (0.3 + 0.2)
    Lap 3: VER 75.1s, HAM 74.9s → HAM +0.4s behind (0.5 - 0.1)
    
    So we sum ALL lap times up to current lap for both drivers.
    
    Args:
        lap_data: Full lap data DataFrame
        current_lap: Lap number we're calculating for
        driver_behind: Driver code for car behind
        driver_ahead: Driver code for car ahead
    
    Returns:
        Time gap in seconds (positive = driver_behind is behind)
    """
    # Get all laps UP TO current lap for driver ahead
    laps_ahead = lap_data[
        (lap_data['Driver'] == driver_ahead) & 
        (lap_data['LapNumber'] <= current_lap)
    ]['LapTime']
    
    # Get all laps UP TO current lap for driver behind
    laps_behind = lap_data[
        (lap_data['Driver'] == driver_behind) & 
        (lap_data['LapNumber'] <= current_lap)
    ]['LapTime']
    
    # If either driver has no data, return NaN
    if laps_ahead.empty or laps_behind.empty:
        return np.nan
    
    # If they have different number of laps (e.g., one pitted), return NaN
    if len(laps_ahead) != len(laps_behind):
        return np.nan
    
    # Sum all lap times (cumulative race time)
    total_time_ahead = laps_ahead.sum()
    total_time_behind = laps_behind.sum()
    
    # Gap = how much slower driver_behind is
    gap = total_time_behind - total_time_ahead
    
    return gap

def add_track_status_flag(lap_data):
    print("  → Adding track status flags...")
    """
    Parse TrackStatus codes into readable boolean flags.
    
    Why this is CRITICAL:
    
    Scenario 1: Safety Car on lap 20
    → Pit now = minimal time loss (everyone slows down)
    → "Free" pit stop changes everything!
    
    Scenario 2: VSC on lap 30
    → Pit = lose ~50% less time than normal
    → Massive strategic advantage
    
    Scenario 3: Yellow flag
    → Can't overtake in that sector
    → Affects racing and tire management
    
    TrackStatus codes (from FastF1):
    '1' = Green flag (normal racing)
    '2' = Yellow flag (danger, no overtaking)
    '4' = Safety Car (SC)
    '5' = Red flag (race stopped)
    '6' = Virtual Safety Car (VSC)
    '7' = VSC ending (about to go green)
    
    Returns:
        lap_data with boolean columns: IsSafetyCar, IsVSC, IsYellowFlag, IsGreen
    """
    print("  → Parsing track status flags...")

    lap_data['TrackStatus'] = lap_data['TrackStatus'].astype(str)

    lap_data['IsSafetyCar'] = lap_data['TrackStatus'].isin(['4'])
    lap_data['IsVSC'] = lap_data['TrackStatus'].isin(['6', '7'])
    lap_data['IsYellowFlag'] = lap_data['TrackStatus'].isin(['2'])
    lap_data['IsGreen'] = lap_data['TrackStatus'].isin(['1'])

    sc_lap = lap_data[lap_data['IsSafetyCar']]['LapNumber'].unique()
    vsc_lap = lap_data[lap_data['IsVSC']]['LapNumber'].unique()
    yellow_lap = lap_data[lap_data['IsYellowFlag']]['LapNumber'].unique()

    if len(sc_lap) > 0:
        print(f"   Safety Car on laps: {sc_lap.tolist()}")
    if len(vsc_lap) > 0:
        print(f"   VSC on laps: {vsc_lap.tolist()}")
    if len(yellow_lap) > 0:
        print(f"   Yellow flags on laps: {yellow_lap.tolist()}")

    if len(sc_lap) == 0 and len(vsc_lap) == 0:
        print("   No Safety Car or VSC periods detected.")

    return lap_data

def add_starting_pos(lap_data, session):
    """
    Add starting grid position for each driver.
    
    Why this matters:
    
    Starting P1 (pole position):
    → Clean air ahead
    → Conservative strategy works (protect position)
    → 1-stop often viable
    
    Starting P15 (midfield):
    → Stuck in traffic
    → Need aggressive strategy to move up
    → 2-stop with fresher tires
    → Or alternative strategy (opposite of leaders)
    
    Starting P20 (back of grid):
    → Nothing to lose
    → High-risk strategies (long first stint, multiple stops)
    → Gamble on weather, safety car timing
    
    How it works:
    1. Get race results from session (includes grid positions)
    2. Create mapping: Driver → GridPosition
    3. Add as column to lap_data
    
    Returns:
        lap_data with StartingPosition column added
    """
    print("  → Adding starting grid positions...")
    try:
        results = session.results
        grid_positions = {}

        for idx, row in results.iterrows():
            driver = row['Abbreviation']
            grid_pos = row['GridPosition']
            grid_positions[driver] = grid_pos

            lap_data['StartingPosition'] = lap_data['Driver'].map(grid_positions)
        print(f" Starting positions added for {len(grid_positions)} drivers")
        return lap_data

    except Exception as e:
        print(f"Could not add starting positions: {e}")
        lap_data['StartingPosition'] = np.nan
        return lap_data
    
def extract_driver_from_message(message):
    """
    Extract driver code from race control message.
    
    Message formats:
    - "CAR 44 (HAM) - ..."
    - "CAR 33 - ..."
    - "NO. 16 - ..."
    
    Returns:
        Driver 3-letter code (e.g., 'HAM') or None
    """
    import re
    
    # Try to extract driver code in parentheses
    # Pattern: (ABC) where ABC is 3 letters
    match = re.search(r'\(([A-Z]{3})\)', message)
    if match:
        return match.group(1)
    
    # If no driver code, try to extract car number
    # Pattern: CAR 44 or NO. 44
    match = re.search(r'(?:CAR|NO\.?)\s+(\d+)', message)
    if match:
        car_number = match.group(1)
        # Note: We'd need to map car number to driver
        # For now, return car number format
        return f"CAR_{car_number}"
    
    return None


def classify_penalty_type(message):
    """
    Determine penalty type from message text.
    
    Returns:
        One of: '5_SECOND', '10_SECOND', 'STOP_AND_GO', 
                'DRIVE_THROUGH', 'TIME_ADDED', 'UNKNOWN'
    """
    message_upper = message.upper()
    
    # Check for each penalty type
    # Order matters! Check specific before general
    
    if 'STOP' in message_upper and 'GO' in message_upper:
        return 'STOP_AND_GO'
    
    if 'DRIVE' in message_upper and 'THROUGH' in message_upper:
        return 'DRIVE_THROUGH'
    
    if '10' in message and 'SEC' in message_upper:
        return '10_SECOND'
    
    if '5' in message and 'SEC' in message_upper:
        return '5_SECOND'
    
    # Sometimes penalties are added post-race
    if 'TIME' in message_upper and 'ADD' in message_upper:
        return 'TIME_ADDED'
    
    # Catch-all
    return 'UNKNOWN'
    
def add_penalty_information(lap_data, session):
    """
    Extract and track penalty status throughout race.
    
    NEW: Tracks whether penalty was served or deferred
    """
    print("  → Extracting penalty information...")
    
    try:
        messages = session.race_control_messages
        
        # Get penalty issued messages
        penalty_issued = messages[
            (messages['Category'] == 'Penalty') | 
            (messages['Message'].str.contains('PENALTY', case=False, na=False))
        ]
        
        # Get penalty served messages  
        penalty_served = messages[
            messages['Message'].str.contains('SERVED', case=False, na=False)
        ]
        
        # Parse penalties
        penalties = {}  # {driver: {type, seconds, issued_time, served}}

        session_start_timestamp = pd.Timestamp(session.event.EventDate)
        
        for idx, msg in penalty_issued.iterrows():
            message_text = msg['Message']
            penalty_time = msg['Time']
            
            driver = extract_driver_from_message(message_text)
            if driver is None:
                continue
            
            penalty_type = classify_penalty_type(message_text)
            penalty_seconds = extract_penalty_seconds(message_text)
            penalty_time_delta = penalty_time - session_start_timestamp
            
            if driver not in penalties:
                penalties[driver] = []
            
            penalties[driver].append({
                'type': penalty_type,
                'seconds': penalty_seconds,
                'issued_time': penalty_time_delta,
                'served_during_race': False,  # Will update if served
                'served_time': None
            })
        
        # Check which penalties were served during race
        for idx, msg in penalty_served.iterrows():
            message_text = msg['Message']
            served_time = msg['Time']
            served_time_delta = served_time - session_start_timestamp
            
            driver = extract_driver_from_message(message_text)
            if driver and driver in penalties:
                # Mark most recent penalty as served
                for penalty in reversed(penalties[driver]):
                    if not penalty['served_during_race']:
                        penalty['served_during_race'] = True
                        penalty['served_time'] = served_time_delta
                        break
        
        # Add penalty info to each lap
        lap_data['HasPendingPenalty'] = False
        lap_data['PenaltySeconds'] = 0
        lap_data['PenaltyType'] = None
        
        # For each driver with penalties
        for driver, penalty_list in penalties.items():
            driver_laps = lap_data[lap_data['Driver'] == driver]
            
            for penalty in penalty_list:
                issued_time = penalty['issued_time']
                
                if penalty['served_during_race']:
                    # Penalty was served during race
                    served_time = penalty['served_time']
                    
                    # Mark laps BETWEEN issue and serve as having pending penalty
                    mask = (
                        (lap_data['Driver'] == driver) &
                        (lap_data['Time'] >= issued_time) &
                        (lap_data['Time'] < served_time)
                    )
                    lap_data.loc[mask, 'HasPendingPenalty'] = True
                    lap_data.loc[mask, 'PenaltySeconds'] = penalty['seconds']
                    lap_data.loc[mask, 'PenaltyType'] = penalty['type']
                    
                else:
                    # Penalty NOT served - added post-race
                    # Mark all laps AFTER penalty issued as having pending penalty
                    mask = (
                        (lap_data['Driver'] == driver) &
                        (lap_data['Time'] >= issued_time)
                    )
                    lap_data.loc[mask, 'HasPendingPenalty'] = True
                    lap_data.loc[mask, 'PenaltySeconds'] = penalty['seconds']
                    lap_data.loc[mask, 'PenaltyType'] = penalty['type']
        
        print(f"  ✓ Tracked {len(penalties)} driver(s) with penalties")
        
        return penalties
        
    except Exception as e:
        print(f"  ⚠ Could not extract penalties: {e}")
        return {}


def extract_penalty_seconds(message):
    """
    Extract penalty duration in seconds from message.
    
    Examples:
    "5 SEC TIME PENALTY" → 5
    "10 SECOND PENALTY" → 10
    
    Returns:
        Integer seconds, or 0 if can't parse
    """
    import re
    
    # Look for pattern: number + "SEC" or "SECOND"
    match = re.search(r'(\d+)\s*SEC', message.upper())
    if match:
        return int(match.group(1))
    
    return 0

def add_pit_stop_info(lap_data, penalties_dict):
    """
    Identify pit laps and calculate duration, accounting for penalties.
    
    Pit stop types we classify:
    
    1. NORMAL (2-4s):
       - Regular tire change
       - Fast, clean stop
    
    2. NORMAL_SLOW (4-6s):
       - Slower tire change (wheel gun issue, etc)
       - Still just tire change
    
    3. TIME_PENALTY (6-15s):
       - 5s or 10s penalty served
       - Includes tire change
       - Duration = penalty + tire change
    
    4. STOP_AND_GO (20-30s):
       - Stop for 10s, can't work on car
       - Usually no tire change
       - Separate from strategy pit
    
    5. SLOW_OR_ISSUE (30s+):
       - Technical problem
       - Very slow stop
       - Or car retired in pits
    
    Args:
        lap_data: DataFrame with lap information
        penalties_dict: Dictionary from add_penalty_information()
    
    Returns:
        lap_data with columns: IsPitLap, PitDuration, PitType, HasPenaltyServed
    """
    print("  → Identifying pit stops and classifying types...")
    
    # Initialize columns
    lap_data['IsPitLap'] = False
    lap_data['PitDuration'] = np.nan
    lap_data['PitType'] = None
    lap_data['HasPenaltyServed'] = False
    
    # Process each driver separately
    for driver in lap_data['Driver'].unique():
        
        # Get this driver's laps
        driver_laps = lap_data[lap_data['Driver'] == driver].copy()
        
        # Shift Stint to compare with previous lap
        driver_laps['PrevStint'] = driver_laps['Stint'].shift(1)
        
        # Detect pit laps (where stint changes)
        pit_lap_mask = driver_laps['Stint'] != driver_laps['PrevStint']
        
        # Get indices of pit laps
        pit_lap_indices = driver_laps[pit_lap_mask].index
        
        # Mark as pit laps
        if len(pit_lap_indices) > 0:
            lap_data.loc[pit_lap_indices, 'IsPitLap'] = True
            
            # Calculate duration if possible
            if 'PitInTime' in lap_data.columns and 'PitOutTime' in lap_data.columns:
                
                for idx in pit_lap_indices:
                    pit_in = lap_data.loc[idx, 'PitInTime']
                    pit_out = lap_data.loc[idx, 'PitOutTime']
                    
                    # Check if times exist
                    if pd.notna(pit_in) and pd.notna(pit_out):
                        
                        # Convert to seconds if needed
                        if hasattr(pit_in, 'total_seconds'):
                            pit_in = pit_in.total_seconds()
                        if hasattr(pit_out, 'total_seconds'):
                            pit_out = pit_out.total_seconds()
                        
                        # Calculate duration
                        duration = pit_out - pit_in
                        
                        # Sanity check (remove clearly wrong data)
                        # Accept 1-120 seconds (very wide range)
                        if 1 <= duration <= 120:
                            lap_data.loc[idx, 'PitDuration'] = duration
                            
                            # Classify pit stop type
                            pit_type = classify_pit_stop_type(
                                duration, 
                                driver, 
                                penalties_dict
                            )
                            lap_data.loc[idx, 'PitType'] = pit_type
                            
                            # Check if this pit served a penalty
                            if 'PENALTY' in pit_type:
                                lap_data.loc[idx, 'HasPenaltyServed'] = True
    
    # Print summary statistics
    total_pits = lap_data['IsPitLap'].sum()
    pits_with_duration = lap_data['PitDuration'].notna().sum()
    
    print(f"  ✓ Identified {total_pits} pit stops")
    
    if pits_with_duration > 0:
        print(f"    → {pits_with_duration} with duration data")
        
        # Average duration by type
        for pit_type in lap_data['PitType'].unique():
            if pd.notna(pit_type):
                type_durations = lap_data[lap_data['PitType'] == pit_type]['PitDuration']
                if not type_durations.empty:
                    avg = type_durations.mean()
                    count = len(type_durations)
                    print(f"    → {pit_type}: {count} stops, avg {avg:.2f}s")
    
    return lap_data


def classify_pit_stop_type(duration, driver, penalties_dict):
    """
    Classify pit stop based on duration and penalty information.
    
    Args:
        duration: Pit stop duration in seconds
        driver: Driver code
        penalties_dict: Dictionary of penalties
    
    Returns:
        Pit stop type string
    """
    # Check if driver has penalties
    has_penalty = driver in penalties_dict and len(penalties_dict[driver]) > 0
    
    # Classification based on duration
    
    if duration < 4:
        # Very fast - normal pit stop
        return 'NORMAL'
    
    elif 4 <= duration < 6:
        # Slightly slow - could be normal or wheel issue
        return 'NORMAL_SLOW'
    
    elif 6 <= duration < 15:
        # Longer - likely time penalty or slow stop
        if has_penalty:
            # Check penalty types
            for penalty in penalties_dict[driver]:
                if penalty['type'] in ['5_SECOND', '10_SECOND']:
                    return f"NORMAL_WITH_{penalty['type']}_PENALTY"
        
        # No penalty info, but duration suggests penalty
        if 6 <= duration < 9:
            return 'LIKELY_5_SEC_PENALTY'
        elif 9 <= duration < 15:
            return 'LIKELY_10_SEC_PENALTY'
        
        return 'SLOW_STOP_OR_PENALTY'
    
    elif 15 <= duration < 35:
        # Very long - likely stop-and-go or drive-through
        if has_penalty:
            for penalty in penalties_dict[driver]:
                if penalty['type'] == 'STOP_AND_GO':
                    return 'STOP_AND_GO_PENALTY'
                elif penalty['type'] == 'DRIVE_THROUGH':
                    return 'DRIVE_THROUGH_PENALTY'
        
        return 'LIKELY_STOP_AND_GO_OR_ISSUE'
    
    else:  # duration >= 35
        # Extremely long - technical issue or retirement
        return 'TECHNICAL_ISSUE_OR_DNF'

def extract_damage_from_messages(session):
    """
    Extract damage information from race control messages.
    
    What we're looking for:
    - "CAR 33 - FRONT WING DAMAGE"
    - "CAR 44 - PUNCTURE"
    - "CAR 16 - CONTACT WITH CAR 55"
    - "DEBRIS ON TRACK FROM CAR 4"
    
    Damage types and strategic impact:
    
    FRONT_WING:
    → Lose 1-2s per lap
    → Must pit for replacement
    → Usually combine with tire change
    
    FLOOR:
    → Lose 0.5-1.5s per lap
    → Hard to repair (might be permanent)
    → Affects balance and tire wear
    
    PUNCTURE:
    → Must pit immediately
    → Lose 30-60s total
    → Race effectively over
    
    SUSPENSION:
    → Usually DNF
    → Can't continue safely
    
    Returns:
        Dictionary: {driver: [damage_events]}
    """
    print("  → Extracting damage from race control messages...")
    
    try:
        # Get all race control messages
        messages = session.race_control_messages
        
        # Keywords that indicate damage
        damage_keywords = [
            'DAMAGE', 'DAMAGED',
            'WING', 'FRONT WING', 'REAR WING',
            'PUNCTURE', 'PUNCTURED', 'FLAT', 'FLAT TYRE',
            'DEBRIS', 'LOOSE',
            'CONTACT', 'COLLISION', 'CRASH', 'HIT',
            'FLOOR', 'SIDEPOD',
            'SUSPENSION', 'BROKEN'
        ]
        
        damage_events = {}
        
        # Check each message
        for idx, msg in messages.iterrows():
            message_text = msg['Message']
            message_time = msg['Time']
            
            # Convert message to uppercase for checking
            message_upper = message_text.upper()
            
            # Check if any damage keyword is present
            has_damage_keyword = any(
                keyword in message_upper 
                for keyword in damage_keywords
            )
            
            if has_damage_keyword:
                # Extract driver from message
                driver = extract_driver_from_message(message_text)
                
                if driver:
                    # Classify damage type
                    damage_type = classify_damage_type(message_text)

                    session_start_timestamp = pd.Timestamp(session.event.EventDate)
                    message_time_delta = message_time - session_start_timestamp
                    
                    # Initialize driver's damage list if needed
                    if driver not in damage_events:
                        damage_events[driver] = []
                    
                    # Add damage event
                    damage_events[driver].append({
                        'type': damage_type,
                        'time': message_time_delta,
                        'message': message_text
                    })
        
        # Print summary
        if damage_events:
            print(f"  ⚠️ Damage detected for {len(damage_events)} driver(s):")
            for driver, events in damage_events.items():
                for event in events:
                    print(f"    → {driver}: {event['type']}")
        else:
            print("  ✓ No damage reported in race control messages")
        
        return damage_events
        
    except Exception as e:
        print(f"  ⚠ Could not extract damage information: {e}")
        return {}


def classify_damage_type(message):
    """
    Classify type of damage from message text.
    
    Args:
        message: Race control message text
    
    Returns:
        Damage type string
    """
    message_upper = message.upper()
    
    # Check for specific damage types
    # Order matters - check specific before general!
    
    if 'FRONT' in message_upper and 'WING' in message_upper:
        return 'FRONT_WING_DAMAGE'
    
    if 'REAR' in message_upper and 'WING' in message_upper:
        return 'REAR_WING_DAMAGE'
    
    if 'PUNCTURE' in message_upper or 'FLAT' in message_upper:
        return 'PUNCTURE'
    
    if 'FLOOR' in message_upper:
        return 'FLOOR_DAMAGE'
    
    if 'SIDEPOD' in message_upper:
        return 'SIDEPOD_DAMAGE'
    
    if 'SUSPENSION' in message_upper:
        return 'SUSPENSION_DAMAGE'
    
    if 'DEBRIS' in message_upper:
        return 'DEBRIS_OR_PARTS_LOSS'
    
    if any(word in message_upper for word in ['CONTACT', 'COLLISION', 'CRASH', 'HIT']):
        return 'CONTACT_POSSIBLE_DAMAGE'
    
    if 'DAMAGE' in message_upper:
        return 'GENERAL_DAMAGE'
    
    return 'UNSPECIFIED_ISSUE'

def detect_damage_from_telemetry(lap_data):
    """
    Detect potential car damage from telemetry anomalies.
    
    How we detect damage:
    
    1. Sudden lap time increase (>1.5s)
       → Not explained by tire degradation
       → Not explained by traffic
       → Persists for multiple laps
    
    2. Sudden speed drop (>10 km/h average)
       → Across entire lap
       → Not just one sector
    
    3. Sudden sector time increase
       → One sector much slower
       → Indicates localized issue
    
    What we DON'T flag as damage:
    - Normal tire degradation (gradual slowdown)
    - Yellow flags / Safety Car (TrackStatus shows this)
    - First lap (always slower)
    - Pit in/out laps (expected to be slow)
    
    Returns:
        lap_data with 'SuspectedDamage' column added
    """
    print("  → Detecting potential damage from telemetry...")
    
    # Initialize column
    lap_data['SuspectedDamage'] = False
    lap_data['DamageConfidence'] = 0.0  # 0-1 score
    
    # Process each driver separately
    for driver in lap_data['Driver'].unique():
        
        # Get this driver's laps
        driver_laps = lap_data[lap_data['Driver'] == driver].copy()
        
        # Skip if not enough laps
        if len(driver_laps) < 5:
            continue
        
        # Calculate rolling averages (3-lap window)
        driver_laps['LapTime_Rolling'] = driver_laps['LapTime'].rolling(
            window=3, 
            min_periods=1
        ).mean()
        
        driver_laps['AvgSpeed_Rolling'] = driver_laps['AvgSpeed'].rolling(
            window=3, 
            min_periods=1
        ).mean()
        
        # Check each lap (starting from lap 4)
        for i in range(3, len(driver_laps)):
            
            current_lap = driver_laps.iloc[i]
            prev_avg = driver_laps.iloc[i-1]
            
            # Skip if data missing
            if pd.isna(current_lap['LapTime']) or pd.isna(prev_avg['LapTime_Rolling']):
                continue
            
            # Skip pit laps and laps with yellow flags
            if current_lap['IsPitLap'] or current_lap['IsYellowFlag'] or current_lap['IsSafetyCar']:
                continue
            
            # === CHECK 1: Sudden Lap Time Increase ===
            lap_time_increase = current_lap['LapTime'] - prev_avg['LapTime_Rolling']
            
            # Expected increase from tire degradation
            # Assume ~0.05s per lap of tire age (conservative)
            expected_deg = current_lap['TyreLife'] * 0.05
            
            # Unexpected slowdown
            unexpected_slowdown = lap_time_increase - expected_deg
            
            if unexpected_slowdown > 1.5:
                # Significant unexplained slowdown
                confidence = min(unexpected_slowdown / 3.0, 1.0)  # Cap at 1.0
                
                lap_data.loc[current_lap.name, 'SuspectedDamage'] = True
                lap_data.loc[current_lap.name, 'DamageConfidence'] = confidence
            
            # === CHECK 2: Sudden Speed Drop ===
            if pd.notna(current_lap['AvgSpeed']) and pd.notna(prev_avg['AvgSpeed_Rolling']):
                
                speed_drop = prev_avg['AvgSpeed_Rolling'] - current_lap['AvgSpeed']
                
                if speed_drop > 10:  # 10+ km/h drop
                    confidence = min(speed_drop / 20.0, 1.0)
                    
                    # If already flagged, take max confidence
                    if lap_data.loc[current_lap.name, 'SuspectedDamage']:
                        old_conf = lap_data.loc[current_lap.name, 'DamageConfidence']
                        lap_data.loc[current_lap.name, 'DamageConfidence'] = max(old_conf, confidence)
                    else:
                        lap_data.loc[current_lap.name, 'SuspectedDamage'] = True
                        lap_data.loc[current_lap.name, 'DamageConfidence'] = confidence
            
            # === CHECK 3: Persistent Slowdown ===
            # If next 2 laps are also slow, increases confidence
            if i + 2 < len(driver_laps):
                next_lap = driver_laps.iloc[i+1]
                next_next_lap = driver_laps.iloc[i+2]
                
                if (pd.notna(next_lap['LapTime']) and pd.notna(next_next_lap['LapTime'])):
                    
                    avg_next_2 = (next_lap['LapTime'] + next_next_lap['LapTime']) / 2
                    
                    # If next 2 laps still slow, likely real damage
                    if avg_next_2 > prev_avg['LapTime_Rolling'] + 1.0:
                        
                        if lap_data.loc[current_lap.name, 'SuspectedDamage']:
                            # Boost confidence (persistent issue)
                            old_conf = lap_data.loc[current_lap.name, 'DamageConfidence']
                            lap_data.loc[current_lap.name, 'DamageConfidence'] = min(old_conf + 0.3, 1.0)
    
    # Count suspected damage laps
    damage_count = lap_data['SuspectedDamage'].sum()
    
    if damage_count > 0:
        # Group by driver
        drivers_with_damage = lap_data[lap_data['SuspectedDamage']]['Driver'].unique()
        print(f"  ⚠️ Telemetry anomalies detected for {len(drivers_with_damage)} driver(s):")
        
        for driver in drivers_with_damage:
            driver_damage_laps = lap_data[
                (lap_data['Driver'] == driver) & 
                (lap_data['SuspectedDamage'])
            ]
            
            lap_numbers = driver_damage_laps['LapNumber'].tolist()
            avg_confidence = driver_damage_laps['DamageConfidence'].mean()
            
            print(f"    → {driver}: Laps {lap_numbers}, confidence {avg_confidence:.2f}")
    else:
        print("  ✓ No telemetry anomalies detected")
    
    return lap_data

def add_damage_information(lap_data, session):
    """
    Combine damage detection from multiple sources.
    
    Two detection methods:
    1. Race control messages (explicit damage reports)
    2. Telemetry anomalies (implicit evidence)
    
    Final damage classification:
    - CONFIRMED: Both methods agree
    - REPORTED: Only in messages
    - SUSPECTED: Only in telemetry
    - NONE: No damage detected
    
    Returns:
        lap_data with damage columns added
    """
    print("\n[DAMAGE DETECTION]")
    
    # Method 1: Extract from race control messages
    damage_messages = extract_damage_from_messages(session)
    
    # Method 2: Detect from telemetry
    lap_data = detect_damage_from_telemetry(lap_data)
    
    # Initialize final damage columns
    lap_data['HasDamage'] = False
    lap_data['DamageType'] = None
    lap_data['DamageSource'] = None  # 'REPORTED', 'SUSPECTED', 'CONFIRMED'
    
    # Add damage from messages to lap data
    for driver, damage_list in damage_messages.items():
        
        for damage_event in damage_list:
            damage_time = damage_event['time']
            damage_type = damage_event['type']
            
            # Mark all laps AFTER damage occurred
            # (until next pit stop, assuming repair)
            mask = (
                (lap_data['Driver'] == driver) &
                (lap_data['Time'] >= damage_time)
            )
            
            # Get laps affected
            affected_laps = lap_data[mask]
            
            if not affected_laps.empty:
                # Find next pit stop after damage
                next_pit = affected_laps[affected_laps['IsPitLap']]
                
                if not next_pit.empty:
                    # Damage from incident to next pit
                    first_pit_time = next_pit.iloc[0]['Time']
                    
                    mask = (
                        (lap_data['Driver'] == driver) &
                        (lap_data['Time'] >= damage_time) &
                        (lap_data['Time'] < first_pit_time)
                    )
                
                # Mark as damaged
                lap_data.loc[mask, 'HasDamage'] = True
                lap_data.loc[mask, 'DamageType'] = damage_type
                lap_data.loc[mask, 'DamageSource'] = 'REPORTED'
    
    # Cross-reference with telemetry detection
    # If both methods agree, upgrade to CONFIRMED
    confirmed_mask = (
        (lap_data['HasDamage'] == True) &
        (lap_data['SuspectedDamage'] == True)
    )
    lap_data.loc[confirmed_mask, 'DamageSource'] = 'CONFIRMED'
    
    # If only telemetry detected (no message), mark as SUSPECTED
    suspected_only_mask = (
        (lap_data['HasDamage'] == False) &
        (lap_data['SuspectedDamage'] == True) &
        (lap_data['DamageConfidence'] > 0.5)  # Only high confidence
    )
    lap_data.loc[suspected_only_mask, 'HasDamage'] = True
    lap_data.loc[suspected_only_mask, 'DamageType'] = 'SUSPECTED_FROM_TELEMETRY'
    lap_data.loc[suspected_only_mask, 'DamageSource'] = 'SUSPECTED'
    
    # Print final summary
    total_damage_laps = lap_data['HasDamage'].sum()
    
    if total_damage_laps > 0:
        print(f"\n  Final damage summary:")
        print(f"    Total laps with damage: {total_damage_laps}")
        
        for source in ['CONFIRMED', 'REPORTED', 'SUSPECTED']:
            count = (lap_data['DamageSource'] == source).sum()
            if count > 0:
                print(f"    {source}: {count} laps")
    else:
        print(f"\n  Clean race - no damage detected")
    
    return lap_data


def add_qualifying_and_tire_context(lap_data, session):
    """
    Add qualifying performance and tire allocation context.
    
    Critical for strategy analysis:
    
    1. Which qualifying session did driver reach?
       Q1 exit (P16-P20): Most tire sets available
       Q2 exit (P11-P15): Free tire choice + good allocation
       Q3 (P1-P10): Locked into Q2 tires (disadvantage)
    
    2. Starting tire compound
       Top 10: Must use Q2 compound (often Softs)
       P11+: Free choice (usually Mediums for longer stint)
    
    3. Tire sets available
       More sets = More strategic flexibility
       Can afford aggressive strategies
    
    Returns:
        lap_data with qualifying context added
    """
    print("  → Adding qualifying and tire context...")
    
    try:
        # Get session results
        results = session.results
        
        # Initialize columns
        lap_data['QualifyingSession'] = None  # 'Q1', 'Q2', 'Q3'
        lap_data['StartingCompound'] = None   # Compound they started race on
        lap_data['FreeCompoundChoice'] = False  # Did they have free choice?
        
        # For each driver
        for idx, result in results.iterrows():
            driver = result['Abbreviation']
            grid_pos = result['GridPosition']
            
            # Determine which quali session they reached
            if pd.isna(grid_pos):
                quali_session = 'DNQ'  # Did not qualify
            elif grid_pos >= 16:
                quali_session = 'Q1'  # Eliminated in Q1
            elif grid_pos >= 11:
                quali_session = 'Q2'  # Eliminated in Q2
            else:  # grid_pos <= 10
                quali_session = 'Q3'  # Made it to Q3
            
            # Determine if they had free compound choice
            # P11+ can choose, P1-P10 must use Q2 tires
            free_choice = (grid_pos > 10)
            
            # Add to all laps for this driver
            driver_mask = lap_data['Driver'] == driver
            lap_data.loc[driver_mask, 'QualifyingSession'] = quali_session
            lap_data.loc[driver_mask, 'FreeCompoundChoice'] = free_choice
            
            # Get starting compound (from lap 1)
            first_lap = lap_data[
                (lap_data['Driver'] == driver) & 
                (lap_data['LapNumber'] == 1)
            ]
            
            if not first_lap.empty:
                starting_compound = first_lap.iloc[0]['Compound']
                lap_data.loc[driver_mask, 'StartingCompound'] = starting_compound
        
        # Print summary
        print(f"  ✓ Qualifying context added")
        
        # Show starting compounds distribution
        if 'StartingCompound' in lap_data.columns:
            for quali in ['Q1', 'Q2', 'Q3']:
                drivers_in_quali = lap_data[
                    lap_data['QualifyingSession'] == quali
                ]['Driver'].unique()
                
                if len(drivers_in_quali) > 0:
                    print(f"    → {quali}: {len(drivers_in_quali)} drivers")
        
        return lap_data
        
    except Exception as e:
        print(f" Could not add qualifying context: {e}")
        lap_data['QualifyingSession'] = None
        lap_data['StartingCompound'] = None
        lap_data['FreeCompoundChoice'] = False
        return lap_data
    
def add_track_characteristics(lap_data, session):
    
    print("  → Adding track characteristics...")
    
    try:
        # Get circuit name from session
        circuit_name = session.event['EventName']
        
        # Get track info from config file
        track_info = get_track_info(circuit_name)
        
        # Add to all laps
        lap_data['TrackLength_km'] = track_info['length_km']
        lap_data['PitLossTime_sec'] = track_info['pit_loss_sec']
        lap_data['OvertakingDifficulty'] = track_info['overtaking_difficulty']
        lap_data['TypicalPitStops'] = track_info['typical_stops']
        lap_data['TrackType'] = track_info['track_type']
        lap_data['DRS_Zones'] = track_info['drs_zones']
        
        print(f"Track: {circuit_name}")
        print(f"Length: {track_info['length_km']} km")
        print(f" Pit loss: {track_info['pit_loss_sec']}s")
        print(f"Overtaking: {track_info['overtaking_difficulty']}")
        print(f"Typical stops: {track_info['typical_stops']}")
        
        return lap_data
        
    except Exception as e:
        print(f"Could not add track characteristics: {e}")
        # Add NaN columns if error
        lap_data['TrackLength_km'] = np.nan
        lap_data['PitLossTime_sec'] = np.nan
        lap_data['OvertakingDifficulty'] = None
        lap_data['TypicalPitStops'] = np.nan
        lap_data['TrackType'] = None
        lap_data['DRS_Zones'] = np.nan
        return lap_data
    
def get_race_calendar(year):
    
    try:
        # FastF1 provides schedule data
        # get_event_schedule returns DataFrame with all events
        schedule = fastf1.get_event_schedule(year)
        
        # Filter to only include races (not testing, not sprint qualifying)
        # EventFormat can be: 'conventional', 'sprint', 'sprint_shootout', 'testing'
        races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]
        
        # Sort by round number to ensure chronological order
        races = races.sort_values('RoundNumber')
        
        # Return list of (round_number, event_name)
        race_list = [
            (row['RoundNumber'], row['EventName']) 
            for idx, row in races.iterrows()
        ]
        
        return race_list
        
    except Exception as e:
        print(f"    ⚠️  Could not fetch race calendar: {e}")
        return []
    
def find_previous_races(current_race_name, race_calendar):
    
    # Find current race round number
    current_round = None
    
    for round_num, race_name in race_calendar:
        # Match race name (case-insensitive, partial match OK)
        if current_race_name.lower() in race_name.lower() or \
           race_name.lower() in current_race_name.lower():
            current_round = round_num
            break
    
    if current_round is None:
        print(f"    ⚠️  Could not find '{current_race_name}' in calendar")
        return []
    
    # Return all races with round number < current_round
    previous_races = [
        race_name 
        for round_num, race_name in race_calendar 
        if round_num < current_round
    ]
    
    return previous_races

def fetch_race_results(year, race_name):
    
    try:
        # Load the race session
        # Note: We only load race results, not full telemetry (faster)
        session = fastf1.get_session(year, race_name, 'R')
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        
        # Get race results
        results = session.results
        
        if results is None or results.empty:
            return None
        
        return results
        
    except Exception as e:
        # Race might not have happened yet, or data unavailable
        return None
    
def calculate_points_from_results(results):
    
    # F1 points system
    RACE_POINTS = {
        1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
        6: 8, 7: 6, 8: 4, 9: 2, 10: 1
    }
    
    SPRINT_POINTS = {
        1: 8, 2: 7, 3: 6, 4: 5, 5: 4,
        6: 3, 7: 2, 8: 1
    }
    
    driver_points = {}
    team_points = {}
    
    # Check if this is a sprint race
    # Sprint races have different points system
    is_sprint = False  # We'll assume regular race for now
    # (Sprint detection would require checking session type)
    
    points_table = SPRINT_POINTS if is_sprint else RACE_POINTS
    
    # Process each driver's result
    for idx, row in results.iterrows():
        driver = row['Abbreviation']  # 3-letter code (VER, HAM, etc)
        team = row['TeamName']
        position = row['Position']
        status = row['Status']
        
        # Initialize if not exists
        if driver not in driver_points:
            driver_points[driver] = 0
        if team not in team_points:
            team_points[team] = 0
        
        # Only award points if driver finished (not DNF, DSQ, etc)
        # Status == 'Finished' or '+1 Lap' or '+2 Laps' (classified finishers)
        if pd.notna(position):
            # Convert position to integer
            try:
                pos = int(position)
            except:
                continue
            
            # Award points if in points-scoring positions
            if pos in points_table:
                points = points_table[pos]
                driver_points[driver] += points
                team_points[team] += points
        
        # Fastest lap bonus (regular races only, not sprint)
        if not is_sprint:
            # Check if this driver got fastest lap
            # FastestLap column might exist in results
            if 'FastestLap' in row and row['FastestLap'] == True:
                # Only award if driver finished in top 10
                if pd.notna(position) and int(position) <= 10:
                    driver_points[driver] += 1
                    team_points[team] += 1
    
    return {
        'drivers': driver_points,
        'teams': team_points
    }

def add_championship_context(lap_data, session, year, race_name):
    
    print("  → Calculating championship standings before race...")
    
    try:
        # Step 1: Get race calendar for this season
        print("    → Fetching race calendar...")
        race_calendar = get_race_calendar(year)
        
        # Step 2: Find which races happened before this one
        print(f"    → Finding races before {race_name}...")
        previous_races = find_previous_races(race_name, race_calendar)
        
        if len(previous_races) == 0:
            print("    ✓ First race of season - all drivers start with 0 points")
            lap_data['DriverPointsBeforeRace'] = 0
            lap_data['TeamPointsBeforeRace'] = 0
            lap_data['DriverChampionshipPosition'] = np.nan
            lap_data['TeamChampionshipPosition'] = np.nan
            return lap_data
        
        print(f"    → Found {len(previous_races)} previous race(s)")
        
        # Step 3: Calculate points from all previous races
        driver_points = {}  # {driver_code: total_points}
        team_points = {}    # {team_name: total_points}
        
        for prev_race in tqdm(previous_races, desc="    Calculating points", leave=False):
            # Fetch results for this previous race
            race_results = fetch_race_results(year, prev_race)
            
            if race_results is None:
                print(f"    ⚠️  Skipping {prev_race} - no results available")
                continue
            
            # Calculate points earned in this race
            race_points = calculate_points_from_results(race_results)
            
            # Add to cumulative totals
            for driver, points in race_points['drivers'].items():
                if driver not in driver_points:
                    driver_points[driver] = 0
                driver_points[driver] += points
            
            for team, points in race_points['teams'].items():
                if team not in team_points:
                    team_points[team] = 0
                team_points[team] += points
        
        # Step 4: Add championship positions (ranking)
        driver_standings = sorted(driver_points.items(), key=lambda x: x[1], reverse=True)
        team_standings = sorted(team_points.items(), key=lambda x: x[1], reverse=True)
        
        # Create position mappings
        driver_positions = {driver: pos+1 for pos, (driver, pts) in enumerate(driver_standings)}
        team_positions = {team: pos+1 for pos, (team, pts) in enumerate(team_standings)}
        
        # Step 5: Add to lap_data
        # For each lap, look up the driver's points
        lap_data['DriverPointsBeforeRace'] = lap_data['Driver'].map(driver_points).fillna(0)
        lap_data['DriverChampionshipPosition'] = lap_data['Driver'].map(driver_positions).fillna(np.nan)
        
        # For teams, we need to map from driver to team first
        # Get team for each driver from session results
        driver_to_team = {}
        try:
            results = session.results
            for idx, row in results.iterrows():
                driver_to_team[row['Abbreviation']] = row['TeamName']
        except:
            # If session results not available, extract from lap_data
            driver_to_team = dict(zip(lap_data['Driver'], lap_data['Team']))
        
        # Map driver → team → team points
        lap_data['TeamPointsBeforeRace'] = lap_data['Driver'].map(
            lambda d: team_points.get(driver_to_team.get(d, ''), 0)
        )
        lap_data['TeamChampionshipPosition'] = lap_data['Driver'].map(
            lambda d: team_positions.get(driver_to_team.get(d, ''), np.nan)
        )
        
        # Print summary
        print(f"    ✓ Championship standings calculated")
        print(f"      Top 3 drivers:")
        for i, (driver, points) in enumerate(driver_standings[:3], 1):
            print(f"        P{i}: {driver} - {points} points")
        
        print(f"      Top 3 teams:")
        for i, (team, points) in enumerate(team_standings[:3], 1):
            print(f"        P{i}: {team} - {points} points")
        
        return lap_data
        
    except Exception as e:
        print(f"Could not calculate championship context: {e}")
        print(f"Using placeholder values (0 points)")
        lap_data['DriverPointsBeforeRace'] = 0
        lap_data['TeamPointsBeforeRace'] = 0
        lap_data['DriverChampionshipPosition'] = np.nan
        lap_data['TeamChampionshipPosition'] = np.nan
        return lap_data

def add_traffic_data(lap_data, session):
    print("  → Adding traffic data...")

    lap_data['InTraffic'] = False
    lap_data['LappingBackmarker'] = False

    try:
        
        print("    → Method 1: Analyzing lap number differences...")
        
        # For each lap number, find the race leader's lap number
        for lap_num in lap_data['LapNumber'].unique():
            
            # Get all cars on this specific lap number
            cars_on_lap = lap_data[lap_data['LapNumber'] == lap_num]
            
            if cars_on_lap.empty:
                continue
            
            # Find the leader (P1) for this lap
            leader_data = cars_on_lap[cars_on_lap['Position'] == 1]
            
            if leader_data.empty:
                continue
            
            leader_lap_num = leader_data.iloc[0]['LapNumber']
            
            # For each driver on this lap
            for idx, row in cars_on_lap.iterrows():
                driver = row['Driver']
                driver_lap_num = row['LapNumber']
                
                # Check if driver is behind on lap count
                lap_difference = leader_lap_num - driver_lap_num
                
                if lap_difference >= 1:
                    # Driver is AT LEAST 1 lap down
                    # They're being lapped → receiving blue flags
                    lap_data.loc[idx, 'InTraffic'] = True

        print("    → Method 2: Detecting position clusters...")
        
        # For each lap, check if cars are bunched together
        for lap_num in lap_data['LapNumber'].unique():
            
            cars_on_lap = lap_data[lap_data['LapNumber'] == lap_num].copy()
            
            if len(cars_on_lap) < 3:
                continue
            
            # Sort by position
            cars_on_lap = cars_on_lap.sort_values('Position')
            
            # For each car, check gap to car ahead
            for i in range(len(cars_on_lap)):
                current_car = cars_on_lap.iloc[i]
                current_idx = current_car.name
                
                # Get gap to car ahead
                gap_ahead = current_car['GapToAhead']
                
                # If gap is less than 2 seconds, consider it "in traffic"
                # This indicates close racing where overtaking is difficult
                if pd.notna(gap_ahead) and gap_ahead < 2.0:
                    lap_data.loc[current_idx, 'InTraffic'] = True

        print("    → Method 3: Detecting when leaders lap backmarkers...")
        
        # For each lap, check if any leader is on a different lap number
        # than cars they're passing
        
        for lap_num in lap_data['LapNumber'].unique():
            
            cars_on_lap = lap_data[lap_data['LapNumber'] == lap_num]
            
            # Get leaders (top 5 positions)
            leaders = cars_on_lap[cars_on_lap['Position'] <= 5]
            
            for idx, leader in leaders.iterrows():
                leader_lap_num = leader['LapNumber']
                leader_pos = leader['Position']
                
                # Check if there are cars ahead in track position
                # but behind in lap count
                # This means the leader is about to lap them
                
                # Get all cars on lower lap numbers
                backmarkers = lap_data[
                    (lap_data['LapNumber'] < leader_lap_num) &
                    (lap_data['LapNumber'] == lap_num)  # Same timestamp/lap in race
                ]
                
                # If leader is close to backmarkers, they're lapping
                if not backmarkers.empty:
                    lap_data.loc[idx, 'LappingBackmarker'] = True
        
        print("    → Method 4: Checking race control messages...")
        
        try:
            messages = session.race_control_messages
            
            # Look for blue flag messages
            blue_flag_messages = messages[
                messages['Message'].str.contains('BLUE', case=False, na=False)
            ]
            
            if not blue_flag_messages.empty:
                print(f"      Found {len(blue_flag_messages)} blue flag messages")
                
                for idx, msg in blue_flag_messages.iterrows():
                    message_text = msg['Message']
                    message_time = msg['Time']

                    session_start_timestamp = pd.Timestamp(session.event.EventDate)
                    message_time_delta = message_time - session_start_timestamp
                    
                    # Extract driver from message
                    driver = extract_driver_from_message(message_text)
                    
                    if driver:
                        # Mark laps around this time as "in traffic"
                        # Find laps within ±30 seconds of blue flag
                        mask = (
                            (lap_data['Driver'] == driver) &
                            (lap_data['Time'] >= message_time_delta - pd.Timedelta(seconds=30)) &
                            (lap_data['Time'] <= message_time_delta + pd.Timedelta(seconds=30))
                        )
                        
                        lap_data.loc[mask, 'InTraffic'] = True
        
        except Exception as e:
            print(f"      ⚠️  Could not process race control messages: {e}")

        total_traffic_laps = lap_data['InTraffic'].sum()
        total_lapping_laps = lap_data['LappingBackmarker'].sum()
        
        print(f"    ✓ Traffic detection complete")
        print(f"      Laps in traffic: {total_traffic_laps}")
        print(f"      Laps lapping backmarkers: {total_lapping_laps}")
        
        # Show which drivers were most affected
        if total_traffic_laps > 0:
            traffic_by_driver = lap_data[lap_data['InTraffic']].groupby('Driver').size()
            traffic_by_driver = traffic_by_driver.sort_values(ascending=False)
            
            print(f"      Most affected drivers:")
            for driver, count in traffic_by_driver.head(3).items():
                print(f"        {driver}: {count} laps in traffic")
        
        return lap_data
        
    except Exception as e:
        print(f"    ⚠️  Traffic detection failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return with False values
        lap_data['InTraffic'] = False
        lap_data['LappingBackmarker'] = False
        return lap_data
    
def add_overtakes(lap_data):
    print ("  → Adding overtake data...")
    lap_data['OvertakesMade'] = False
    lap_data['OvertakenThisLap'] = False

    try:
        # Process each driver separately
        for driver in lap_data['Driver'].unique():
            
            # Get this driver's laps, sorted by lap number
            driver_laps = lap_data[lap_data['Driver'] == driver].sort_values('LapNumber')
            
            # Skip if driver has fewer than 2 laps
            if len(driver_laps) < 2:
                continue
            
            # For each lap (starting from lap 2)
            for i in range(1, len(driver_laps)):
                
                current_lap = driver_laps.iloc[i]
                previous_lap = driver_laps.iloc[i-1]
                
                current_idx = current_lap.name  # DataFrame index
                
                # Get positions
                current_pos = current_lap['Position']
                previous_pos = previous_lap['Position']
                
                # Skip if either position is missing
                if pd.isna(current_pos) or pd.isna(previous_pos):
                    continue
                
                # Skip if this is a pit lap
                # Position changes during pit stops aren't "real" overtakes
                # (you lose positions because you're in the pits)
                if current_lap['IsPitLap']:
                    continue
                
                # Also skip if PREVIOUS lap was a pit lap
                # (position changes when rejoining track aren't "real" overtakes)
                if previous_lap['IsPitLap']:
                    continue
                
                # Convert to integers for comparison
                try:
                    current_pos = int(current_pos)
                    previous_pos = int(previous_pos)
                except:
                    continue
                
                # Position DECREASED = moved up the order = overtook someone
                # Example: P5 → P4 (gained 1 position)
                if current_pos < previous_pos:
                    lap_data.loc[current_idx, 'OvertakeThisLap'] = True
                
                # Position INCREASED = moved down the order = got overtaken
                # Example: P4 → P5 (lost 1 position)
                elif current_pos > previous_pos:
                    lap_data.loc[current_idx, 'OvertakenThisLap'] = True
                
                # Position SAME = no change
                # (most common case)
                else:
                    pass  # Already False by default
        
        total_overtakes = lap_data['OvertakeThisLap'].sum()
        total_overtaken = lap_data['OvertakenThisLap'].sum()
        
        print(f"    ✓ Overtake detection complete")
        print(f"      Total overtakes: {total_overtakes}")
        print(f"      Total times overtaken: {total_overtaken}")
        
        # Show most aggressive drivers (most overtakes)
        if total_overtakes > 0:
            overtakes_by_driver = lap_data[lap_data['OvertakeThisLap'] == True].groupby('Driver').size()
            overtakes_by_driver = overtakes_by_driver.sort_values(ascending=False)
            
            print(f"      Most aggressive (overtakes):")
            for driver, count in overtakes_by_driver.head(3).items():
                print(f"        {driver}: {count} overtakes")
        
        # Show drivers who lost most positions
        if total_overtaken > 0:
            overtaken_by_driver = lap_data[lap_data['OvertakenThisLap'] == True].groupby('Driver').size()
            overtaken_by_driver = overtaken_by_driver.sort_values(ascending=False)
            
            print(f"      Most defensive (overtaken):")
            for driver, count in overtaken_by_driver.head(3).items():
                print(f"        {driver}: {count} times overtaken")
        
        return lap_data
        
    except Exception as e:
        print(f"    ⚠️  Overtake detection failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Return with False values
        lap_data['OvertakeThisLap'] = False
        lap_data['OvertakenThisLap'] = False
        return lap_data
    
def add_race_result(lap_data, session):

    print("  → Adding race results...")
    
    # Initialize columns
    lap_data['FinalPosition'] = np.nan
    lap_data['FinishStatus'] = None
    lap_data['DidFinish'] = False
    
    try:
        # Get race results (final classification)
        results = session.results
        
        if results is None or results.empty:
            print("    ⚠️  No race results available")
            return lap_data
        
        # Process each driver's result
        for idx, result_row in results.iterrows():
            
            driver = result_row['Abbreviation']  # 3-letter code (VER, HAM, etc.)
            final_pos = result_row['Position']
            status = result_row['Status']
            
            # Find all laps for this driver
            driver_mask = lap_data['Driver'] == driver
            
            # Add final position
            if pd.notna(final_pos):
                try:
                    lap_data.loc[driver_mask, 'FinalPosition'] = int(final_pos)
                except:
                    lap_data.loc[driver_mask, 'FinalPosition'] = np.nan
            
            # Add finish status
            lap_data.loc[driver_mask, 'FinishStatus'] = status
            
            # Determine if driver finished the race
            # "Finished" OR "+1 Lap" OR "+2 Laps" etc. = completed race
            if status and (
                'Finished' in status or 
                '+' in status or  # "+1 Lap", "+2 Laps"
                'Lap' in status
            ):
                lap_data.loc[driver_mask, 'DidFinish'] = True
            else:
                # DNF, Accident, Collision, Disqualified, etc.
                lap_data.loc[driver_mask, 'DidFinish'] = False
        
        # Count finish statuses
        finishers = lap_data[lap_data['DidFinish']]['Driver'].nunique()
        dnfs = lap_data[~lap_data['DidFinish']]['Driver'].nunique()
        
        print(f"    ✓ Race results added")
        print(f"      Finishers: {finishers} drivers")
        print(f"      DNFs: {dnfs} drivers")
        
        # Show podium
        podium = lap_data[lap_data['FinalPosition'].isin([1, 2, 3])][['Driver', 'FinalPosition']].drop_duplicates().sort_values('FinalPosition')
        
        if not podium.empty:
            print(f"      Podium:")
            for idx, row in podium.iterrows():
                pos = int(row['FinalPosition'])
                driver = row['Driver']
                
                if pos == 1:
                    print(f"        🥇 P1: {driver}")
                elif pos == 2:
                    print(f"        🥈 P2: {driver}")
                elif pos == 3:
                    print(f"        🥉 P3: {driver}")
        
        # Show DNFs if any
        if dnfs > 0:
            dnf_drivers = lap_data[~lap_data['DidFinish']][['Driver', 'FinishStatus']].drop_duplicates()
            print(f"      DNFs:")
            for idx, row in dnf_drivers.iterrows():
                print(f"        {row['Driver']}: {row['FinishStatus']}")
        
        return lap_data
        
    except Exception as e:
        print(f"    ⚠️  Could not add race results: {e}")
        import traceback
        traceback.print_exc()
        
        # Return with default values
        lap_data['FinalPosition'] = np.nan
        lap_data['FinishStatus'] = None
        lap_data['DidFinish'] = False
        return lap_data

def fetch_race_data(year, race_name):
    """
    Main function: Fetch complete race data with all 53 features.
    
    This orchestrates ALL the feature extraction functions in the correct order.
    """
    
    print()
    print("=" * 60)
    print(f"Fetching: {year} {race_name} Grand Prix")
    print("=" * 60)
    print()
    
    try:
        # ====================================================================
        # STEP 1: LOAD SESSION
        # ====================================================================
        print("[1/17] Loading session from FastF1...")
        
        import fastf1
        session = fastf1.get_session(year, race_name, 'R')
        session.load()
        
        print(f"  ✓ Session loaded: {session.event['EventName']}")
        print()
        
        
        # ====================================================================
        # STEP 2: EXTRACT BASE LAP DATA
        # ====================================================================
        print("[2/17] Extracting lap data...")
        
        lap_data = get_lap_data(session)
        
        print(f"  ✓ Extracted {len(lap_data)} laps")
        print()
        
        
        # ====================================================================
        # STEP 3: ADD PER-LAP WEATHER
        # ====================================================================
        print("[3/17] Adding per-lap weather...")
        
        lap_data = add_weather_per_lap(lap_data, session)
        
        print(f"  ✓ Weather matched to {len(lap_data)} laps")
        print()
        
        
        # ====================================================================
        # STEP 4: AGGREGATE TELEMETRY
        # ====================================================================
        print("[4/17] Aggregating telemetry per lap...")
        
        telemetry_df = aggregate_telemetry_per_lap(session, lap_data)
        
        # Combine with lap_data
        lap_data = pd.concat([
            lap_data.reset_index(drop=True),
            telemetry_df.reset_index(drop=True)
        ], axis=1)
        
        print(f"  ✓ Telemetry aggregated for {len(lap_data)} laps")
        print()
        
        
        # ====================================================================
        # STEP 5: CALCULATE POSITION GAPS
        # ====================================================================
        print("[5/17] Calculating gaps to cars ahead/behind...")
        
        lap_data = calculate_gaps(lap_data)
        
        print(f"  ✓ Gaps calculated")
        print()
        
        
        # ====================================================================
        # STEP 6: ADD TRACK STATUS FLAGS
        # ====================================================================
        print("[6/17] Adding track status flags...")
        
        lap_data = add_track_status_flag(lap_data)
        
        print(f"  ✓ Track status flags added")
        print()
        
        
        # ====================================================================
        # STEP 7: ADD STARTING POSITIONS
        # ====================================================================
        print("[7/17] Adding starting grid positions...")
        
        lap_data = add_starting_pos(lap_data, session)
        
        print(f"  ✓ Starting positions added")
        print()
        
        
        # ====================================================================
        # STEP 8: ADD PIT STOP INFORMATION
        # ====================================================================
        print("[8/17] Identifying pit stops...")
        
        # First get penalties (needed for pit stop classification)
        penalties = add_penalty_information(lap_data, session)
        
        # Then add pit stops
        lap_data = add_pit_stop_info(lap_data, penalties)
        
        print(f"  ✓ Pit stops identified")
        print()
        
        
        # ====================================================================
        # STEP 9: ADD PENALTY INFORMATION (already done above, just mark complete)
        # ====================================================================
        print("[9/17] Penalty information added")
        print()
        
        
        # ====================================================================
        # STEP 10: ADD DAMAGE DETECTION
        # ====================================================================
        print("[10/17] Detecting damage...")
        
        lap_data = add_damage_information(lap_data, session)
        
        print(f"  ✓ Damage detection complete")
        print()
        
        
        # ====================================================================
        # STEP 11: ADD QUALIFYING CONTEXT
        # ====================================================================
        print("[11/17] Adding qualifying context...")
        
        lap_data = add_qualifying_and_tire_context(lap_data, session)
        
        print(f"  ✓ Qualifying context added")
        print()
        
        
        # ====================================================================
        # STEP 12: ADD TRACK CHARACTERISTICS
        # ====================================================================
        print("[12/17] Adding track characteristics...")
        
        lap_data = add_track_characteristics(lap_data, session)
        
        print(f"  ✓ Track characteristics added")
        print()
        
        
        # ====================================================================
        # STEP 13: ADD CHAMPIONSHIP CONTEXT
        # ====================================================================
        print("[13/17] Adding championship context...")
        
        lap_data = add_championship_context(lap_data, session, year, race_name)
        
        print(f"  ✓ Championship context added")
        print()
        
        
        # ====================================================================
        # STEP 14: ADD TRAFFIC DETECTION
        # ====================================================================
        print("[14/17] Detecting traffic situations...")
        
        lap_data = add_traffic_data(lap_data, session)
        
        print(f"  ✓ Traffic detection complete")
        print()
        
        
        # ====================================================================
        # STEP 15: ADD OVERTAKES
        # ====================================================================
        print("[15/17] Detecting overtakes...")
        
        lap_data = add_overtakes(lap_data)
        
        print(f"  ✓ Overtake detection complete")
        print()
        
        
        # ====================================================================
        # STEP 16: ADD RACE RESULTS
        # ====================================================================
        print("[16/17] Adding race results...")
        
        lap_data = add_race_result(lap_data, session)
        
        print(f"  ✓ Race results added")
        print()
        
        
        # ====================================================================
        # STEP 17: ADD METADATA
        # ====================================================================
        print("[17/17] Adding metadata...")
        
        lap_data['Year'] = year
        lap_data['RaceName'] = session.event['EventName']
        lap_data['EventDate'] = session.event['EventDate']
        
        print(f"  ✓ Metadata added")
        print()
        
        
        # ====================================================================
        # FINAL SUMMARY
        # ====================================================================
        print("=" * 60)
        print(f"✓ SUCCESS: Fetched {len(lap_data)} laps")
        print(f"  Features: {len(lap_data.columns)} columns")
        print("=" * 60)
        print()
        
        return lap_data
        
    except Exception as e:
        print()
        print("=" * 60)
        print(f"✗ FATAL ERROR: {e}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return None


def save_race_data(lap_data, year, race_name, output_dir='data/raw'):
    """
    Save race data to CSV file.
    
    **FILE NAMING CONVENTION:**
    {year}_{race_name}.csv
    
    Example: 2024_Monaco.csv
    
    **FILE LOCATION:**
    data/raw/
    
    Why 'raw'? Because this is unprocessed race data.
    Later phases will create 'processed' data (normalized, filtered, etc.)
    
    Args:
        lap_data: DataFrame to save
        year: Season year
        race_name: Race name (sanitized for filename)
        output_dir: Directory to save to (default: data/raw)
    
    Returns:
        Path to saved file, or None if error
    """
    
    try:
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Sanitize race name for filename
        # Remove spaces, special characters
        safe_race_name = race_name.replace(' ', '_').replace('Grand_Prix', '').strip('_')
        
        # Create filename
        filename = f"{year}_{safe_race_name}.csv"
        filepath = output_path / filename
        
        # Save to CSV
        lap_data.to_csv(filepath, index=False)
        
        # Get file size
        file_size_kb = filepath.stat().st_size / 1024
        
        print()
        print("💾 Data saved successfully!")
        print(f"   File: {filepath}")
        print(f"   Size: {file_size_kb:.1f} KB")
        print()
        
        return filepath
        
    except Exception as e:
        print()
        print(f"✗ Could not save data: {e}")
        import traceback
        traceback.print_exc()
        return None


def print_data_summary(lap_data):
    """
    Print summary statistics about the fetched data.
    
    Shows:
    - Total laps
    - Number of drivers
    - Tire compounds used
    - All feature columns
    """
    
    print("=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    print()
    
    # Basic stats
    print(f"Total laps: {len(lap_data)}")
    print(f"Drivers: {lap_data['Driver'].nunique()}")
    
    # Tire compounds
    if 'Compound' in lap_data.columns:
        compounds = lap_data['Compound'].unique()
        compounds = [c for c in compounds if pd.notna(c)]
        print(f"Tire compounds: {compounds}")
    
    # Feature count
    print(f"Features: {len(lap_data.columns)} columns")
    print()
    
    # List all features
    print("All features:")
    for i, col in enumerate(lap_data.columns, 1):
        print(f"  {i:2d}. {col}")
    
    print()
    print("=" * 60)
    print()


def main():
    """
    Command-line interface for fetching race data.
    
    Usage:
        python fetch_race_data.py --year 2024 --race "Monaco"
        python fetch_race_data.py --year 2023 --race "Silverstone"
        python fetch_race_data.py --year 2024 --race "Singapore" --output data/raw
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Fetch F1 race data with comprehensive feature engineering',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fetch_race_data.py --year 2024 --race "Monaco"
  python fetch_race_data.py --year 2024 --race "Silverstone"
  python fetch_race_data.py --year 2023 --race "Spa"
  
Output:
  Saves CSV file to data/raw/{year}_{race}.csv
        """
    )
    
    parser.add_argument(
        '--year',
        type=int,
        required=True,
        help='Season year (e.g., 2024)'
    )
    
    parser.add_argument(
        '--race',
        type=str,
        required=True,
        help='Race name (e.g., "Monaco", "Silverstone", "Singapore")'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save to CSV (just fetch and display)'
    )
    
    args = parser.parse_args()
    
    # Setup FastF1 cache
    setup_fastf1_cache()
    
    # Fetch race data
    lap_data = fetch_race_data(args.year, args.race)
    
    if lap_data is None:
        print("Failed to fetch race data.")
        return 1
    
    # Print summary
    print_data_summary(lap_data)
    
    # Save to CSV (unless --no-save flag)
    if not args.no_save:
        saved_path = save_race_data(lap_data, args.year, args.race, args.output)
        
        if saved_path is None:
            print("Failed to save data.")
            return 1
    
    print("✓ All done!")
    return 0


if __name__ == "__main__":
    exit(main())