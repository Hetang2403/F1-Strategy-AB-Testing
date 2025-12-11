"""
src/testing/lap_time_predictor.py

Lap Time Predictor Wrapper
Clean interface to trained XGBoost model
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict

import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.testing.driver_state import DriverState
from src.testing.strategy import TireCompound
from src.testing.race_state import RaceState

class LapTimePredictor:
    """
    Wrapper for trained lap time prediction model.
    
    Provides clean interface to predict lap times.
    """
    
    def __init__(self):
        """Load trained model and supporting data"""
        models_dir = project_root / 'data' / 'models'
        
        with open(models_dir / 'laptime_predictor.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        with open(models_dir / 'laptime_features.pkl', 'rb') as f:
            self.features = pickle.load(f)
        
        with open(models_dir / 'laptime_encoders.pkl', 'rb') as f:
            self.encoders = pickle.load(f)
        
        with open(models_dir / 'track_baselines.pkl', 'rb') as f:
            self.track_baselines = pickle.load(f)
    
    def predict(self, driver: DriverState, current_lap: int, 
                race_state: RaceState) -> float:
        """
        Predict lap time for driver in current conditions.
        
        Args:
            driver: Driver state
            current_lap: Current lap number
            race_state: Race conditions
            
        Returns:
            Predicted lap time in seconds
        """
        # Get track baseline
        baseline = self.track_baselines.get(race_state.track_name, 90.0)
        
        # Build features
        features = self._build_features(driver, current_lap, race_state)
        
        # Predict delta
        delta = self.model.predict(features)[0]
        
        # Calculate lap time
        lap_time = baseline + delta
        
        # Add small random variation for realism
        lap_time += np.random.uniform(-0.2, 0.2)
        
        return max(lap_time, 60.0)  # Sanity check
    
    def _build_features(self, driver: DriverState, current_lap: int,
                       race_state: RaceState) -> np.ndarray:
        """Build feature vector for prediction"""
        feature_dict = {
            'TyreLife': driver.tire_age,
            'Compound': self._encode_compound(driver.tire_compound),
            'FreshTyre': int(driver.tire_age == 0),
            'LapNumber': current_lap,
            'Position': driver.position,
            'GapToAhead': 2.0,  # Approximate
            'TrackTemp': race_state.track_temp,
            'AirTemp': race_state.air_temp,
            'OvertakingDifficulty': self._encode_difficulty(race_state.track_name)
        }
        
        # Build array in correct order
        feature_array = [feature_dict.get(f, 0) for f in self.features]
        return np.array(feature_array).reshape(1, -1)
    
    def _encode_compound(self, compound: TireCompound) -> int:
        """Encode tire compound to integer"""
        encoder = self.encoders.get('compound')
        if encoder is None:
            return 0
        try:
            return encoder.transform([compound.value])[0]
        except:
            return 0
    
    def _encode_difficulty(self, track_name: str) -> int:
        """Encode overtaking difficulty"""
        difficulties = {
            'Monaco': 3, 'Hungary': 3, 'Singapore': 3,
            'Zandvoort': 2, 'Silverstone': 1, 'Spa': 0, 'Monza': 0
        }
        for track, diff in difficulties.items():
            if track in track_name:
                return diff
        return 1