"""
src/testing/race_simulator.py

Race Simulation Engine
Core lap-by-lap race simulation with all competitors
"""

from typing import List
import numpy as np

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.testing.driver_state import DriverState
from src.testing.race_state import RaceState, Competitor
from src.testing.strategy import Strategy, TireCompound
from src.testing.lap_time_predictor import LapTimePredictor

class RaceSimulator:
    """
    Core race simulation engine.
    
    Simulates all 20 drivers lap-by-lap with:
    - Lap time prediction
    - Pit stops
    - Position tracking
    - Overtakes
    """
    
    # Pit loss times by track
    PIT_LOSSES = {
        'Monaco': 16.0, 'Silverstone': 22.0, 'Spa': 24.0,
        'Monza': 20.0, 'Bahrain': 19.0, 'Singapore': 23.0,
        'Zandvoort': 18.0
    }
    
    def __init__(self):
        """Initialize with lap time predictor"""
        self.predictor = LapTimePredictor()
    
    def simulate_race(self, race_state: RaceState, strategy: Strategy) -> List[DriverState]:
        """
        Simulate race from current state to finish.
        
        Args:
            race_state: Current race conditions
            strategy: Strategy for our driver
            
        Returns:
            List of final driver states sorted by position
        """
        # Initialize all drivers
        drivers = self._initialize_drivers(race_state, strategy)
        
        # Get pit loss time
        pit_loss = self._get_pit_loss_time(race_state.track_name)
        
        # Simulate lap-by-lap
        for lap in range(race_state.current_lap, race_state.total_laps + 1):
            self._simulate_lap(drivers, lap, race_state, pit_loss)
            self._update_positions(drivers)
        
        return drivers
    
    def _simulate_lap(self, drivers: List[DriverState], lap: int,
                     race_state: RaceState, pit_loss: float):
        """Simulate one lap for all drivers"""
        for driver in drivers:
            if driver.should_pit(lap):
                # Pit stop
                lap_time = self.predictor.predict(driver, lap, race_state) + pit_loss
                new_compound = driver.choose_next_compound()
                driver.pit(new_compound)
                driver.complete_lap(lap_time)
            else:
                # Normal lap
                lap_time = self.predictor.predict(driver, lap, race_state)
                driver.complete_lap(lap_time)
    
    def _update_positions(self, drivers: List[DriverState]):
        """Update positions based on cumulative time"""
        drivers.sort(key=lambda d: d.cumulative_time)
        for pos, driver in enumerate(drivers, 1):
            driver.position = pos
    
    def _initialize_drivers(self, state: RaceState, 
                           strategy: Strategy) -> List[DriverState]:
        """Initialize all 20 drivers"""
        drivers = []
        
        # Create our driver
        our_driver = DriverState(
            driver_code=state.driver,
            position=state.position,
            tire_age=state.tire_age,
            tire_compound=state.tire_compound,
            strategy=strategy,
            is_our_driver=True
        )
        
        # Add competitors
        if state.competitors:
            for comp in state.competitors:
                drivers.append(DriverState(
                    driver_code=comp.driver_code,
                    position=comp.position,
                    tire_age=comp.tire_age,
                    tire_compound=comp.tire_compound
                ))
        else:
            # Generate reasonable competitors
            for pos in range(1, 21):
                if pos == state.position:
                    continue
                tire_age = np.random.randint(5, 30)
                compounds = [TireCompound.SOFT, TireCompound.MEDIUM, TireCompound.HARD]
                drivers.append(DriverState(
                    driver_code=f"P{pos}",
                    position=pos,
                    tire_age=tire_age,
                    tire_compound=np.random.choice(compounds)
                ))
        
        drivers.append(our_driver)
        drivers.sort(key=lambda d: d.position)
        
        # Set initial cumulative times based on gaps
        self._initialize_gaps(drivers, state)
        
        return drivers
    
    def _initialize_gaps(self, drivers: List[DriverState], state: RaceState):
        """Set initial cumulative times based on position gaps"""
        for i, driver in enumerate(drivers):
            if driver.position == 1:
                driver.cumulative_time = 0.0
            elif driver.driver_code == state.driver:
                driver.cumulative_time = state.gap_ahead * (state.position - 1)
            else:
                driver.cumulative_time = driver.position * 2.0
    
    def _get_pit_loss_time(self, track_name: str) -> float:
        """Get pit loss time for track"""
        for track, loss in self.PIT_LOSSES.items():
            if track in track_name:
                return loss
        return 21.0