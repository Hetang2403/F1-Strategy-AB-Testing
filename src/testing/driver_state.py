"""
src/testing/driver_state.py

Driver State Management
Models individual driver state during race simulation
"""

from dataclasses import dataclass
from typing import Optional
from copy import deepcopy
import numpy as np

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.testing.strategy import Strategy, TireCompound

@dataclass
class DriverState:
    """
    State of a single driver during race simulation.
    
    Tracks:
    - Position and timing
    - Tire state
    - Strategy (if known)
    """
    driver_code: str
    position: int
    tire_age: int
    tire_compound: TireCompound
    cumulative_time: float = 0.0
    current_lap_time: float = 0.0
    strategy: Optional[Strategy] = None
    is_our_driver: bool = False
    
    def copy(self):
        """Create deep copy of driver state"""
        return deepcopy(self)
    
    def should_pit(self, current_lap: int) -> bool:
        """
        Decide if driver should pit this lap.
        
        For our driver: Check strategy
        For competitors: Simple tire age logic
        """
        if self.is_our_driver and self.strategy:
            return current_lap in self.strategy.pit_laps
        
        # Competitor logic: pit based on tire age
        return self._should_competitor_pit()
    
    def _should_competitor_pit(self) -> bool:
        """
        Competitor pit decision based on tire age.
        
        Pit windows:
        - SOFT: 18-22 laps
        - MEDIUM: 25-30 laps  
        - HARD: 35-40 laps
        """
        pit_windows = {
            TireCompound.SOFT: (18, 22),
            TireCompound.MEDIUM: (25, 30),
            TireCompound.HARD: (35, 40)
        }
        
        min_age, max_age = pit_windows.get(self.tire_compound, (25, 30))
        return self.tire_age >= np.random.randint(min_age, max_age + 1)
    
    def choose_next_compound(self) -> TireCompound:
        """
        Choose next tire compound after pit.
        
        For our driver: Use strategy
        For competitors: Opposite of current
        """
        if self.is_our_driver and self.strategy:
            pit_count = len([lap for lap in self.strategy.pit_laps])
            if pit_count > 0:
                return self.strategy.tire_compounds[pit_count]
        
        # Competitor logic: cycle through compounds
        if self.tire_compound == TireCompound.SOFT:
            return TireCompound.MEDIUM
        elif self.tire_compound == TireCompound.MEDIUM:
            return TireCompound.HARD
        else:
            return TireCompound.MEDIUM
    
    def pit(self, new_compound: TireCompound):
        """Execute pit stop - change tires"""
        self.tire_compound = new_compound
        self.tire_age = 0
    
    def complete_lap(self, lap_time: float):
        """Update state after completing a lap"""
        self.cumulative_time += lap_time
        self.current_lap_time = lap_time
        self.tire_age += 1