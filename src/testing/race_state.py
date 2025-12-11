"""
src/testing/race_state.py

Race State Definition
Represents current race conditions at a specific moment in time
"""

from dataclasses import dataclass
from typing import Optional, List, Dict
from enum import Enum

# Import our Strategy types
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.testing.strategy import TireCompound

class WeatherCondition(Enum):
    """Weather conditions"""
    DRY = "dry"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"
    CHANGING = "changing"

@dataclass
class Competitor:
    """
    Represents another driver in the race.
    Used to model traffic, undercuts, reactions, etc.
    """
    # Required fields (no defaults) - MUST come first
    driver_code: str  # 'VER', 'HAM', etc.
    position: int
    tire_age: int
    tire_compound: TireCompound
    gap_to_us: float  # Gap in seconds (negative if behind, positive if ahead)
    
    # Optional fields (with defaults) - MUST come after required fields
    pit_laps_completed: int = 0
    expected_next_pit: Optional[int] = None

@dataclass
class RaceState:
    """
    Complete snapshot of race conditions at a specific moment.
    
    This is ALL the information we need to simulate strategies.
    
    Example:
        state = RaceState(
            current_lap=25,
            total_laps=52,
            driver='VER',
            position=2,
            tire_age=25,
            tire_compound=TireCompound.MEDIUM,
            gap_ahead=3.2,
            gap_behind=5.8,
            track_name='Silverstone'
        )
    """
    
    # ========================================
    # REQUIRED FIELDS (no defaults) - MUST come first
    # ========================================
    
    # Race progression
    current_lap: int
    total_laps: int
    
    # Our driver state
    driver: str
    position: int
    tire_age: int
    tire_compound: TireCompound
    gap_ahead: float
    gap_behind: float
    
    # Track
    track_name: str
    
    # ========================================
    # OPTIONAL FIELDS (with defaults) - MUST come after required
    # ========================================
    
    # Tire state
    fresh_tire: bool = False
    
    # Conditions
    weather: WeatherCondition = WeatherCondition.DRY
    track_temp: float = 35.0
    air_temp: float = 20.0
    
    # Track status
    safety_car: bool = False
    vsc: bool = False
    red_flag: bool = False
    
    # Competitors
    competitors: Optional[List[Competitor]] = None
    
    # ========================================
    # COMPUTED PROPERTIES
    # ========================================
    
    @property
    def laps_remaining(self) -> int:
        """How many laps left in race"""
        return self.total_laps - self.current_lap
    
    @property
    def race_progress(self) -> float:
        """Race completion percentage (0.0 to 1.0)"""
        return self.current_lap / self.total_laps
    
    @property
    def is_leading(self) -> bool:
        """Are we in P1?"""
        return self.position == 1
    
    @property
    def estimated_fuel_load(self) -> float:
        """
        Estimate remaining fuel based on laps remaining.
        
        F1 cars start with ~110kg fuel, burn ~1.6kg per lap
        This is approximate but good enough for simulation
        """
        fuel_per_lap = 1.6  # kg
        fuel_remaining = self.laps_remaining * fuel_per_lap
        return max(0, fuel_remaining)
    
    @property
    def is_rain_race(self) -> bool:
        """Is it raining?"""
        return self.weather in [WeatherCondition.LIGHT_RAIN, 
                                WeatherCondition.HEAVY_RAIN]
    
    # ========================================
    # HELPER METHODS
    # ========================================
    
    def get_competitor_by_position(self, position: int) -> Optional[Competitor]:
        """Get competitor in specific position"""
        if self.competitors is None:
            return None
        
        for comp in self.competitors:
            if comp.position == position:
                return comp
        return None
    
    def get_car_ahead(self) -> Optional[Competitor]:
        """Get car directly ahead of us"""
        if self.position == 1:
            return None
        return self.get_competitor_by_position(self.position - 1)
    
    def get_car_behind(self) -> Optional[Competitor]:
        """Get car directly behind us"""
        return self.get_competitor_by_position(self.position + 1)
    
    def describe(self) -> str:
        """Human-readable description of race state"""
        lines = []
        lines.append(f"Race State - Lap {self.current_lap}/{self.total_laps}")
        lines.append(f"  Driver: {self.driver}")
        lines.append(f"  Position: P{self.position}")
        lines.append(f"  Tire: {self.tire_compound.value} ({self.tire_age} laps old)")
        lines.append(f"  Gap ahead: {self.gap_ahead:+.1f}s")
        lines.append(f"  Gap behind: {self.gap_behind:+.1f}s")
        lines.append(f"  Track: {self.track_name}")
        lines.append(f"  Weather: {self.weather.value}")
        lines.append(f"  Track temp: {self.track_temp}°C")
        
        if self.safety_car:
            lines.append(f"  ⚠️  SAFETY CAR")
        if self.vsc:
            lines.append(f"  ⚠️  VSC")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for logging/saving"""
        return {
            'current_lap': self.current_lap,
            'total_laps': self.total_laps,
            'driver': self.driver,
            'position': self.position,
            'tire_age': self.tire_age,
            'tire_compound': self.tire_compound.value,
            'gap_ahead': self.gap_ahead,
            'gap_behind': self.gap_behind,
            'track_name': self.track_name,
            'weather': self.weather.value,
            'track_temp': self.track_temp,
            'air_temp': self.air_temp,
            'safety_car': self.safety_car,
            'vsc': self.vsc
        }
    
    def copy(self):
        """Create a copy of this state for simulation"""
        from copy import deepcopy
        return deepcopy(self)


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("RACE STATE EXAMPLES")
    print("="*60)
    
    # Example 1: Simple race state
    state1 = RaceState(
        current_lap=25,
        total_laps=52,
        driver='VER',
        position=2,
        tire_age=25,
        tire_compound=TireCompound.MEDIUM,
        gap_ahead=3.2,
        gap_behind=5.8,
        track_name='Silverstone',
        weather=WeatherCondition.DRY,
        track_temp=45.0,
        air_temp=22.0
    )
    
    print("\nExample 1: Mid-race state")
    print("-"*60)
    print(state1.describe())
    print(f"\nComputed properties:")
    print(f"  Laps remaining: {state1.laps_remaining}")
    print(f"  Race progress: {state1.race_progress*100:.1f}%")
    print(f"  Fuel load: ~{state1.estimated_fuel_load:.1f}kg")
    print(f"  Is leading: {state1.is_leading}")
    
    # Example 2: Race with competitors
    state2 = RaceState(
        current_lap=30,
        total_laps=52,
        driver='HAM',
        position=3,
        tire_age=18,
        tire_compound=TireCompound.SOFT,
        gap_ahead=2.5,
        gap_behind=8.2,
        track_name='Monaco',
        weather=WeatherCondition.DRY,
        track_temp=42.0,
        air_temp=25.0,
        competitors=[
            Competitor('VER', 1, 20, TireCompound.MEDIUM, +15.3),
            Competitor('LEC', 2, 19, TireCompound.SOFT, +2.5),
            Competitor('NOR', 4, 22, TireCompound.MEDIUM, -8.2)
        ]
    )
    
    print("\n\nExample 2: Race with competitors")
    print("-"*60)
    print(state2.describe())
    
    car_ahead = state2.get_car_ahead()
    if car_ahead:
        print(f"\nCar ahead: {car_ahead.driver_code} (P{car_ahead.position})")
        print(f"  Tire: {car_ahead.tire_compound.value} ({car_ahead.tire_age} laps)")
        print(f"  Gap: {car_ahead.gap_to_us:.1f}s")
    
    # Example 3: Safety car scenario
    state3 = RaceState(
        current_lap=40,
        total_laps=52,
        driver='PER',
        position=4,
        tire_age=32,
        tire_compound=TireCompound.HARD,
        gap_ahead=1.2,
        gap_behind=1.5,
        track_name='Spa',
        weather=WeatherCondition.DRY,
        track_temp=38.0,
        air_temp=18.0,
        safety_car=True
    )
    
    print("\n\nExample 3: Safety car scenario")
    print("-"*60)
    print(state3.describe())
    
    print("\n" + "="*60)
    print("✅ RACE STATE EXAMPLES COMPLETE!")
    print("="*60)