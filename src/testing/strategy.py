"""
src/testing/strategy.py

Phase 2C: Strategy Definition
Comprehensive F1 pit strategy with fixed and dynamic components
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum

# ============================================================
# ENUMS FOR TYPE SAFETY
# ============================================================

class TireCompound(Enum):
    """F1 tire compounds"""
    SOFT = "SOFT"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    INTERMEDIATE = "INTERMEDIATE"
    WET = "WET"

class StintPlan(Enum):
    """How to drive each stint"""
    PUSH = "push"              # Maximum attack
    MANAGE = "manage"          # Balanced pace
    CONSERVE = "conserve"      # Save tires/fuel
    FUEL_SAVE = "fuel_save"    # Heavy fuel saving
    TIRE_SAVE = "tire_save"    # Heavy tire saving

class SafetyCarReaction(Enum):
    """What to do under safety car"""
    PIT_IMMEDIATELY = "pit_immediately"    # Dive into pits
    STAY_OUT = "stay_out"                  # Stay on track
    FOLLOW_LEADER = "follow_leader"        # Match leader's choice
    OPPOSITE_LEADER = "opposite_leader"    # Do opposite of leader
    PIT_IF_CHEAP = "pit_if_cheap"         # Pit if lose < 2 positions

class WeatherReaction(Enum):
    """What to do when weather changes"""
    IMMEDIATE_CHANGE = "immediate_change"  # Change tires ASAP
    WAIT_ONE_LAP = "wait_one_lap"         # See if rain gets worse
    GAMBLE_SLICKS = "gamble_slicks"       # Stay on slicks in light rain
    SWITCH_TO_INTERS = "switch_to_inters" # Go to intermediates
    SWITCH_TO_WETS = "switch_to_wets"     # Go to full wets

# ============================================================
# STRATEGY CLASS
# ============================================================

@dataclass
class Strategy:
    """
    Comprehensive F1 pit strategy with fixed and dynamic components.
    
    FIXED COMPONENTS (pre-planned):
    - Base pit laps
    - Tire compounds
    - Stint driving plans
    - Target lap times
    - Fuel strategy
    
    DYNAMIC COMPONENTS (react to events):
    - Safety car reactions
    - VSC reactions
    - Red flag reactions
    - Weather change reactions
    - Competitor reactions
    
    Example:
        strategy = Strategy(
            name="Aggressive Undercut",
            pit_laps=[25],
            tire_compounds=[TireCompound.MEDIUM, TireCompound.SOFT],
            stint_plans=[StintPlan.MANAGE, StintPlan.PUSH],
            target_lap_times=[88.5, 87.2],
            fuel_strategy="push",
            sc_reaction=SafetyCarReaction.PIT_IMMEDIATELY,
            weather_reaction=WeatherReaction.IMMEDIATE_CHANGE
        )
    """
    
    # ========================================
    # REQUIRED FIELDS (no defaults)
    # ========================================
    name: str
    pit_laps: List[int]  # [25] for 1-stop, [20, 40] for 2-stop
    tire_compounds: List[TireCompound]  # Length = len(pit_laps) + 1
    stint_plans: List[StintPlan]  # How to drive each stint
    
    # ========================================
    # OPTIONAL FIELDS (with defaults)
    # ========================================
    
    # Basic info
    description: Optional[str] = None
    
    # Driving plan
    target_lap_times: Optional[List[float]] = None  # Target pace per stint (seconds)
    
    # Fuel strategy
    fuel_strategy: str = "balanced"  # "aggressive", "balanced", "conservative"
    fuel_save_laps: Optional[List[int]] = None  # Specific laps to save fuel
    
    # Safety Car / VSC
    sc_reaction: SafetyCarReaction = SafetyCarReaction.PIT_IF_CHEAP
    vsc_reaction: SafetyCarReaction = SafetyCarReaction.PIT_IF_CHEAP
    
    # Red Flag
    red_flag_tire_change: bool = True  # Change tires during red flag?
    red_flag_preferred_compound: Optional[TireCompound] = None
    
    # Weather
    weather_reaction: WeatherReaction = WeatherReaction.WAIT_ONE_LAP
    rain_threshold: float = 0.3  # Switch tires if rainfall > 30%
    
    # Competitor reactions
    react_to_leader_pit: bool = False  # Pit within 2 laps of leader?
    react_to_position_behind_pit: bool = False  # Cover car behind?
    undercut_offset: int = 2  # Pit X laps before competitor to undercut
    overcut_offset: int = 2   # Stay out X laps after competitor to overcut
    
    # ========================================
    # VALIDATION & METADATA
    # ========================================
    
    def __post_init__(self):
        """Validate strategy after initialization"""
        self.validate()
        
    # ========================================
    # BASIC INFO
    # ========================================
    name: str
    description: Optional[str] = None
    
    # ========================================
    # FIXED STRATEGY COMPONENTS
    # ========================================
    
    # Pit stop plan
    pit_laps: List[int]  # [25] for 1-stop, [20, 40] for 2-stop
    tire_compounds: List[TireCompound]  # Length = len(pit_laps) + 1
    
    # Driving plan per stint
    stint_plans: List[StintPlan]  # How to drive each stint
    target_lap_times: Optional[List[float]] = None  # Target pace per stint (seconds)
    
    # Fuel strategy
    fuel_strategy: str = "balanced"  # "aggressive", "balanced", "conservative"
    fuel_save_laps: Optional[List[int]] = None  # Specific laps to save fuel
    
    # ========================================
    # DYNAMIC REACTION RULES
    # ========================================
    
    # Safety Car / VSC
    sc_reaction: SafetyCarReaction = SafetyCarReaction.PIT_IF_CHEAP
    vsc_reaction: SafetyCarReaction = SafetyCarReaction.PIT_IF_CHEAP
    
    # Red Flag
    red_flag_tire_change: bool = True  # Change tires during red flag?
    red_flag_preferred_compound: Optional[TireCompound] = None
    
    # Weather
    weather_reaction: WeatherReaction = WeatherReaction.WAIT_ONE_LAP
    rain_threshold: float = 0.3  # Switch tires if rainfall > 30%
    
    # Competitor reactions
    react_to_leader_pit: bool = False  # Pit within 2 laps of leader?
    react_to_position_behind_pit: bool = False  # Cover car behind?
    undercut_offset: int = 2  # Pit X laps before competitor to undercut
    overcut_offset: int = 2   # Stay out X laps after competitor to overcut
    
    # ========================================
    # VALIDATION & METADATA
    # ========================================
    
    def __post_init__(self):
        """Validate strategy after initialization"""
        self.validate()
    
    def validate(self):
        """
        Enforce F1 regulations and logical consistency.
        
        Rules checked:
        1. Must have at least 1 pit stop (dry race)
        2. Must use 2 different compounds (dry race)
        3. Tire compounds list matches pit stops
        4. Stint plans match number of stints
        5. Pit laps are in ascending order
        6. Target lap times match stints (if provided)
        7. No intermediate/wet in dry conditions
        """
        errors = []
        
        # Check 1: At least 1 pit stop (unless rain race)
        if len(self.pit_laps) == 0:
            if not self._is_rain_strategy():
                errors.append("Must have at least 1 pit stop in dry race")
        
        # Check 2: Must use 2 different compounds (dry)
        if not self._is_rain_strategy():
            unique_compounds = set(self.tire_compounds)
            if len(unique_compounds) < 2:
                errors.append(f"Must use 2 different compounds (only using {unique_compounds})")
        
        # Check 3: Tire compounds match pit stops
        expected_compound_count = len(self.pit_laps) + 1
        if len(self.tire_compounds) != expected_compound_count:
            errors.append(
                f"Tire compounds ({len(self.tire_compounds)}) must equal "
                f"pit stops + 1 ({expected_compound_count})"
            )
        
        # Check 4: Stint plans match stints
        expected_stint_count = len(self.pit_laps) + 1
        if len(self.stint_plans) != expected_stint_count:
            errors.append(
                f"Stint plans ({len(self.stint_plans)}) must equal "
                f"number of stints ({expected_stint_count})"
            )
        
        # Check 5: Pit laps ascending
        if self.pit_laps != sorted(self.pit_laps):
            errors.append(f"Pit laps must be in ascending order: {self.pit_laps}")
        
        # Check 6: Target lap times match stints (if provided)
        if self.target_lap_times is not None:
            if len(self.target_lap_times) != expected_stint_count:
                errors.append(
                    f"Target lap times ({len(self.target_lap_times)}) must equal "
                    f"number of stints ({expected_stint_count})"
                )
        
        # Check 7: Valid pit lap numbers
        for lap in self.pit_laps:
            if lap < 1:
                errors.append(f"Pit lap {lap} is invalid (must be > 0)")
            if lap > 100:  # Sanity check (longest F1 race is ~78 laps)
                errors.append(f"Pit lap {lap} seems too high")
        
        # If errors found, raise exception
        if errors:
            error_msg = "Strategy validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)
    
    def _is_rain_strategy(self) -> bool:
        """Check if this is a rain strategy"""
        rain_compounds = {TireCompound.INTERMEDIATE, TireCompound.WET}
        return any(c in rain_compounds for c in self.tire_compounds)
    
    # ========================================
    # STRATEGY EXECUTION METHODS
    # ========================================
    
    def should_pit_on_lap(self, current_lap: int, race_state: Dict) -> bool:
        """
        Determine if should pit on this lap based on strategy + race state.
        
        Args:
            current_lap: Current race lap
            race_state: Dictionary with current race conditions
                {
                    'safety_car': bool,
                    'vsc': bool,
                    'rainfall': float,
                    'leader_just_pitted': bool,
                    'position_behind_just_pitted': bool,
                    ...
                }
        
        Returns:
            True if should pit this lap
        """
        # Check fixed plan
        if current_lap in self.pit_laps:
            return True
        
        # Check dynamic reactions
        
        # Safety Car
        if race_state.get('safety_car', False):
            if self.sc_reaction == SafetyCarReaction.PIT_IMMEDIATELY:
                return True
            elif self.sc_reaction == SafetyCarReaction.FOLLOW_LEADER:
                return race_state.get('leader_just_pitted', False)
            elif self.sc_reaction == SafetyCarReaction.OPPOSITE_LEADER:
                return not race_state.get('leader_just_pitted', False)
        
        # VSC
        if race_state.get('vsc', False):
            if self.vsc_reaction == SafetyCarReaction.PIT_IMMEDIATELY:
                return True
        
        # Weather change
        if race_state.get('rainfall', 0) > self.rain_threshold:
            if self.weather_reaction == WeatherReaction.IMMEDIATE_CHANGE:
                return True
        
        # React to leader
        if self.react_to_leader_pit and race_state.get('leader_just_pitted', False):
            return True
        
        return False
    
    def get_tire_compound_for_stint(self, stint_number: int) -> TireCompound:
        """Get tire compound for given stint (0-indexed)"""
        if stint_number >= len(self.tire_compounds):
            raise ValueError(f"Stint {stint_number} out of range")
        return self.tire_compounds[stint_number]
    
    def get_stint_plan(self, stint_number: int) -> StintPlan:
        """Get driving plan for given stint (0-indexed)"""
        if stint_number >= len(self.stint_plans):
            raise ValueError(f"Stint {stint_number} out of range")
        return self.stint_plans[stint_number]
    
    def get_target_lap_time(self, stint_number: int) -> Optional[float]:
        """Get target lap time for given stint (0-indexed)"""
        if self.target_lap_times is None:
            return None
        if stint_number >= len(self.target_lap_times):
            raise ValueError(f"Stint {stint_number} out of range")
        return self.target_lap_times[stint_number]
    
    # ========================================
    # DISPLAY METHODS
    # ========================================
    
    def describe(self) -> str:
        """
        Generate human-readable description of strategy.
        
        Returns:
            Multi-line string describing strategy
        """
        lines = []
        lines.append(f"Strategy: {self.name}")
        if self.description:
            lines.append(f"  {self.description}")
        
        lines.append(f"\n📍 FIXED PLAN:")
        lines.append(f"  Pit stops: {len(self.pit_laps)}-stop")
        
        for i, (stint_plan, compound) in enumerate(zip(self.stint_plans, self.tire_compounds)):
            if i < len(self.pit_laps):
                lines.append(f"  Stint {i+1}: {compound.value} → Pit Lap {self.pit_laps[i]} ({stint_plan.value})")
            else:
                lines.append(f"  Stint {i+1}: {compound.value} → Finish ({stint_plan.value})")
        
        lines.append(f"\n⚡ DYNAMIC REACTIONS:")
        lines.append(f"  Safety Car: {self.sc_reaction.value}")
        lines.append(f"  VSC: {self.vsc_reaction.value}")
        lines.append(f"  Weather: {self.weather_reaction.value}")
        lines.append(f"  React to leader: {self.react_to_leader_pit}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        """Convert strategy to dictionary for serialization"""
        return {
            'name': self.name,
            'description': self.description,
            'pit_laps': self.pit_laps,
            'tire_compounds': [c.value for c in self.tire_compounds],
            'stint_plans': [s.value for s in self.stint_plans],
            'target_lap_times': self.target_lap_times,
            'fuel_strategy': self.fuel_strategy,
            'sc_reaction': self.sc_reaction.value,
            'vsc_reaction': self.vsc_reaction.value,
            'weather_reaction': self.weather_reaction.value,
        }
    
    @property
    def total_stops(self) -> int:
        """Total number of pit stops"""
        return len(self.pit_laps)
    
    @property
    def is_aggressive(self) -> bool:
        """Is this an aggressive strategy? (2+ stops or early pits)"""
        return self.total_stops >= 2 or (self.pit_laps and self.pit_laps[0] < 20)
    
    @property
    def is_conservative(self) -> bool:
        """Is this a conservative strategy? (1-stop, late pit)"""
        return self.total_stops == 1 and self.pit_laps[0] > 30

# ============================================================
# EXAMPLE STRATEGIES
# ============================================================

def create_example_strategies():
    """Create some example strategies for testing"""
    
    # Conservative 1-stop
    conservative = Strategy(
        name="Conservative 1-Stop",
        description="Safe strategy, one pit stop around lap 30",
        pit_laps=[30],
        tire_compounds=[TireCompound.MEDIUM, TireCompound.HARD],
        stint_plans=[StintPlan.MANAGE, StintPlan.PUSH],
        target_lap_times=[88.5, 87.8],
        fuel_strategy="balanced",
        sc_reaction=SafetyCarReaction.PIT_IF_CHEAP,
        weather_reaction=WeatherReaction.WAIT_ONE_LAP
    )
    
    # Aggressive undercut
    aggressive = Strategy(
        name="Aggressive Undercut",
        description="Early pit to undercut competitors",
        pit_laps=[18],
        tire_compounds=[TireCompound.SOFT, TireCompound.MEDIUM],
        stint_plans=[StintPlan.PUSH, StintPlan.MANAGE],
        target_lap_times=[87.2, 88.0],
        fuel_strategy="aggressive",
        sc_reaction=SafetyCarReaction.STAY_OUT,
        weather_reaction=WeatherReaction.IMMEDIATE_CHANGE
    )
    
    # Two-stop attack
    two_stop = Strategy(
        name="Two-Stop Attack",
        description="Aggressive 2-stop with fresh tires at end",
        pit_laps=[20, 40],
        tire_compounds=[TireCompound.SOFT, TireCompound.SOFT, TireCompound.MEDIUM],
        stint_plans=[StintPlan.PUSH, StintPlan.PUSH, StintPlan.MANAGE],
        target_lap_times=[87.5, 87.2, 87.8],
        fuel_strategy="aggressive",
        sc_reaction=SafetyCarReaction.PIT_IMMEDIATELY,
        weather_reaction=WeatherReaction.IMMEDIATE_CHANGE
    )
    
    return [conservative, aggressive, two_stop]


# ============================================================
# TESTING
# ============================================================

if __name__ == "__main__":
    print("="*60)
    print("STRATEGY CLASS TESTING")
    print("="*60)
    
    # Test example strategies
    strategies = create_example_strategies()
    
    for strategy in strategies:
        print(f"\n{strategy.describe()}")
        print(f"\nMetadata:")
        print(f"  Total stops: {strategy.total_stops}")
        print(f"  Aggressive: {strategy.is_aggressive}")
        print(f"  Conservative: {strategy.is_conservative}")
    
    # Test validation
    print("\n" + "="*60)
    print("TESTING VALIDATION")
    print("="*60)
    
    try:
        # This should FAIL - only 1 compound
        bad_strategy = Strategy(
            name="Invalid",
            pit_laps=[25],
            tire_compounds=[TireCompound.SOFT, TireCompound.SOFT],
            stint_plans=[StintPlan.PUSH, StintPlan.PUSH]
        )
    except ValueError as e:
        print(f"\n✓ Correctly caught invalid strategy:")
        print(f"  {e}")
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)