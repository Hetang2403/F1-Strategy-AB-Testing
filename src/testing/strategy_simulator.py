"""
src/testing/strategy_simulator.py

Strategy Simulator - Main Interface
Clean, simple interface to race simulation
"""

from typing import Dict
import numpy as np

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.testing.strategy import Strategy
from src.testing.race_state import RaceState
from src.testing.race_simulator import RaceSimulator

class StrategySimulator:
    """
    High-level interface for strategy simulation.
    
    Usage:
        simulator = StrategySimulator()
        result = simulator.simulate(race_state, strategy)
    """
    
    def __init__(self):
        """Initialize simulator"""
        self.race_sim = RaceSimulator()
        print("✓ Strategy Simulator loaded")
    
    def simulate(self, race_state: RaceState, strategy: Strategy,
                 verbose: bool = True) -> Dict:
        """
        Simulate strategy execution.
        
        Args:
            race_state: Current race conditions
            strategy: Strategy to execute
            verbose: Print progress
            
        Returns:
            Dictionary with detailed results
        """
        if verbose:
            print(f"\n🏁 Simulating: {strategy.name}")
            print(f"   From Lap {race_state.current_lap} to {race_state.total_laps}")
        
        # Run simulation
        final_drivers = self.race_sim.simulate_race(race_state, strategy)
        
        # Find our driver
        our_driver = next(d for d in final_drivers if d.is_our_driver)
        
        # Build results
        result = {
            'strategy_name': strategy.name,
            'predicted_position': our_driver.position,
            'predicted_time': our_driver.cumulative_time,
            'total_pits': len(strategy.pit_laps),
            'avg_lap_time': our_driver.cumulative_time / (race_state.total_laps - race_state.current_lap + 1),
            'pit_laps': strategy.pit_laps,
            'final_tire_age': our_driver.tire_age,
            'final_tire_compound': our_driver.tire_compound.value
        }
        
        if verbose:
            print(f"   ✓ Complete! P{result['predicted_position']}, {result['predicted_time']:.1f}s")
        
        return result


# ============================================================
# TEST WITH DETAILED INPUT/OUTPUT
# ============================================================

if __name__ == "__main__":
    from src.testing.strategy import TireCompound, StintPlan
    from src.testing.race_state import Competitor
    
    print("="*70)
    print(" "*20 + "STRATEGY SIMULATOR TEST")
    print("="*70)
    
    simulator = StrategySimulator()
    
    # Test race state
    race_state = RaceState(
        current_lap=25,
        total_laps=52,
        driver='VER',
        position=2,
        tire_age=25,
        tire_compound=TireCompound.MEDIUM,
        gap_ahead=3.2,
        gap_behind=5.8,
        track_name='Silverstone',
        track_temp=45.0,
        air_temp=22.0,
        competitors=[
            Competitor('HAM', 1, 20, TireCompound.MEDIUM, -3.2),
            Competitor('LEC', 3, 18, TireCompound.SOFT, +5.8),
            Competitor('NOR', 4, 22, TireCompound.MEDIUM, +8.5),
            Competitor('SAI', 5, 19, TireCompound.SOFT, +11.2)
        ]
    )
    
    # Display input state
    print("\n" + "="*70)
    print(" "*25 + "📍 RACE STATE INPUT")
    print("="*70)
    print(race_state.describe())
    
    if race_state.competitors:
        print(f"\n  Known Competitors:")
        for comp in race_state.competitors:
            print(f"    P{comp.position}: {comp.driver_code} - "
                  f"{comp.tire_compound.value} ({comp.tire_age} laps), "
                  f"Gap: {comp.gap_to_us:+.1f}s")
    
    # Test strategies
    strategy_1 = Strategy(
        name="Conservative 1-Stop",
        pit_laps=[30],
        tire_compounds=[TireCompound.MEDIUM, TireCompound.HARD],
        stint_plans=[StintPlan.MANAGE, StintPlan.PUSH],
        description="Pit once on lap 30, switch to hards for the end"
    )
    
    strategy_2 = Strategy(
        name="Aggressive 2-Stop",
        pit_laps=[28, 42],
        tire_compounds=[TireCompound.MEDIUM, TireCompound.SOFT, TireCompound.SOFT],
        stint_plans=[StintPlan.MANAGE, StintPlan.PUSH, StintPlan.PUSH],
        description="Two stops with softs for pace advantage"
    )
    
    # Display strategies
    print("\n" + "="*70)
    print(" "*23 + "🎯 STRATEGIES TO TEST")
    print("="*70)
    
    print("\n📋 STRATEGY 1:")
    print(strategy_1.describe())
    
    print("\n📋 STRATEGY 2:")
    print(strategy_2.describe())
    
    # Simulate both
    print("\n" + "="*70)
    print(" "*23 + "🏁 RUNNING SIMULATIONS")
    print("="*70)
    
    result_1 = simulator.simulate(race_state, strategy_1)
    result_2 = simulator.simulate(race_state, strategy_2)
    
    # Detailed comparison
    print("\n" + "="*70)
    print(" "*25 + "📊 RESULTS COMPARISON")
    print("="*70)
    
    print(f"\n{'Metric':<30} {'Strategy 1':<20} {'Strategy 2':<20}")
    print("-"*70)
    print(f"{'Strategy Name':<30} {result_1['strategy_name']:<20} {result_2['strategy_name']:<20}")
    print(f"{'Final Position':<30} P{result_1['predicted_position']:<19} P{result_2['predicted_position']:<19}")
    print(f"{'Total Race Time':<30} {result_1['predicted_time']:.1f}s{'':<15} {result_2['predicted_time']:.1f}s")
    print(f"{'Average Lap Time':<30} {result_1['avg_lap_time']:.3f}s{'':<13} {result_2['avg_lap_time']:.3f}s")
    print(f"{'Number of Pit Stops':<30} {result_1['total_pits']:<20} {result_2['total_pits']:<20}")
    print(f"{'Pit Laps':<30} {str(result_1['pit_laps']):<20} {str(result_2['pit_laps']):<20}")
    print(f"{'Final Tire':<30} {result_1['final_tire_compound']:<20} {result_2['final_tire_compound']:<20}")
    print(f"{'Final Tire Age':<30} {result_1['final_tire_age']:<20} {result_2['final_tire_age']:<20}")
    
    # Analysis
    print("\n" + "="*70)
    print(" "*27 + "📈 ANALYSIS")
    print("="*70)
    
    position_diff = result_1['predicted_position'] - result_2['predicted_position']
    time_diff = result_1['predicted_time'] - result_2['predicted_time']
    
    print(f"\n  Position Comparison:")
    if position_diff < 0:
        print(f"    ✅ Strategy 1 is {abs(position_diff)} position(s) BETTER")
    elif position_diff > 0:
        print(f"    ✅ Strategy 2 is {position_diff} position(s) BETTER")
    else:
        print(f"    ⚖️  Same final position")
    
    print(f"\n  Time Comparison:")
    if time_diff < 0:
        print(f"    ✅ Strategy 1 is {abs(time_diff):.1f}s FASTER")
    elif time_diff > 0:
        print(f"    ✅ Strategy 2 is {abs(time_diff):.1f}s FASTER")
    else:
        print(f"    ⚖️  Same race time")
    
    # Recommendation
    print("\n" + "="*70)
    print(" "*28 + "🏆 RECOMMENDATION")
    print("="*70)
    
    if position_diff < 0 or (position_diff == 0 and time_diff < 0):
        winner = "Strategy 1: Conservative 1-Stop"
        print(f"\n  ✅ {winner}")
        print(f"\n  Why it wins:")
        if position_diff < 0:
            print(f"    • Better position (P{result_1['predicted_position']} vs P{result_2['predicted_position']})")
        if time_diff < 0:
            print(f"    • Faster by {abs(time_diff):.1f}s")
        print(f"    • Fewer pit stops = less time lost")
        print(f"    • Gap behind ({race_state.gap_behind:.1f}s) covers pit loss")
    elif position_diff > 0 or (position_diff == 0 and time_diff > 0):
        winner = "Strategy 2: Aggressive 2-Stop"
        print(f"\n  ✅ {winner}")
        print(f"\n  Why it wins:")
        if position_diff > 0:
            print(f"    • Better position (P{result_2['predicted_position']} vs P{result_1['predicted_position']})")
        if time_diff > 0:
            print(f"    • Faster by {abs(time_diff):.1f}s")
        print(f"    • Fresh tires provide pace advantage")
    else:
        print(f"\n  ⚖️  Both strategies perform similarly")
    
    print("\n" + "="*70)
    print(" "*25 + "✅ TEST COMPLETE!")
    print("="*70)