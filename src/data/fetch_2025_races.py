"""
Simple 2025 Race Fetcher
========================

Fetches all available 2025 F1 races.

Usage:
    python fetch_2025_races.py
"""

import time
from pathlib import Path
import fastf1

# Import from your main script
from fetch_race_data import setup_fastf1_cache, fetch_race_data, save_race_data


def get_2025_races():
    """Get all 2025 races from calendar."""
    try:
        schedule = fastf1.get_event_schedule(2025)
        
        # Filter to only races (not testing)
        races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]
        
        # Get race names
        race_names = races['EventName'].tolist()
        
        print(f"Found {len(race_names)} races in 2025 calendar")
        print()
        
        return race_names
        
    except Exception as e:
        print(f"❌ Could not get 2025 schedule: {e}")
        return []


def main():
    """Fetch all 2025 races."""
    
    print("="*70)
    print("2025 F1 RACE FETCHER")
    print("="*70)
    print()
    
    # Setup FastF1 cache
    setup_fastf1_cache()
    
    # Get 2025 races
    race_names = get_2025_races()
    
    if not race_names:
        print("No races found!")
        return
    
    # Output directory
    output_dir = Path('data/raw')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check which races already exist
    races_to_fetch = []
    for race_name in race_names:
        safe_name = race_name.replace(' ', '_').replace('Grand_Prix', '').strip('_')
        filename = f"2025_{safe_name}.csv"
        filepath = output_dir / filename
        
        if filepath.exists():
            print(f"⏭️  Skipping {race_name} (already exists)")
        else:
            races_to_fetch.append(race_name)
    
    print()
    print(f"Races to fetch: {len(races_to_fetch)}")
    print()
    
    if len(races_to_fetch) == 0:
        print("✓ All 2025 races already fetched!")
        return
    
    # Track results
    successful = []
    failed = []
    
    # Fetch each race
    start_time = time.time()
    
    for i, race_name in enumerate(races_to_fetch, 1):
        print("="*70)
        print(f"[{i}/{len(races_to_fetch)}] Fetching: 2025 {race_name}")
        print("="*70)
        
        try:
            # Fetch race data
            lap_data = fetch_race_data(2025, race_name)
            
            if lap_data is not None:
                # Save to CSV
                saved_path = save_race_data(lap_data, 2025, race_name, str(output_dir))
                
                if saved_path:
                    successful.append(race_name)
                    print(f"✅ SUCCESS: {race_name}")
                else:
                    failed.append((race_name, "Save failed"))
                    print(f"❌ FAILED: {race_name} (could not save)")
            else:
                failed.append((race_name, "Fetch failed - data not available yet"))
                print(f"⚠️  SKIPPED: {race_name} (data not available yet)")
        
        except Exception as e:
            failed.append((race_name, str(e)))
            print(f"❌ FAILED: {race_name}")
            print(f"   Error: {e}")
        
        # Show progress
        elapsed = time.time() - start_time
        avg_time = elapsed / i
        remaining = avg_time * (len(races_to_fetch) - i)
        
        print()
        print(f"⏱️  Progress: {i}/{len(races_to_fetch)} races")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Estimated remaining: {remaining/60:.1f} minutes")
        print()
    
    # Final summary
    print()
    print("="*70)
    print("2025 FETCH COMPLETE")
    print("="*70)
    print()
    print(f"✅ Successful: {len(successful)} races")
    print(f"⚠️  Failed/Unavailable: {len(failed)} races")
    
    total_time = time.time() - start_time
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print()
    
    # List successful races
    if successful:
        print("Successfully fetched:")
        for race_name in successful:
            print(f"  ✓ 2025 {race_name}")
        print()
    
    # List failed/unavailable races
    if failed:
        print("Failed or not available yet:")
        for race_name, error in failed:
            print(f"  ⚠️  2025 {race_name}")
            if "not available" not in error:
                print(f"      Reason: {error}")
        print()
        print("💡 These races will be available after they're run in 2025!")
    
    print()
    print(f"📁 All data saved to: {output_dir}")
    print()


if __name__ == "__main__":
    main()
