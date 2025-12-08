import argparse
import time
from pathlib import Path
import pandas as pd
from datetime import datetime

# Import from your main script
# (Assumes this file is in same directory as fetch_race_data.py)
from fetch_race_data import setup_fastf1_cache, fetch_race_data, save_race_data


def get_season_races(year, only_completed=True):
    import fastf1
    from datetime import datetime, timezone
    
    try:
        schedule = fastf1.get_event_schedule(year)
        
        # Filter to only races (not testing)
        races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint'])]
        
        # If only_completed, filter by date
        if only_completed:
            now = datetime.now(timezone.utc)
            
            # Filter races where EventDate has passed
            # Add a 1-day buffer to ensure race is complete
            races = races[races['EventDate'] < now - pd.Timedelta(days=1)]
            
            print(f"   (Filtering to completed races only)")
        
        # Get race names
        race_names = races['EventName'].tolist()
        
        return race_names
        
    except Exception as e:
        print(f"Could not get schedule for {year}: {e}")
        return []


def batch_fetch_races(years=None, race_names=None, output_dir='data/raw', skip_existing=True):
    
    print()
    print("=" * 70)
    print("F1 BATCH RACE FETCHER")
    print("=" * 70)
    print()
    
    # Build list of (year, race_name) tuples to fetch
    races_to_fetch = []
    
    if race_names and len(years) == 1:
        # Specific races from one year
        for race_name in race_names:
            races_to_fetch.append((years[0], race_name))
    
    else:
        # All races from specified years
        for year in years:
            print(f"📅 Getting race schedule for {year}...")
            
            # For current/future years, only get completed races
            from datetime import datetime
            current_year = datetime.now().year
            only_completed = (year >= current_year)
            
            season_races = get_season_races(year, only_completed=only_completed)
            
            if season_races:
                print(f"   ✓ Found {len(season_races)} races")
                for race_name in season_races:
                    races_to_fetch.append((year, race_name))
            else:
                print(f"   ⚠️  No races found for {year}")
    
    print()
    print(f"Total races to fetch: {len(races_to_fetch)}")
    print()
    
    # Filter out existing files if skip_existing=True
    if skip_existing:
        original_count = len(races_to_fetch)
        output_path = Path(output_dir)
        
        races_to_fetch_filtered = []
        for year, race_name in races_to_fetch:
            # Check if file exists
            safe_race_name = race_name.replace(' ', '_').replace('Grand_Prix', '').strip('_')
            filename = f"{year}_{safe_race_name}.csv"
            filepath = output_path / filename
            
            if filepath.exists():
                print(f"⏭️  Skipping {year} {race_name} (already exists)")
            else:
                races_to_fetch_filtered.append((year, race_name))
        
        races_to_fetch = races_to_fetch_filtered
        skipped_count = original_count - len(races_to_fetch)
        
        if skipped_count > 0:
            print()
            print(f"Skipped {skipped_count} existing files")
            print(f"Remaining to fetch: {len(races_to_fetch)}")
            print()
    
    if len(races_to_fetch) == 0:
        print("✓ All races already fetched!")
        return {'success': 0, 'failed': 0, 'skipped': original_count if skip_existing else 0}
    
    # Track results
    successful = []
    failed = []
    
    # Fetch each race
    start_time = time.time()
    
    for i, (year, race_name) in enumerate(races_to_fetch, 1):
        
        print("=" * 70)
        print(f"[{i}/{len(races_to_fetch)}] Fetching: {year} {race_name}")
        print("=" * 70)
        
        try:
            # Fetch race data
            lap_data = fetch_race_data(year, race_name)
            
            if lap_data is not None:
                # Save to CSV
                saved_path = save_race_data(lap_data, year, race_name, output_dir)
                
                if saved_path:
                    successful.append((year, race_name))
                    print(f"✅ SUCCESS: {year} {race_name}")
                else:
                    failed.append((year, race_name, "Save failed"))
                    print(f"❌ FAILED: {year} {race_name} (could not save)")
            
            else:
                failed.append((year, race_name, "Fetch failed"))
                print(f"❌ FAILED: {year} {race_name} (could not fetch)")
        
        except Exception as e:
            failed.append((year, race_name, str(e)))
            print(f"❌ FAILED: {year} {race_name}")
            print(f"   Error: {e}")
        
        # Estimate time remaining
        elapsed = time.time() - start_time
        avg_time_per_race = elapsed / i
        remaining_races = len(races_to_fetch) - i
        estimated_remaining = avg_time_per_race * remaining_races
        
        print()
        print(f"⏱️  Progress: {i}/{len(races_to_fetch)} races")
        print(f"   Elapsed: {elapsed/60:.1f} minutes")
        print(f"   Estimated remaining: {estimated_remaining/60:.1f} minutes")
        print()
    
    # Final summary
    print()
    print("=" * 70)
    print("BATCH FETCH COMPLETE")
    print("=" * 70)
    print()
    print(f"✅ Successful: {len(successful)} races")
    print(f"❌ Failed: {len(failed)} races")
    
    total_time = time.time() - start_time
    print(f"⏱️  Total time: {total_time/60:.1f} minutes")
    print()
    
    # List successful races
    if successful:
        print("Successfully fetched:")
        for year, race_name in successful:
            print(f"  ✓ {year} {race_name}")
        print()
    
    # List failed races
    if failed:
        print("Failed races:")
        for year, race_name, error in failed:
            print(f"  ✗ {year} {race_name}")
            print(f"    Reason: {error}")
        print()
        
        # Save failed races to file for retry
        failed_file = Path(output_dir) / 'failed_races.txt'
        with open(failed_file, 'w') as f:
            f.write("# Failed races - retry with:\n")
            f.write("# python batch_fetch_races.py --retry\n\n")
            for year, race_name, error in failed:
                f.write(f"{year},{race_name},{error}\n")
        
        print(f"📝 Failed races saved to: {failed_file}")
        print()
    
    return {
        'success': len(successful),
        'failed': len(failed),
        'successful_races': successful,
        'failed_races': failed
    }


def main():
    """
    Command-line interface for batch fetching.
    """
    
    parser = argparse.ArgumentParser(
        description='Batch fetch F1 race data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fetch all of 2024
  python batch_fetch_races.py --year 2024
  
  # Fetch multiple years
  python batch_fetch_races.py --years 2023 2024
  
  # Fetch all available (2022-2025, only completed races)
  python batch_fetch_races.py --all
  
  # Fetch specific races from 2024
  python batch_fetch_races.py --year 2024 --races "Monaco" "Silverstone"
  
  # Fetch only completed 2025 races
  python batch_fetch_races.py --year 2025
  
  # Don't skip existing files (re-fetch everything)
  python batch_fetch_races.py --year 2024 --no-skip
        """
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Single year to fetch (e.g., 2024)'
    )
    
    parser.add_argument(
        '--years',
        type=int,
        nargs='+',
        help='Multiple years to fetch (e.g., 2022 2023 2024)'
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Fetch all available years (2022-2025, only completed races)'
    )
    
    parser.add_argument(
        '--races',
        type=str,
        nargs='+',
        help='Specific races to fetch (requires --year)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/raw',
        help='Output directory (default: data/raw)'
    )
    
    parser.add_argument(
        '--no-skip',
        action='store_true',
        help='Re-fetch races even if CSV exists'
    )
    
    args = parser.parse_args()
    
    # Determine which years to fetch
    years = []
    
    if args.all:
        years = [2022, 2023, 2024, 2025]
    elif args.years:
        years = args.years
    elif args.year:
        years = [args.year]
    else:
        print("❌ Error: Must specify --year, --years, or --all")
        parser.print_help()
        return 1
    
    print(f"Years to fetch: {years}")
    
    # Setup FastF1 cache
    setup_fastf1_cache()
    
    # Batch fetch
    results = batch_fetch_races(
        years=years,
        race_names=args.races,
        output_dir=args.output,
        skip_existing=not args.no_skip
    )
    
    # Return error code if any failures
    if results['failed'] > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
