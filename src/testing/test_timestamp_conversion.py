"""
Simple Test - Does Timestamp Conversion Work?
=============================================

Run this to test if the conversion is working correctly.
"""

import fastf1
import pandas as pd

# Load Bahrain 2024
print("Loading Bahrain 2024...")
session = fastf1.get_session(2024, 'Bahrain', 'R')
session.load()

print("\n" + "="*70)
print("TESTING TIMESTAMP TYPES")
print("="*70)

# Test 1: What is lap_data['Time']?
laps = session.laps
print(f"\n1. session.laps['Time'] dtype: {laps['Time'].dtype}")
print(f"   First value: {laps['Time'].iloc[0]}")
print(f"   Type: {type(laps['Time'].iloc[0])}")

# Test 2: What is race_control_messages['Time']?
messages = session.race_control_messages
print(f"\n2. messages['Time'] dtype: {messages['Time'].dtype}")
print(f"   First value: {messages['Time'].iloc[0]}")
print(f"   Type: {type(messages['Time'].iloc[0])}")

# Test 3: What is session_start_time?
print(f"\n3. session.session_start_time: {session.session_start_time}")
print(f"   Type: {type(session.session_start_time)}")

# Test 4: Can we convert?
print(f"\n4. Testing conversion:")
message_time = messages['Time'].iloc[0]
session_start = session.session_start_time

print(f"   message_time: {message_time} ({type(message_time).__name__})")
print(f"   session_start: {session_start} ({type(session_start).__name__})")

try:
    converted = message_time - session_start
    print(f"   ✓ Conversion works!")
    print(f"   Result: {converted} ({type(converted).__name__})")
except Exception as e:
    print(f"   ✗ Conversion FAILED: {e}")

# Test 5: Can we compare?
print(f"\n5. Testing comparison:")
lap_time = laps['Time'].iloc[0]

try:
    result = lap_time >= converted
    print(f"   ✓ Comparison works!")
    print(f"   {lap_time} >= {converted} = {result}")
except Exception as e:
    print(f"   ✗ Comparison FAILED: {e}")

print("\n" + "="*70)
