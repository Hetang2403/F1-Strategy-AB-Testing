"""
Flask API for F1 Strategy A/B Testing
Exposes our simulator as REST API
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.testing.strategy_simulator import StrategySimulator
from src.testing.strategy import Strategy, TireCompound, StintPlan
from src.testing.race_state import RaceState, Competitor

app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize simulator (load model once)
print("🏁 Loading F1 Strategy Simulator...")
simulator = StrategySimulator()
print("✅ Simulator ready!")

@app.route('/')
def home():
    """Health check"""
    return jsonify({
        'status': 'online',
        'service': 'F1 Strategy A/B Testing API',
        'version': '1.0.0'
    })

@app.route('/api/simulate', methods=['POST'])
def simulate():
    """
    Simulate two strategies and compare.
    
    Request body:
    {
        "race_state": {
            "current_lap": 25,
            "total_laps": 52,
            "driver": "VER",
            "position": 2,
            "tire_age": 25,
            "tire_compound": "MEDIUM",
            "gap_ahead": 3.2,
            "gap_behind": 5.8,
            "track_name": "Silverstone",
            "track_temp": 45.0,
            "air_temp": 22.0,
            "competitors": [...]
        },
        "strategy_a": {
            "name": "Conservative 1-Stop",
            "pit_laps": [30],
            "tire_compounds": ["MEDIUM", "HARD"],
            "stint_plans": ["MANAGE", "PUSH"]
        },
        "strategy_b": {
            "name": "Aggressive 2-Stop",
            "pit_laps": [28, 42],
            "tire_compounds": ["MEDIUM", "SOFT", "SOFT"],
            "stint_plans": ["MANAGE", "PUSH", "PUSH"]
        }
    }
    """
    try:
        data = request.json
        
        # Parse race state
        race_state_data = data['race_state']
        race_state = RaceState(
            current_lap=race_state_data['current_lap'],
            total_laps=race_state_data['total_laps'],
            driver=race_state_data['driver'],
            position=race_state_data['position'],
            tire_age=race_state_data['tire_age'],
            tire_compound=TireCompound[race_state_data['tire_compound']],
            gap_ahead=race_state_data['gap_ahead'],
            gap_behind=race_state_data['gap_behind'],
            track_name=race_state_data['track_name'],
            track_temp=race_state_data.get('track_temp', 35.0),
            air_temp=race_state_data.get('air_temp', 20.0),
            competitors=_parse_competitors(race_state_data.get('competitors', []))
        )
        
        # Parse strategies
        strategy_a = _parse_strategy(data['strategy_a'])
        strategy_b = _parse_strategy(data['strategy_b'])
        
        # Simulate both
        result_a = simulator.simulate(race_state, strategy_a, verbose=False)
        result_b = simulator.simulate(race_state, strategy_b, verbose=False)
        
        # Compare
        position_diff = result_a['predicted_position'] - result_b['predicted_position']
        time_diff = result_a['predicted_time'] - result_b['predicted_time']
        
        # Determine winner
        if position_diff < 0 or (position_diff == 0 and time_diff < 0):
            winner = 'strategy_a'
            winner_name = strategy_a.name
        elif position_diff > 0 or (position_diff == 0 and time_diff > 0):
            winner = 'strategy_b'
            winner_name = strategy_b.name
        else:
            winner = 'tie'
            winner_name = 'Both strategies similar'
        
        return jsonify({
            'success': True,
            'result_a': result_a,
            'result_b': result_b,
            'comparison': {
                'winner': winner,
                'winner_name': winner_name,
                'position_diff': position_diff,
                'time_diff': time_diff
            }
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/tracks', methods=['GET'])
def get_tracks():
    """Get list of available tracks"""
    tracks = [
        'Silverstone', 'Monaco', 'Spa', 'Monza', 'Bahrain',
        'Singapore', 'Zandvoort', 'Austria', 'Hungary', 'Belgium'
    ]
    return jsonify({'tracks': tracks})

@app.route('/api/compounds', methods=['GET'])
def get_compounds():
    """Get tire compounds"""
    return jsonify({
        'compounds': ['SOFT', 'MEDIUM', 'HARD', 'INTERMEDIATE', 'WET']
    })

@app.route('/api/stint-plans', methods=['GET'])
def get_stint_plans():
    """Get stint plan options"""
    return jsonify({
        'stint_plans': ['PUSH', 'MANAGE', 'CONSERVE']
    })

def _parse_competitors(competitors_data):
    """Parse competitors from JSON"""
    if not competitors_data:
        return None
    
    return [
        Competitor(
            driver_code=c['driver_code'],
            position=c['position'],
            tire_age=c['tire_age'],
            tire_compound=TireCompound[c['tire_compound']],
            gap_to_us=c['gap_to_us']
        )
        for c in competitors_data
    ]

def _parse_strategy(strategy_data):
    """Parse strategy from JSON"""
    return Strategy(
        name=strategy_data['name'],
        pit_laps=strategy_data['pit_laps'],
        tire_compounds=[TireCompound[c] for c in strategy_data['tire_compounds']],
        stint_plans=[StintPlan[p] for p in strategy_data['stint_plans']],
        description=strategy_data.get('description', '')
    )

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)