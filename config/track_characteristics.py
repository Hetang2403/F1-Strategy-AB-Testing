"""
track_characteristics.py

Track-specific data for F1 strategy analysis.

Data sources:
- FIA official track maps
- Historical race data
- F1 timing data analysis

"""

# Track characteristics database
TRACK_CHARACTERISTICS = {
    
    # === EUROPEAN TRACKS ===
    
    'Monaco': {
        'length_km': 3.337,
        'pit_loss_sec': 16,
        'overtaking_difficulty': 'VERY_HARD',
        'typical_stops': 1,
        'track_type': 'STREET',
        'drs_zones': 1,
        'notes': 'Slowest circuit, hardest to overtake, strategy critical'
    },
    
    'Monza': {
        'length_km': 5.793,
        'pit_loss_sec': 20,
        'overtaking_difficulty': 'EASY',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Fastest circuit, low downforce, slipstream crucial'
    },
    
    'Silverstone': {
        'length_km': 5.891,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'High-speed corners, variable weather'
    },
    
    'Spa': {
        'length_km': 7.004,
        'pit_loss_sec': 18,
        'overtaking_difficulty': 'EASY',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Longest circuit, weather unpredictable, Eau Rouge famous'
    },
    
    'Imola': {
        'length_km': 4.909,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'HARD',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Old-school track, narrow, hard to follow'
    },
    
    'Barcelona': {
        'length_km': 4.675,
        'pit_loss_sec': 18,
        'overtaking_difficulty': 'HARD',
        'typical_stops': 2,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'High tire degradation, testing venue'
    },
    
    'Zandvoort': {
        'length_km': 4.259,
        'pit_loss_sec': 17,
        'overtaking_difficulty': 'VERY_HARD',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Banked corners, narrow, sea breeze affects conditions'
    },
    
    'Austria': {
        'length_km': 4.318,
        'pit_loss_sec': 17,
        'overtaking_difficulty': 'EASY',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 3,
        'notes': 'Short lap, lots of overtaking, altitude affects engines'
    },
    
    'Hungary': {
        'length_km': 4.381,
        'pit_loss_sec': 18,
        'overtaking_difficulty': 'VERY_HARD',
        'typical_stops': 2,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Monaco without walls, strategy > speed'
    },
    
    # === MIDDLE EAST ===
    
    'Bahrain': {
        'length_km': 5.412,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 3,
        'notes': 'Night race, desert conditions, good overtaking'
    },
    
    'Saudi Arabia': {
        'length_km': 6.174,
        'pit_loss_sec': 20,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'STREET',
        'drs_zones': 3,
        'notes': 'Fastest street circuit, long straights, blind corners'
    },
    
    'Abu Dhabi': {
        'length_km': 5.281,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Twilight race, marina section, season finale'
    },
    
    'Qatar': {
        'length_km': 5.380,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'High-speed corners, extreme heat, tire stress'
    },
    
    # === AMERICAS ===
    
    'Miami': {
        'length_km': 5.410,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 2,
        'track_type': 'STREET',
        'drs_zones': 3,
        'notes': 'Stadium section, bumpy, high temperatures'
    },
    
    'Austin': {
        'length_km': 5.513,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 2,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'COTA, Turn 1 uphill, inspired by multiple famous corners'
    },
    
    'Mexico': {
        'length_km': 4.304,
        'pit_loss_sec': 17,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 3,
        'notes': 'High altitude (2,200m), thin air, long straight'
    },
    
    'Brazil': {
        'length_km': 4.309,
        'pit_loss_sec': 17,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Interlagos, anti-clockwise, rain likely, elevation changes'
    },
    
    'Las Vegas': {
        'length_km': 6.120,
        'pit_loss_sec': 20,
        'overtaking_difficulty': 'EASY',
        'typical_stops': 2,
        'track_type': 'STREET',
        'drs_zones': 2,
        'notes': 'Night race, very cold track, long straights, casino strip'
    },
    
    'Canada': {
        'length_km': 4.361,
        'pit_loss_sec': 18,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'SEMI_PERMANENT',
        'drs_zones': 3,
        'notes': 'Circuit Gilles Villeneuve, Wall of Champions, brake stress'
    },
    
    # === ASIA-PACIFIC ===
    
    'Singapore': {
        'length_km': 4.940,
        'pit_loss_sec': 22,
        'overtaking_difficulty': 'HARD',
        'typical_stops': 2,
        'track_type': 'STREET',
        'drs_zones': 3,
        'notes': 'Night race, humid, physically demanding, safety car likely'
    },
    
    'Japan': {
        'length_km': 5.807,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Suzuka, figure-8 layout, iconic Spoon Curve and 130R'
    },
    
    'China': {
        'length_km': 5.451,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'EASY',
        'typical_stops': 2,
        'track_type': 'PERMANENT',
        'drs_zones': 2,
        'notes': 'Shanghai, long back straight, variable weather'
    },
    
    'Australia': {
        'length_km': 5.278,
        'pit_loss_sec': 19,
        'overtaking_difficulty': 'MEDIUM',
        'typical_stops': 1,
        'track_type': 'SEMI_PERMANENT',
        'drs_zones': 4,
        'notes': 'Albert Park, season opener, lakeside, bumpy'
    },
    
}


# Default values for unknown tracks
DEFAULT_TRACK = {
    'length_km': 5.0,
    'pit_loss_sec': 19,
    'overtaking_difficulty': 'MEDIUM',
    'typical_stops': 1,
    'track_type': 'PERMANENT',
    'drs_zones': 2,
    'notes': 'Unknown track - using average values'
}


def get_track_info(circuit_name):
    """
    Get track characteristics by circuit name.
    
    Handles partial matches and common variations:
    - "Monaco Grand Prix" → matches "Monaco"
    - "British Grand Prix" → matches "Silverstone"
    - "Italian Grand Prix" → matches "Monza"
    
    Args:
        circuit_name: Name from FastF1 session.event['EventName']
    
    Returns:
        Dictionary with track characteristics
    """
    circuit_upper = circuit_name.upper()
    
    # Try exact match first
    for track_name, track_info in TRACK_CHARACTERISTICS.items():
        if track_name.upper() in circuit_upper:
            return track_info.copy()
    
    # Try common name variations
    name_mappings = {
        'BRITISH': 'Silverstone',
        'ITALIAN': 'Monza',
        'BELGIAN': 'Spa',
        'DUTCH': 'Zandvoort',
        'AUSTRIAN': 'Austria',
        'HUNGARIAN': 'Hungary',
        'SPANISH': 'Barcelona',
        'FRENCH': 'Paul Ricard',
        'GERMAN': 'Hockenheim',
        'PORTUGUESE': 'Portimao',
        'EMILIA': 'Imola',
        'UNITED STATES': 'Austin',
        'JAPANESE': 'Japan',
        'MEXICAN': 'Mexico',
        'BRAZILIAN': 'Brazil',
        'SINGAPORE': 'Singapore',
        'RUSSIAN': 'Sochi',
        'TURKISH': 'Turkey',
        'BAHRAIN': 'Bahrain',
        'ABU DHABI': 'Abu Dhabi',
        'SAUDI': 'Saudi Arabia',
        'AUSTRALIAN': 'Australia',
        'MIAMI': 'Miami',
        'LAS VEGAS': 'Las Vegas',
        'CANADIAN': 'Canada',
        'CHINESE': 'China',
        'QATAR': 'Qatar',
    }
    
    for key, track_name in name_mappings.items():
        if key in circuit_upper:
            if track_name in TRACK_CHARACTERISTICS:
                return TRACK_CHARACTERISTICS[track_name].copy()
    
    # If no match found, return defaults
    print(f"Unknown track: {circuit_name}, using defaults")
    return DEFAULT_TRACK.copy()