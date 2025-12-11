# F1 Strategy A/B Testing Framework

A machine learning-powered Formula 1 race strategy comparison tool that simulates and compares different pit stop strategies using historical race data.

**Live Demo:** https://your-vercel-url.vercel.app  
**Backend API:** https://your-railway-url.railway.app

## Project Overview

This project enables F1 teams and enthusiasts to compare different race strategies using machine learning models trained on 86,000+ laps of historical F1 data. The system simulates complete races with 20 drivers, predicting lap times and final positions based on tire strategy choices.

### Key Capabilities

- Strategy comparison with side-by-side results
- ML-based lap time predictions (XGBoost, MAE: 2.6s)
- Full race simulation with 20 drivers
- Support for 26 F1 tracks with accurate characteristics
- Optional competitor tire data input for improved accuracy
- Validated against Abu Dhabi 2024 GP actual race outcomes

## Technology Stack

### Backend
- Python 3.11
- Flask 3.0 (RESTful API)
- XGBoost 2.0 (ML models)
- scikit-learn 1.4
- Pandas 2.2
- NumPy 1.26
- Gunicorn (WSGI server)

### Frontend
- React 18
- Vite (build tool)
- Tailwind CSS
- Axios (HTTP client)
- Lucide React (icons)

### Deployment
- Railway (backend hosting with Docker)
- Vercel (frontend hosting)
- GitHub (version control)

## Model Performance

### Pit Stop Predictor
- F1-Score: 0.80
- Training Data: 64,237 laps from 74 races
- Features: 52 leakage-free features
- Algorithm: XGBoost Classifier

### Lap Time Predictor
- Mean Absolute Error: 2.587 seconds
- R² Score: 0.441
- Training Set: 50,628 laps (80%)
- Test Set: 13,136 laps (20%)
- Algorithm: XGBoost Regressor

**Key Finding**: Tire compound dominates lap time prediction with 46.1% feature importance, aligning with F1 domain knowledge that tire choice is the primary determinant of pace.

## Validation Against Real Race Data

**Test Case:** Abu Dhabi 2024 Grand Prix, Lando Norris

**Scenario:**
- Position: P2
- Current Lap: 20 of 58
- Tire: MEDIUM compound, 20 laps old
- Gap ahead: 3.8s (to P1)
- Gap behind: 2.2s (to P3)

**Strategies Compared:**
1. Conservative 1-Stop: Pit lap 26, MEDIUM to HARD (actual McLaren strategy)
2. Alternative 2-Stop: Pit laps 23 & 43, MEDIUM to HARD to MEDIUM

**Model Prediction:** Strategy 1 (1-stop) maintains P2 position  
**Actual Race Result:** Norris finished P2 with 1-stop strategy  
**Validation Status:** Model correctly identified optimal strategy

## Quick Start

### Try the Live Demo

1. Visit the live application
2. Select "Abu Dhabi Grand Prix" as track
3. Configure race state:
   - Driver: NOR
   - Position: 2
   - Current Lap: 20 / Total Laps: 58
   - Tire: MEDIUM, Age: 20 laps
   - Gap Ahead: 3.8s, Gap Behind: 2.2s
4. Define Strategy A: 1-stop at lap 26 (MEDIUM to HARD)
5. Define Strategy B: 2-stop at laps 23 & 43
6. Run simulation

**Expected Result:** Conservative 1-stop wins with P2 finish

## Local Development

### Prerequisites
- Python 3.11 or higher
- Node.js 18 or higher
- npm or yarn package manager

### Backend Setup
```bash
# Clone repository
git clone https://github.com/Hetang2403/F1-Strategy-AB-Testing.git
cd F1-Strategy-AB-Testing

# Install Python dependencies
cd backend
pip install -r requirements.txt

# Start Flask development server
python app.py

# Server runs at http://localhost:5000
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd frontend

# Install Node dependencies
npm install

# Start development server
npm run dev

# Application runs at http://localhost:5173
```

## Project Structure
```
F1-Strategy-AB-Testing/
│
├── backend/                    # Flask REST API
│   ├── app.py                 # API endpoints and CORS configuration
│   ├── requirements.txt       # Python dependencies
│   └── Dockerfile            # Railway deployment configuration
│
├── frontend/                   # React web application
│   ├── src/
│   │   ├── App.jsx           # Main application component
│   │   └── components/       # React UI components
│   │       ├── RaceStateForm.jsx         # Race configuration input
│   │       ├── StrategyForm.jsx          # Strategy definition
│   │       └── ComparisonResults.jsx     # Results display
│   ├── package.json
│   └── vercel.json           # Vercel deployment configuration
│
├── src/                        # Core simulation logic
│   ├── data/
│   │   └── fetch_race_data.py # Data extraction pipeline (Phase 1)
│   └── testing/               # Strategy simulation modules
│       ├── strategy.py        # Strategy class with F1 rule validation
│       ├── race_state.py      # Race state representation
│       ├── driver_state.py    # Driver state tracking
│       ├── lap_time_predictor.py  # ML prediction wrapper
│       ├── race_simulator.py  # Full race simulation engine
│       └── strategy_simulator.py  # High-level simulation interface
│
├── config/
│   └── track_characteristics.py  # F1 track database (26 circuits)
│
├── data/
│   ├── models/                # Trained ML model artifacts
│   │   ├── laptime_predictor.pkl      # XGBoost regressor
│   │   ├── laptime_features.pkl       # Feature list
│   │   ├── laptime_encoders.pkl       # Label encoders
│   │   ├── track_baselines.pkl        # Track-specific baselines
│   │   └── laptime_metadata.pkl       # Model metadata
│   ├── raw/                   # Raw race data (86,000+ laps)
│   └── processed/             # Processed datasets
│
├── Dockerfile                 # Root Docker configuration
└── README.md
```

## API Documentation

### Base URL
Production: `https://your-railway-url.railway.app`  
Development: `http://localhost:5000`

### Endpoints

#### Health Check
```
GET /
```

Response:
```json
{
  "status": "online",
  "service": "F1 Strategy A/B Testing API",
  "version": "1.0.0"
}
```

#### Run Strategy Simulation
```
POST /api/simulate
Content-Type: application/json
```

Request Body:
```json
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
    "competitors": []
  },
  "strategy_a": {
    "name": "Conservative 1-Stop",
    "pit_laps": [30],
    "tire_compounds": ["MEDIUM", "HARD"],
    "stint_plans": ["MANAGE", "PUSH"],
    "description": "Single pit stop strategy"
  },
  "strategy_b": {
    "name": "Aggressive 2-Stop",
    "pit_laps": [28, 42],
    "tire_compounds": ["MEDIUM", "SOFT", "SOFT"],
    "stint_plans": ["MANAGE", "PUSH", "PUSH"],
    "description": "Two stop strategy"
  }
}
```

Response:
```json
{
  "success": true,
  "result_a": {
    "predicted_position": 2,
    "predicted_time": 3407.2,
    "avg_lap_time": 87.365,
    "total_pits": 1,
    "pit_laps": [30]
  },
  "result_b": {
    "predicted_position": 4,
    "predicted_time": 3430.7,
    "avg_lap_time": 87.967,
    "total_pits": 2,
    "pit_laps": [28, 42]
  },
  "comparison": {
    "winner": "strategy_a",
    "winner_name": "Conservative 1-Stop",
    "position_diff": 2,
    "time_diff": 23.5
  }
}
```

#### Get Available Tracks
```
GET /api/tracks
```

#### Get Tire Compounds
```
GET /api/compounds
```

#### Get Stint Plans
```
GET /api/stint-plans
```

## Key Technical Decisions

### 1. Feature Selection for Lap Time Prediction

**Challenge:** Initial model included outcome features (AvgSpeed, AvgThrottle) that are results of tire degradation rather than predictors.

**Solution:** Restricted to predictive-only features available before each lap:
- Tire state: TyreLife, Compound, FreshTyre
- Race context: LapNumber, Position, GapToAhead
- Environmental: TrackTemp, AirTemp, OvertakingDifficulty

**Result:** Eliminated circular reasoning while maintaining acceptable MAE (2.6s).

### 2. Model Architecture Choice

**Evaluation:** Compared tree-based models (XGBoost, Random Forest) against LSTM neural networks.

**Finding:** XGBoost significantly outperformed LSTM (MAE: 2.6s vs 6.0s).

**Explanation:** 
- Insufficient sequential data (64k laps) for effective deep learning
- Tabular data with strong feature relationships favor tree-based methods
- XGBoost handles non-linear relationships without requiring massive datasets

### 3. Relative vs Absolute Accuracy

**Insight:** For A/B testing, relative comparison accuracy is more important than absolute lap time precision.

**Implication:** 2.6s MAE is acceptable when both strategies are evaluated under identical model biases, as the comparison remains valid.

### 4. Domain Knowledge Integration

**Observation:** Tire compound explains 46.1% of lap time variance in the trained model.

**Validation:** This aligns with F1 engineering reality where tire choice is the primary pace determinant.

**Benefit:** Model learned and reinforces actual F1 strategy principles rather than spurious correlations.

### 5. Modular System Architecture

**Design:** Separated simulation into 4 focused modules (370 lines total):
- `driver_state.py`: State management (80 lines)
- `lap_time_predictor.py`: ML wrapper (90 lines)
- `race_simulator.py`: Core simulation (120 lines)
- `strategy_simulator.py`: High-level interface (80 lines)

**Benefits:**
- Easier unit testing
- Clear separation of concerns
- Simplified maintenance and extension

## Development Timeline

### Phase 1: Data Pipeline (Complete)
- Extracted 71 features from FastF1 API
- Processed 74 races spanning 2022-2024 seasons
- Generated 86,000+ laps of training data
- Implemented advanced features: damage detection, penalty tracking, traffic analysis

### Phase 2A: Baseline Models (Complete)
- Built pit stop predictor achieving F1-score of 0.80
- Implemented proper train/test splits with temporal validation
- Identified and resolved data leakage issues
- Achieved production-ready performance metrics

### Phase 2B: Model Optimization (Complete)
- Explored LSTM architectures (learned limitations for this dataset)
- Optimized XGBoost hyperparameters
- Conducted comprehensive feature importance analysis
- Validated models against 2024 race outcomes

### Phase 2C: Production Deployment (Complete)
- Developed modular race simulation engine
- Created lap time prediction model (MAE: 2.6s, R²: 0.441)
- Built full-stack web application with React and Flask
- Deployed to production infrastructure (Railway + Vercel)
- Implemented CORS, error handling, and API documentation

### Phase 3: Future Enhancements (Planned)
- Real-time race data integration during live events
- Historical race analysis with "what-if" scenarios
- Driver-specific prediction models
- Weather probability integration
- Native mobile applications (iOS/Android)
- Multi-user collaboration features

## Data Sources

Training data extracted using FastF1 API:
- **Seasons:** 2022, 2023, 2024
- **Total Races:** 74
- **Total Laps:** 86,000+
- **Features per Lap:** 71
- **Data Quality:** 95% complete after cleaning

### Feature Categories
- Core racing data (15 features): lap times, positions, tire state
- Weather data (4 features): per-lap temperature and rainfall
- Telemetry (11 features): speed, throttle, braking patterns
- Position dynamics (2 features): gaps to cars ahead/behind
- Track status (4 features): safety car, VSC, yellow flags
- Pit stops (4 features): timing, duration, type classification
- Penalties (4 features): detection and tracking
- Damage (4 features): from messages and telemetry anomalies
- Track characteristics (6 features): circuit-specific attributes
- Qualifying context (3 features): session and tire allocation
- Traffic (2 features): backmarker detection
- Overtakes (3 features): position change tracking
- Race results (3 features): final positions and status

## Contributing

This is a student project developed for educational purposes. Suggestions and feedback are welcome through GitHub issues.

To contribute:
1. Fork the repository
2. Create a feature branch
3. Commit your changes with clear messages
4. Push to your branch
5. Open a Pull Request with detailed description

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Acknowledgments

- FastF1 project for comprehensive F1 data API
- Open source community for excellent tools and libraries
- F1 community for domain knowledge and strategy insights
- Anthropic Claude for development assistance

## Contact

**Developer:** Hetang  
**GitHub:** [@Hetang2403](https://github.com/Hetang2403)  
**Project Repository:** https://github.com/Hetang2403/F1-Strategy-AB-Testing

---

**Student Project - 2024**

Built as a comprehensive demonstration of machine learning, software engineering, and full-stack development capabilities.
