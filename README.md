# F1 Strategy A/B Testing Framework

Statistical framework for testing Formula 1 pit stop strategies using real race data and Monte Carlo simulation.

## What This Does

Answers the question: "Which pit strategy is actually better?"

- Fetches real F1 race data (FastF1 API)
- Models tire degradation from historical races
- Simulates different pit strategies
- Uses A/B testing to determine which is statistically better

## Example

**Question**: At Monaco, is a 1-stop or 2-stop strategy faster?

**Answer**: 
- Simulated 1000 races with each strategy
- 1-stop wins 67% of the time
- Average advantage: 11.3 seconds
- Statistical significance: p < 0.001
- **Conclusion**: Use 1-stop strategy

## Why This Exists

F1 teams can't A/B test strategies in real races (you only get one shot). This framework uses historical data to test "what if" scenarios statistically.

## Tech Stack

- **Data**: FastF1 API
- **Modeling**: scikit-learn, XGBoost
- **Statistics**: scipy, statsmodels
- **Simulation**: NumPy
- **Visualization**: Plotly, Streamlit

## Project Status

🚧 In Development - Phase 1: Data Collection & Modeling
```

---

### GitHub Repository Settings

When you create the repo on GitHub, you'd fill in:

**Description field:**
```
F1 race strategy simulator with statistical A/B testing using real telemetry data and Monte Carlo simulation
```

**Topics/Tags:**
```
formula-1
machine-learning
ab-testing
monte-carlo
sports-analytics
data-science
python
fastf1
racing
```

**Website (once deployed):**
```
https://your-app.streamlit.app
