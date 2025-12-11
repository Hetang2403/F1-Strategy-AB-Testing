import { useState } from 'react'
import axios from 'axios'
import { Play, AlertCircle } from 'lucide-react'
import RaceStateForm from './components/RaceStateForm'
import StrategyForm from './components/StrategyForm'
import ComparisonResults from './components/ComparisonResults'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000'

function App() {
  const [raceState, setRaceState] = useState({
    current_lap: 25,
    total_laps: 52,
    driver: 'VER',
    position: 2,
    tire_age: 25,
    tire_compound: 'MEDIUM',
    gap_ahead: 3.2,
    gap_behind: 5.8,
    track_name: 'British Grand Prix',
    track_temp: 45.0,
    air_temp: 22.0,
    competitors: []
  })

  const [strategyA, setStrategyA] = useState({
    name: 'Conservative 1-Stop',
    pit_laps: [30],
    tire_compounds: ['MEDIUM', 'HARD'],
    stint_plans: ['MANAGE', 'PUSH'],
    description: 'Single pit stop strategy'
  })

  const [strategyB, setStrategyB] = useState({
    name: 'Aggressive 2-Stop',
    pit_laps: [28, 42],
    tire_compounds: ['MEDIUM', 'SOFT', 'SOFT'],
    stint_plans: ['MANAGE', 'PUSH', 'PUSH'],
    description: 'Two stop strategy with softs'
  })

  const [results, setResults] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const buildFinalCompetitors = () => {
    const position = raceState.position
    const competitors = []

    const tempCompetitors = raceState.competitors || []

    if (position >= 3) {
      const comp = tempCompetitors.find(c => c.relativePosition === -2)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position - 2,
          tire_age: comp.tire_age || 20,
          tire_compound: comp.tire_compound || 'MEDIUM',
          gap_to_us: -(raceState.gap_ahead + (comp.gap_to_next || 2.0))
        })
      }
    }

    if (position >= 2) {
      const comp = tempCompetitors.find(c => c.relativePosition === -1)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position - 1,
          tire_age: comp.tire_age || 20,
          tire_compound: comp.tire_compound || 'MEDIUM',
          gap_to_us: -raceState.gap_ahead
        })
      }
    }

    if (position <= 19) {
      const comp = tempCompetitors.find(c => c.relativePosition === 1)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position + 1,
          tire_age: comp.tire_age || 20,
          tire_compound: comp.tire_compound || 'MEDIUM',
          gap_to_us: raceState.gap_behind
        })
      }
    }

    if (position <= 18) {
      const comp = tempCompetitors.find(c => c.relativePosition === 2)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position + 2,
          tire_age: comp.tire_age || 20,
          tire_compound: comp.tire_compound || 'MEDIUM',
          gap_to_us: raceState.gap_behind + (comp.gap_to_next || 2.0)
        })
      }
    }

    return competitors
  }

  const runSimulation = async () => {
    setLoading(true)
    setError(null)
    setResults(null)
    
    try {
      const finalCompetitors = buildFinalCompetitors()
      
      const requestData = {
        race_state: {
          ...raceState,
          competitors: finalCompetitors
        },
        strategy_a: strategyA,
        strategy_b: strategyB
      }

      console.log('Sending request:', requestData)
      
      const response = await axios.post(`${API_URL}/api/simulate`, requestData)
      
      setResults(response.data)
    } catch (err) {
      console.error('Simulation error:', err)
      setError(err.response?.data?.error || err.message || 'Simulation failed. Please check your inputs.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <header className="bg-white border-b border-gray-200">
        <div className="max-w-[1600px] mx-auto px-6 py-5">
          <h1 className="text-2xl font-semibold text-gray-900">
            F1 Strategy A/B Testing
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Machine Learning Strategy Comparison
          </p>
        </div>
      </header>

      <main className="max-w-[1600px] mx-auto px-6 py-6">
        <section className="mb-6">
          <h2 className="text-lg font-medium text-gray-900 mb-3">
            Race State
          </h2>
          <RaceStateForm raceState={raceState} setRaceState={setRaceState} />
        </section>

        <section className="mb-6">
          <h2 className="text-lg font-medium text-gray-900 mb-3">
            Compare Strategies
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <StrategyForm 
              strategy={strategyA} 
              setStrategy={setStrategyA}
              title="Strategy A"
            />
            <StrategyForm 
              strategy={strategyB} 
              setStrategy={setStrategyB}
              title="Strategy B"
            />
          </div>
        </section>

        <section className="mb-6">
          <div className="flex justify-center">
            <button
              onClick={runSimulation}
              disabled={loading}
              className="inline-flex items-center px-8 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors duration-200 shadow-sm"
            >
              <Play className="w-5 h-5 mr-2" />
              {loading ? 'Running Simulation...' : 'Run Simulation'}
            </button>
          </div>
        </section>

        {error && (
          <section className="mb-6">
            <div className="bg-red-50 border border-red-200 rounded-lg p-4">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-red-600 mt-0.5 mr-3 flex-shrink-0" />
                <p className="text-sm text-red-800">{error}</p>
              </div>
            </div>
          </section>
        )}

        {results && (
          <section>
            <h2 className="text-lg font-medium text-gray-900 mb-3">
              Results
            </h2>
            <ComparisonResults results={results} />
          </section>
        )}
      </main>

      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-[1600px] mx-auto px-6 py-5">
          <p className="text-sm text-gray-500 text-center">
            ML-Based Strategy Simulation | Built with React, Flask, and XGBoost
          </p>
        </div>
      </footer>
    </div>
  )
}

export default App
