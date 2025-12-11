import { Plus, Trash2 } from 'lucide-react'

const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']
const STINT_PLANS = ['PUSH', 'MANAGE', 'CONSERVE']

export default function StrategyForm({ strategy, setStrategy, title }) {
  const updateField = (field, value) => {
    setStrategy(prev => ({ ...prev, [field]: value }))
  }

  const updatePitLap = (index, value) => {
    const newPitLaps = [...strategy.pit_laps]
    newPitLaps[index] = parseInt(value)
    setStrategy(prev => ({ ...prev, pit_laps: newPitLaps }))
  }

  const addPitLap = () => {
    const lastLap = strategy.pit_laps[strategy.pit_laps.length - 1] || 20
    setStrategy(prev => ({
      ...prev,
      pit_laps: [...prev.pit_laps, lastLap + 10],
      tire_compounds: [...prev.tire_compounds, 'MEDIUM'],
      stint_plans: [...prev.stint_plans, 'MANAGE']
    }))
  }

  const removePitLap = (index) => {
    setStrategy(prev => ({
      ...prev,
      pit_laps: prev.pit_laps.filter((_, i) => i !== index),
      tire_compounds: prev.tire_compounds.filter((_, i) => i !== index + 1),
      stint_plans: prev.stint_plans.filter((_, i) => i !== index + 1)
    }))
  }

  const updateCompound = (index, value) => {
    const newCompounds = [...strategy.tire_compounds]
    newCompounds[index] = value
    setStrategy(prev => ({ ...prev, tire_compounds: newCompounds }))
  }

  const updateStintPlan = (index, value) => {
    const newPlans = [...strategy.stint_plans]
    newPlans[index] = value
    setStrategy(prev => ({ ...prev, stint_plans: newPlans }))
  }

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5">
      <h3 className="text-base font-medium text-gray-900 mb-4">{title}</h3>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Strategy Name
        </label>
        <input
          type="text"
          value={strategy.name}
          onChange={(e) => updateField('name', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
      </div>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Description
        </label>
        <input
          type="text"
          value={strategy.description}
          onChange={(e) => updateField('description', e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          placeholder="Optional description"
        />
      </div>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Starting Compound
        </label>
        <select
          value={strategy.tire_compounds[0]}
          onChange={(e) => updateCompound(0, e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {COMPOUNDS.map(c => (
            <option key={c} value={c}>{c}</option>
          ))}
        </select>
      </div>

      <div className="mb-3">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Starting Stint Plan
        </label>
        <select
          value={strategy.stint_plans[0]}
          onChange={(e) => updateStintPlan(0, e.target.value)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        >
          {STINT_PLANS.map(p => (
            <option key={p} value={p}>{p}</option>
          ))}
        </select>
      </div>

      <div className="border-t border-gray-200 pt-6">
        <div className="flex items-center justify-between mb-4">
          <label className="block text-sm font-medium text-gray-700">
            Pit Stops
          </label>
          <button
            onClick={addPitLap}
            className="inline-flex items-center text-sm text-blue-600 hover:text-blue-700"
          >
            <Plus className="w-4 h-4 mr-1" />
            Add Pit Stop
          </button>
        </div>

        <div className="space-y-4">
          {strategy.pit_laps.map((lap, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <span className="text-sm font-medium text-gray-700">
                  Pit Stop {index + 1}
                </span>
                {strategy.pit_laps.length > 1 && (
                  <button
                    onClick={() => removePitLap(index)}
                    className="text-gray-400 hover:text-red-600"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                )}
              </div>

              <div className="grid grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">
                    Lap
                  </label>
                  <input
                    type="number"
                    value={lap}
                    onChange={(e) => updatePitLap(index, e.target.value)}
                    min="1"
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  />
                </div>

                <div>
                  <label className="block text-xs text-gray-600 mb-1">
                    New Compound
                  </label>
                  <select
                    value={strategy.tire_compounds[index + 1]}
                    onChange={(e) => updateCompound(index + 1, e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {COMPOUNDS.map(c => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-xs text-gray-600 mb-1">
                    Stint Plan
                  </label>
                  <select
                    value={strategy.stint_plans[index + 1]}
                    onChange={(e) => updateStintPlan(index + 1, e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  >
                    {STINT_PLANS.map(p => (
                      <option key={p} value={p}>{p}</option>
                    ))}
                  </select>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}