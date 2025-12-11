import { MapPin, Thermometer, Users } from 'lucide-react'

const TRACKS = [
  'Abu Dhabi Grand Prix',
  'Austrian Grand Prix',
  'Australian Grand Prix',
  'Azerbaijan Grand Prix',
  'Bahrain Grand Prix',
  'Belgian Grand Prix',
  'Brazilian Grand Prix',
  'British Grand Prix',
  'Canadian Grand Prix',
  'Chinese Grand Prix',
  'Dutch Grand Prix',
  'Emilia Romagna Grand Prix',
  'Hungarian Grand Prix',
  'Italian Grand Prix',
  'Japanese Grand Prix',
  'Las Vegas Grand Prix',
  'Mexican Grand Prix',
  'Miami Grand Prix',
  'Monaco Grand Prix',
  'Portuguese Grand Prix',
  'Qatar Grand Prix',
  'Saudi Arabian Grand Prix',
  'Singapore Grand Prix',
  'Spanish Grand Prix',
  'Turkish Grand Prix',
  'United States Grand Prix'
].sort()

const DRIVERS = [
  'ALB', 'ALO', 'BEA', 'BOT', 'DEV', 'DRU',
  'GAS', 'HAD', 'HAM', 'HUL', 'LAW', 'LEC',
  'MAG', 'NOR', 'OCO', 'PER', 'PIA', 'RIC',
  'RUS', 'SAI', 'SAR', 'STR', 'TSU', 'VER',
  'ZHO'
].sort()

const COMPOUNDS = ['SOFT', 'MEDIUM', 'HARD']

export default function RaceStateForm({ raceState, setRaceState }) {
  const updateField = (field, value) => {
    setRaceState(prev => ({ ...prev, [field]: value }))
  }

  const updateCompetitor = (relativePosition, field, value) => {
  setRaceState(prev => {
    const competitors = [...(prev.competitors || [])]
    const index = competitors.findIndex(c => c.relativePosition === relativePosition)
    
    if (index >= 0) {
      competitors[index] = { ...competitors[index], [field]: value }
    } else {
      const newComp = {
        relativePosition,
        driver_code: '',
        tire_age: 20,
        tire_compound: 'MEDIUM',
        gap_to_next: 2.0
      }
      newComp[field] = value
      competitors.push(newComp)
    }
    
    return { ...prev, competitors }
  })
}

  const getCompetitorValue = (relativePosition, field) => {
    const comp = raceState.competitors.find(c => c.relativePosition === relativePosition)
    return comp ? comp[field] : (field === 'tire_compound' ? 'MEDIUM' : field === 'tire_age' ? 20 : '')
  }

  const buildFinalCompetitors = () => {
    const position = raceState.position
    const competitors = []

    if (position >= 3) {
      const comp = raceState.competitors.find(c => c.relativePosition === -2)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position - 2,
          tire_age: comp.tire_age,
          tire_compound: comp.tire_compound,
          gap_to_us: -(raceState.gap_ahead + (comp.gap_to_next || 2.0))
        })
      }
    }

    if (position >= 2) {
      const comp = raceState.competitors.find(c => c.relativePosition === -1)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position - 1,
          tire_age: comp.tire_age,
          tire_compound: comp.tire_compound,
          gap_to_us: -raceState.gap_ahead
        })
      }
    }

    if (position <= 19) {
      const comp = raceState.competitors.find(c => c.relativePosition === 1)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position + 1,
          tire_age: comp.tire_age,
          tire_compound: comp.tire_compound,
          gap_to_us: raceState.gap_behind
        })
      }
    }

    if (position <= 18) {
      const comp = raceState.competitors.find(c => c.relativePosition === 2)
      if (comp && comp.driver_code) {
        competitors.push({
          driver_code: comp.driver_code,
          position: position + 2,
          tire_age: comp.tire_age,
          tire_compound: comp.tire_compound,
          gap_to_us: raceState.gap_behind + (comp.gap_to_next || 2.0)
        })
      }
    }

    return competitors
  }

  const handleSubmit = () => {
    const finalCompetitors = buildFinalCompetitors()
    setRaceState(prev => ({ ...prev, competitors: finalCompetitors }))
  }

  const position = raceState.position

  return (
    <div className="bg-white rounded-lg border border-gray-200 p-5">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Track
          </label>
          <div className="relative">
            <MapPin className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <select
              value={raceState.track_name}
              onChange={(e) => updateField('track_name', e.target.value)}
              className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              {TRACKS.map(track => (
                <option key={track} value={track}>{track}</option>
              ))}
            </select>
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Driver
          </label>
          <select
            value={raceState.driver}
            onChange={(e) => updateField('driver', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {DRIVERS.map(driver => (
              <option key={driver} value={driver}>{driver}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Current Lap
          </label>
          <input
            type="number"
            value={raceState.current_lap}
            onChange={(e) => updateField('current_lap', parseInt(e.target.value))}
            min="1"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Total Laps
          </label>
          <input
            type="number"
            value={raceState.total_laps}
            onChange={(e) => updateField('total_laps', parseInt(e.target.value))}
            min="1"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Position
          </label>
          <input
            type="number"
            value={raceState.position}
            onChange={(e) => updateField('position', parseInt(e.target.value))}
            min="1"
            max="20"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tire Compound
          </label>
          <select
            value={raceState.tire_compound}
            onChange={(e) => updateField('tire_compound', e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          >
            {COMPOUNDS.map(compound => (
              <option key={compound} value={compound}>{compound}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Tire Age (laps)
          </label>
          <input
            type="number"
            value={raceState.tire_age}
            onChange={(e) => updateField('tire_age', parseInt(e.target.value))}
            min="0"
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Gap to P{position - 1} (s)
          </label>
          <input
            type="number"
            step="0.1"
            value={raceState.gap_ahead}
            onChange={(e) => updateField('gap_ahead', parseFloat(e.target.value))}
            disabled={position === 1}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Gap to P{position + 1} (s)
          </label>
          <input
            type="number"
            step="0.1"
            value={raceState.gap_behind}
            onChange={(e) => updateField('gap_behind', parseFloat(e.target.value))}
            disabled={position === 20}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Track Temp (°C)
          </label>
          <div className="relative">
            <Thermometer className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="number"
              step="0.1"
              value={raceState.track_temp}
              onChange={(e) => updateField('track_temp', parseFloat(e.target.value))}
              className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Air Temp (°C)
          </label>
          <div className="relative">
            <Thermometer className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="number"
              step="0.1"
              value={raceState.air_temp}
              onChange={(e) => updateField('air_temp', parseFloat(e.target.value))}
              className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      <div className="border-t border-gray-200 pt-6">
        <div className="flex items-center mb-4">
          <Users className="w-5 h-5 text-gray-700 mr-2" />
          <label className="text-sm font-medium text-gray-700">
            Additional Competitors (Optional)
          </label>
        </div>

        <p className="text-sm text-gray-600 mb-4">
          Add tire data for nearby competitors to improve prediction accuracy.
        </p>

        <div className="space-y-6">
          {position >= 3 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                P{position - 2} (Two places ahead)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Driver</label>
                  <select
                    value={getCompetitorValue(-2, 'driver_code')}
                    onChange={(e) => updateCompetitor(-2, 'driver_code', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Skip</option>
                    {DRIVERS.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Compound</label>
                  <select
                    value={getCompetitorValue(-2, 'tire_compound')}
                    onChange={(e) => updateCompetitor(-2, 'tire_compound', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    {COMPOUNDS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Tire Age</label>
                  <input
                    type="number"
                    value={getCompetitorValue(-2, 'tire_age')}
                    onChange={(e) => updateCompetitor(-2, 'tire_age', parseInt(e.target.value))}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Gap to P{position - 1} (s)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={getCompetitorValue(-2, 'gap_to_next')}
                    onChange={(e) => updateCompetitor(-2, 'gap_to_next', parseFloat(e.target.value))}
                    placeholder="2.0"
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}

          {position >= 2 && (
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                P{position - 1} (Directly ahead - gap already known: {raceState.gap_ahead}s)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Driver</label>
                  <select
                    value={getCompetitorValue(-1, 'driver_code')}
                    onChange={(e) => updateCompetitor(-1, 'driver_code', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Skip</option>
                    {DRIVERS.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Compound</label>
                  <select
                    value={getCompetitorValue(-1, 'tire_compound')}
                    onChange={(e) => updateCompetitor(-1, 'tire_compound', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    {COMPOUNDS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Tire Age</label>
                  <input
                    type="number"
                    value={getCompetitorValue(-1, 'tire_age')}
                    onChange={(e) => updateCompetitor(-1, 'tire_age', parseInt(e.target.value))}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}

          {position <= 19 && (
            <div className="bg-blue-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                P{position + 1} (Directly behind - gap already known: {raceState.gap_behind}s)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Driver</label>
                  <select
                    value={getCompetitorValue(1, 'driver_code')}
                    onChange={(e) => updateCompetitor(1, 'driver_code', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Skip</option>
                    {DRIVERS.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Compound</label>
                  <select
                    value={getCompetitorValue(1, 'tire_compound')}
                    onChange={(e) => updateCompetitor(1, 'tire_compound', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    {COMPOUNDS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Tire Age</label>
                  <input
                    type="number"
                    value={getCompetitorValue(1, 'tire_age')}
                    onChange={(e) => updateCompetitor(1, 'tire_age', parseInt(e.target.value))}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}

          {position <= 18 && (
            <div className="bg-gray-50 rounded-lg p-4">
              <h4 className="text-sm font-medium text-gray-900 mb-3">
                P{position + 2} (Two places behind)
              </h4>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Driver</label>
                  <select
                    value={getCompetitorValue(2, 'driver_code')}
                    onChange={(e) => updateCompetitor(2, 'driver_code', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="">Skip</option>
                    {DRIVERS.map(d => <option key={d} value={d}>{d}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Compound</label>
                  <select
                    value={getCompetitorValue(2, 'tire_compound')}
                    onChange={(e) => updateCompetitor(2, 'tire_compound', e.target.value)}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    {COMPOUNDS.map(c => <option key={c} value={c}>{c}</option>)}
                  </select>
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Tire Age</label>
                  <input
                    type="number"
                    value={getCompetitorValue(2, 'tire_age')}
                    onChange={(e) => updateCompetitor(2, 'tire_age', parseInt(e.target.value))}
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
                <div>
                  <label className="block text-xs text-gray-600 mb-1">Gap to P{position + 1} (s)</label>
                  <input
                    type="number"
                    step="0.1"
                    value={getCompetitorValue(2, 'gap_to_next')}
                    onChange={(e) => updateCompetitor(2, 'gap_to_next', parseFloat(e.target.value))}
                    placeholder="2.0"
                    className="w-full px-2 py-1 text-sm border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  />
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}