import { Trophy, Clock, Flag } from 'lucide-react'

export default function ComparisonResults({ results }) {
  const { result_a, result_b, comparison } = results

  const getWinnerStyle = (isWinner) => 
    isWinner ? 'border-blue-500 bg-blue-50' : 'border-gray-200 bg-white'

  const isAWinner = comparison.winner === 'strategy_a'
  const isBWinner = comparison.winner === 'strategy_b'

  return (
    <div className="space-y-4">
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-center">
          <Trophy className="w-5 h-5 text-blue-600 mr-3" />
          <div>
            <p className="text-sm font-medium text-blue-900">
              Recommended Strategy
            </p>
            <p className="text-lg font-semibold text-blue-700">
              {comparison.winner_name}
            </p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className={`rounded-lg border-2 ${getWinnerStyle(isAWinner)} p-6`}>
          <h3 className="text-base font-semibold text-gray-900 mb-4">
            {result_a.strategy_name}
          </h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <div className="flex items-center text-sm text-gray-600">
                <Flag className="w-4 h-4 mr-2" />
                Final Position
              </div>
              <span className="text-lg font-semibold text-gray-900">
                P{result_a.predicted_position}
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <div className="flex items-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-2" />
                Total Time
              </div>
              <span className="text-base font-medium text-gray-900">
                {result_a.predicted_time.toFixed(1)}s
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <span className="text-sm text-gray-600">Avg Lap Time</span>
              <span className="text-base font-medium text-gray-900">
                {result_a.avg_lap_time.toFixed(3)}s
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <span className="text-sm text-gray-600">Pit Stops</span>
              <span className="text-base font-medium text-gray-900">
                {result_a.total_pits}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Pit Laps</span>
              <span className="text-base font-medium text-gray-900">
                {result_a.pit_laps.join(', ')}
              </span>
            </div>
          </div>
        </div>

        <div className={`rounded-lg border-2 ${getWinnerStyle(isBWinner)} p-6`}>
          <h3 className="text-base font-semibold text-gray-900 mb-4">
            {result_b.strategy_name}
          </h3>

          <div className="space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <div className="flex items-center text-sm text-gray-600">
                <Flag className="w-4 h-4 mr-2" />
                Final Position
              </div>
              <span className="text-lg font-semibold text-gray-900">
                P{result_b.predicted_position}
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <div className="flex items-center text-sm text-gray-600">
                <Clock className="w-4 h-4 mr-2" />
                Total Time
              </div>
              <span className="text-base font-medium text-gray-900">
                {result_b.predicted_time.toFixed(1)}s
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <span className="text-sm text-gray-600">Avg Lap Time</span>
              <span className="text-base font-medium text-gray-900">
                {result_b.avg_lap_time.toFixed(3)}s
              </span>
            </div>

            <div className="flex items-center justify-between pb-3 border-b border-gray-200">
              <span className="text-sm text-gray-600">Pit Stops</span>
              <span className="text-base font-medium text-gray-900">
                {result_b.total_pits}
              </span>
            </div>

            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Pit Laps</span>
              <span className="text-base font-medium text-gray-900">
                {result_b.pit_laps.join(', ')}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="bg-gray-50 border border-gray-200 rounded-lg p-6">
        <h3 className="text-base font-medium text-gray-900 mb-4">Analysis</h3>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600 mb-1">Position Difference</p>
            <p className="text-base font-semibold text-gray-900">
              {Math.abs(comparison.position_diff)} position{Math.abs(comparison.position_diff) !== 1 ? 's' : ''}
            </p>
          </div>

          <div>
            <p className="text-sm text-gray-600 mb-1">Time Difference</p>
            <p className="text-base font-semibold text-gray-900">
              {Math.abs(comparison.time_diff).toFixed(1)}s
            </p>
          </div>
        </div>
      </div>
    </div>
  )
}