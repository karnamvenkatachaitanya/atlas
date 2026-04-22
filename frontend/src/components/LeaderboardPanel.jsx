import React from "react";

export default function LeaderboardPanel({ rows, onReplay, onPdf }) {
  return (
    <div className="card h-72 overflow-auto">
      <h2 className="font-semibold mb-2">Leaderboard</h2>
      <div className="space-y-2 text-sm">
        {Array.isArray(rows) && rows.map((row) => (
          <div key={row.id} className="border border-slate-800 rounded p-2">
            <div>
              Episode #{row.id} ({row.mode})
            </div>
            <div className="text-emerald-300">Reward: {row.total_reward.toFixed(2)}</div>
            <div className="flex gap-2 mt-2">
              <button
                className="bg-slate-700 px-2 py-1 rounded"
                onClick={() => onReplay(row.id)}
              >
                Replay
              </button>
              <a className="bg-indigo-700 px-2 py-1 rounded" href={onPdf(row.id)} target="_blank">
                PDF
              </a>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
