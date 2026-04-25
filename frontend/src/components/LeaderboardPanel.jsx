import React from "react";

export default function LeaderboardPanel({ rows, onReplay }) {
  if (!rows || rows.length === 0) {
    return (
      <div className="card col-span-full">
        <h2 className="text-xl font-bold mb-4">🏆 AI Agents Leaderboard</h2>
        <div className="text-slate-500 text-sm">No completed episodes yet. Run a simulation to populate the leaderboard.</div>
      </div>
    );
  }

  return (
    <div className="card col-span-full">
      <h2 className="text-xl font-bold mb-4">🏆 AI Agents Leaderboard</h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-slate-800 text-slate-400 text-sm">
              <th className="pb-2">ID</th>
              <th className="pb-2">Agent</th>
              <th className="pb-2">Preset</th>
              <th className="pb-2 text-right">Final Revenue</th>
              <th className="pb-2 text-right">Total Reward</th>
              <th className="pb-2 text-center">Morale</th>
              <th className="pb-2 text-center">Steps</th>
              <th className="pb-2 text-center">Action</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const summary = row.summary || {};
              const morale = typeof summary.morale === "number" ? summary.morale.toFixed(0) : "—";
              return (
                <tr key={row.id} className="border-b border-slate-800/50 last:border-0 hover:bg-slate-800/30 transition-colors">
                  <td className="py-3 text-slate-500 font-mono text-xs">#{row.id}</td>
                  <td className="py-3 font-semibold">
                    {row.policy_name === "ppo" ? "🤖 Atlas Prime" : "🎲 Random Agent"}
                  </td>
                  <td className="py-3 text-slate-400 capitalize text-sm">{row.mode}</td>
                  <td className="py-3 text-right text-emerald-400 font-mono">
                    ${Number(row.final_revenue || 0).toLocaleString()}
                  </td>
                  <td className="py-3 text-right text-purple-400 font-mono">
                    {Number(row.total_reward || 0).toFixed(1)}
                  </td>
                  <td className="py-3 text-center text-cyan-400 font-mono">{morale}%</td>
                  <td className="py-3 text-center text-slate-400 font-mono">{row.steps || 0}</td>
                  <td className="py-3 text-center">
                    <button
                      onClick={() => onReplay(row.id)}
                      className="text-xs bg-slate-800 hover:bg-slate-700 px-3 py-1 rounded-lg border border-slate-700"
                    >
                      Replay
                    </button>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
