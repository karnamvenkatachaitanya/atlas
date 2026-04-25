import React, { useState } from "react";

export default function RLTrainingPanel({ episodeRewards = [] }) {
  const [training, setTraining] = useState(false);
  const [metrics, setMetrics] = useState(null);

  const startTraining = async () => {
    setTraining(true);
    try {
      const res = await fetch("/api/train", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ episodes: 20, algorithm: "ppo" })
      });
      
      if (!res.ok) {
        throw new Error(`Training failed: ${res.status} ${res.statusText}`);
      }
      
      const data = await res.json();
      setMetrics(data);
    } catch (err) {
      console.error("Training failed:", err);
      alert("Training failed. Make sure the backend server is running on port 8000.");
    }
    setTraining(false);
  };

  return (
    <div className="card flex flex-col h-80">
      <h2 className="text-xl font-bold mb-3 text-cyan-300">RL Training</h2>
      
      <div className="flex-1 space-y-4 overflow-y-auto">
        <button 
          className="w-full bg-cyan-600 rounded-xl px-4 py-2 font-semibold hover:bg-cyan-500 disabled:opacity-50 disabled:cursor-not-allowed"
          onClick={startTraining}
          disabled={training}
        >
          {training ? "Training..." : "Start PPO Training"}
        </button>

        {metrics && metrics.mean_reward != null && (
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-slate-400">Mean Reward:</span>
              <span className="text-white font-mono">{Number(metrics.mean_reward).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Best Episode:</span>
              <span className="text-emerald-400 font-mono">{Number(metrics.max_reward).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Worst Episode:</span>
              <span className="text-rose-400 font-mono">{Number(metrics.min_reward).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Std Dev:</span>
              <span className="text-amber-400 font-mono">{Number(metrics.std_reward).toFixed(3)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-400">Episodes Trained:</span>
              <span className="text-white font-mono">{metrics.episodes_trained || 0}</span>
            </div>
          </div>
        )}

        {/* Reward curve visualization */}
        {(episodeRewards.length > 0 || (metrics && metrics.episode_rewards?.length > 0)) && (
          <div className="mt-4">
            <div className="text-xs text-slate-400 mb-2">Reward Curve</div>
            <div className="h-32 flex items-end gap-1">
              {(metrics?.episode_rewards || episodeRewards)
                .filter(r => r != null)
                .map((r, i) => {
                const rewards = (metrics?.episode_rewards || episodeRewards).filter(x => x != null);
                const maxR = Math.max(...rewards.map(Math.abs));
                const height = maxR > 0 ? (Math.abs(r) / maxR) * 100 : 0;
                const isPositive = r >= 0;
                return (
                  <div
                    key={i}
                    className={`flex-1 rounded-sm ${isPositive ? 'bg-emerald-600' : 'bg-rose-600'}`}
                    style={{ height: `${Math.max(height, 5)}%` }}
                    title={`Ep ${i+1}: ${Number(r).toFixed(3)}`}
                  />
                );
              })}
            </div>
          </div>
        )}

        {!metrics && episodeRewards.length === 0 && (
          <div className="text-center text-slate-500 text-sm py-4">
            Click "Start PPO Training" to begin training episodes
          </div>
        )}
      </div>
    </div>
  );
}
