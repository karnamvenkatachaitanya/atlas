import React from "react";

export default function Header({ day, phase, onReset, onReport }) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-4 mb-8">
      <div className="flex items-center gap-4">
        <div className="status-chip neon-border-cyan">
          <span className="text-cyan-400 font-bold">Day {day}</span>
          <span className="w-1 h-1 rounded-full bg-slate-600" />
          <span className="text-slate-300 capitalize">{phase} Phase</span>
          <span className="text-amber-400 text-lg">☀️</span>
        </div>
      </div>

      <div className="flex items-center gap-3">
        <button className="btn-secondary" onClick={onReset}>
          🔄 Reset Simulation
        </button>
        <button className="btn-primary flex items-center gap-2" onClick={onReport}>
          📄 Generate Report
        </button>
      </div>
    </div>
  );
}
