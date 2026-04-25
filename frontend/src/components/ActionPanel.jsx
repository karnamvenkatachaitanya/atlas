import React from "react";

const ACTIONS = [
  "hire_employee",
  "fire_employee",
  "increase_salaries",
  "assign_engineering_task",
  "launch_product",
  "run_ads",
  "negotiate_client",
  "reduce_costs",
  "raise_funding",
  "fix_bug_crisis",
  "improve_culture",
  "give_bonuses",
  "change_roadmap",
];

export default function ActionPanel({ onAction, disabled }) {
  return (
    <div className="glass-panel p-4 flex flex-col mt-4">
      <div className="flex flex-wrap justify-between items-center gap-2 mb-3">
        <h2 className="text-xl font-bold text-emerald-300">Impact Actions Console</h2>
        {disabled ? (
          <span className="impact-badge border-amber-500/50 text-amber-300 bg-amber-500/10">AI CEO ACTIVE</span>
        ) : (
          <span className="impact-badge border-emerald-500/50 text-emerald-300 bg-emerald-500/10">MANUAL CONTROL</span>
        )}
      </div>
      <p className="text-sm text-slate-400 mb-3">
        Trigger high-impact decisions and watch negotiation alignment, tool usage, and reward shift in real time.
      </p>
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 gap-2">
        {ACTIONS.map((actionName, idx) => (
          <button
            key={actionName}
            disabled={disabled}
            onClick={() => onAction(idx)}
            className={`px-3 py-2 text-xs font-mono rounded border transition-colors ${
              disabled
                ? "bg-slate-900/80 border-slate-700 text-slate-500 cursor-not-allowed"
                : "bg-slate-900/80 border-slate-600 text-slate-200 hover:bg-emerald-600 hover:text-white hover:border-emerald-500"
            }`}
          >
            {actionName.replace(/_/g, " ")}
          </button>
        ))}
      </div>
    </div>
  );
}
