import React from "react";

export default function NegotiationPanel({ negotiation, deptReports }) {
  const neg = negotiation || null;
  const reports = deptReports || [];

  return (
    <div className="card h-72 overflow-auto">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-cyan-300">Negotiation Engine</h2>
        {neg && (
          <span
            className={`impact-badge ${
              neg.matched
                ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10"
                : "border-rose-500/40 text-rose-300 bg-rose-500/10"
            }`}
          >
            {neg.matched ? "ALIGNED" : "MISALIGNED"}
          </span>
        )}
      </div>

      {!neg ? (
        <div className="text-slate-400 text-sm">No negotiation signal yet.</div>
      ) : (
        <div className="text-sm">
          <div className="flex flex-wrap gap-2 items-center">
            <span className="text-slate-400">Preferred:</span>
            <span className="text-cyan-300 font-mono">{neg.preferred_action_name}</span>
          </div>
          {neg.rationale && <div className="text-slate-400 text-xs mt-2">{neg.rationale}</div>}
        </div>
      )}

      <div className="mt-4">
        <h3 className="font-semibold text-sm mb-2">Department Reports</h3>
        {reports.length === 0 ? (
          <div className="text-slate-400 text-sm">No reports yet.</div>
        ) : (
          <ul className="space-y-1 text-sm">
            {reports.map((r, idx) => (
              <li key={idx} className="flex items-center justify-between gap-2 border border-slate-800 rounded-lg px-2 py-1">
                <span className="text-slate-200">{r.dept}</span>
                <span className="text-slate-400">{r.alert}</span>
                <span className="text-amber-300 font-mono">{r.suggested_action}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

