import React from "react";

export default function ToolCallsPanel({ toolCalls }) {
  const calls = toolCalls || [];
  return (
    <div className="card h-72 overflow-auto">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-fuchsia-300">Tool Intelligence</h2>
        <span className="impact-badge border-fuchsia-500/40 text-fuchsia-300 bg-fuchsia-500/10">
          {calls.length} calls
        </span>
      </div>
      {calls.length === 0 ? (
        <div className="text-slate-400 text-sm">No tool calls yet.</div>
      ) : (
        <ul className="space-y-2 text-sm">
          {calls.map((c, idx) => (
            <li key={idx} className="border border-slate-700 rounded-xl p-2 bg-slate-950/50">
              <div className="flex items-center justify-between gap-2">
                <span className="text-fuchsia-300 font-mono">{c.tool}</span>
              </div>
              {c.result && (
                <pre className="mt-2 text-xs text-slate-200 whitespace-pre-wrap">
                  {JSON.stringify(c.result, null, 2)}
                </pre>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

