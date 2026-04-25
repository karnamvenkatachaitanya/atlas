import React, { useState } from "react";
import { api } from "../services/api";

export default function DecisionLog({ decisions, done }) {
  const [feedbackState, setFeedbackState] = useState({});

  const handleFeedback = async (decision, idx, type) => {
    try {
      await api.feedback({
        episode_id: decision.episode_id || 0,
        day: decision.day,
        phase: decision.phase,
        action: decision.action,
        state: decision.state || {},
        preference: type
      });
      setFeedbackState(prev => ({ ...prev, [idx]: type }));
    } catch (err) {
      console.error("Feedback error:", err);
    }
  };

  return (
    <div className="card h-72 overflow-hidden flex flex-col">
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-semibold text-cyan-300">CEO Decision Log & Feedback</h2>
        <span className={`impact-badge ${done ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10" : "border-amber-500/40 text-amber-300 bg-amber-500/10"}`}>
          {done ? "DONE" : "LIVE"}
        </span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-1 text-sm">
        {decisions.length === 0 && (
          <div className="text-slate-500 italic text-sm py-4 text-center">Waiting for first decision...</div>
        )}
        {decisions.map((d, i) => (
          <div key={i} className="flex flex-col gap-1 py-2 border-b border-slate-800/50 last:border-0">
            <div className="flex items-center justify-between gap-2">
              <div className="flex items-center gap-2">
                <span className="text-slate-500 text-xs font-mono w-14 shrink-0">D{d.day} {d.phase?.[0]?.toUpperCase() || "?"}</span>
                <span className="text-slate-200 font-mono text-xs">
                  {typeof d.action === "string" ? d.action.replace(/_/g, " ") : "—"}
                </span>
              </div>
              <div className="flex gap-1">
                <button 
                  onClick={() => handleFeedback(d, i, "upvote")}
                  disabled={feedbackState[i]}
                  className={`px-2 py-0.5 text-xs rounded border ${feedbackState[i] === "upvote" ? "bg-emerald-500/20 text-emerald-400 border-emerald-500/30" : "border-slate-700 text-slate-400 hover:text-emerald-400 hover:border-emerald-500/30"} disabled:opacity-50 transition-colors`}
                  title="Good decision"
                >
                  👍
                </button>
                <button 
                  onClick={() => handleFeedback(d, i, "downvote")}
                  disabled={feedbackState[i]}
                  className={`px-2 py-0.5 text-xs rounded border ${feedbackState[i] === "downvote" ? "bg-rose-500/20 text-rose-400 border-rose-500/30" : "border-slate-700 text-slate-400 hover:text-rose-400 hover:border-rose-500/30"} disabled:opacity-50 transition-colors`}
                  title="Bad decision"
                >
                  👎
                </button>
              </div>
            </div>
            {d.rationale && (
              <div className="text-xs text-slate-400 italic pl-16 border-l-2 border-slate-800 ml-1">
                "{d.rationale}"
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
