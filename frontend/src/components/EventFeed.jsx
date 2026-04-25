import React from "react";

export default function EventFeed({ events }) {
  return (
    <div className="flex-1 overflow-y-auto space-y-2 pr-1">
      {events.length === 0 && (
        <div className="text-slate-500 text-sm italic py-4 text-center">
          No market events yet.
        </div>
      )}
      {events.map((e, i) => (
        <div key={i} className="flex gap-2 items-start text-sm">
          <span className="text-rose-400 shrink-0">⚠️</span>
          <span className="text-slate-300">{typeof e === "string" ? e.replace(/_/g, " ") : JSON.stringify(e)}</span>
        </div>
      ))}
    </div>
  );
}
