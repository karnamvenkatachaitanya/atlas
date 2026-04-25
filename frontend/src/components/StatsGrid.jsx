import React from "react";

const usd0 = new Intl.NumberFormat("en-US", {
  maximumFractionDigits: 0,
});

function Stat({ title, value, tone = "border-slate-700 bg-slate-900/70" }) {
  return (
    <div className={`stat-card ${tone}`}>
      <div className="text-slate-400 text-xs uppercase tracking-wide">{title}</div>
      <div className="text-2xl font-bold mt-1">{value}</div>
    </div>
  );
}

export default function StatsGrid({ state }) {
  if (!state) return null;
  return (
    <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
      <Stat
        title="Cash Balance"
        value={`$${usd0.format(Math.round(state.cash_balance))}`}
        tone="border-emerald-600/40 bg-emerald-950/25"
      />
      <Stat
        title="Revenue"
        value={`$${usd0.format(Math.round(state.revenue))}`}
        tone="border-cyan-600/40 bg-cyan-950/25"
      />
      <Stat
        title="Burn Rate"
        value={`$${usd0.format(Math.round(state.burn_rate))}`}
        tone="border-rose-600/40 bg-rose-950/25"
      />
      <Stat
        title="Morale"
        value={state.employee_morale.toFixed(1)}
        tone="border-violet-600/40 bg-violet-950/25"
      />
      <Stat
        title="Customer Sat"
        value={state.customer_satisfaction.toFixed(1)}
        tone="border-amber-600/40 bg-amber-950/25"
      />
    </div>
  );
}
