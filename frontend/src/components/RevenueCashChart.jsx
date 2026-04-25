import React from "react";

import {
  CartesianGrid,
  Label,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

function formatMoney(value) {
  const n = Number(value || 0);
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(2)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${Math.round(n)}`;
}

export default function RevenueCashChart({ data, state }) {
  const cash = Number(state?.cash_balance || 0);
  const burn = Math.max(1, Number(state?.burn_rate || 0));
  const runwayDays = cash / burn;
  const runwayLabel = runwayDays >= 90 ? "Healthy" : runwayDays >= 45 ? "Watch" : "Critical";
  const runwayTone =
    runwayLabel === "Healthy"
      ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10"
      : runwayLabel === "Watch"
        ? "border-amber-500/40 text-amber-300 bg-amber-500/10"
        : "border-rose-500/40 text-rose-300 bg-rose-500/10";

  return (
    <div className="card h-72">
      <div className="flex items-center justify-between gap-2 mb-2">
        <h2 className="font-semibold text-emerald-300">Revenue & Cash Runway</h2>
        <span className={`impact-badge ${runwayTone}`}>
          Runway: {runwayDays.toFixed(1)}d ({runwayLabel})
        </span>
      </div>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data} margin={{ top: 10, right: 10, left: 8, bottom: 14 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="step" stroke="#94a3b8" tick={{ fontSize: 11 }}>
            <Label value="Episode Step" position="insideBottom" offset={-8} fill="#94a3b8" />
          </XAxis>
          <YAxis
            yAxisId="left"
            stroke="#22c55e"
            tick={{ fontSize: 11 }}
            tickFormatter={formatMoney}
            width={66}
          >
            <Label angle={-90} position="insideLeft" value="Revenue (USD)" fill="#22c55e" />
          </YAxis>
          <YAxis
            yAxisId="right"
            orientation="right"
            stroke="#38bdf8"
            tick={{ fontSize: 11 }}
            tickFormatter={formatMoney}
            width={66}
          >
            <Label angle={90} position="insideRight" value="Cash (USD)" fill="#38bdf8" />
          </YAxis>
          <Tooltip
            formatter={(value, name) => [formatMoney(value), name === "revenue" ? "Revenue" : "Cash Balance"]}
            labelFormatter={(label) => `Step ${label}`}
            contentStyle={{
              backgroundColor: "rgba(15,23,42,0.95)",
              border: "1px solid #334155",
              borderRadius: "10px",
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="revenue"
            yAxisId="left"
            stroke="#22c55e"
            dot={false}
            strokeWidth={2}
            name="Revenue"
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
          <Line
            type="monotone"
            dataKey="cash"
            yAxisId="right"
            stroke="#38bdf8"
            dot={false}
            strokeWidth={2}
            name="Cash Balance"
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
