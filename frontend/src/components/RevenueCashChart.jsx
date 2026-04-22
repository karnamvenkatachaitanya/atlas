import React from "react";

import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function RevenueCashChart({ data }) {
  return (
    <div className="card h-72">
      <h2 className="font-semibold mb-2">Revenue & Cash Runway</h2>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="step" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="revenue"
            stroke="#22c55e"
            dot={false}
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
          <Line
            type="monotone"
            dataKey="cash"
            stroke="#38bdf8"
            dot={false}
            isAnimationActive
            animationDuration={280}
            animationEasing="linear"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
