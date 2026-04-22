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

export default function RewardChart({ data }) {
  return (
    <div className="card h-72">
      <h2 className="font-semibold mb-2">Training Reward Curve (Live)</h2>
      <ResponsiveContainer width="100%" height="90%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="step" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#f59e0b"
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
