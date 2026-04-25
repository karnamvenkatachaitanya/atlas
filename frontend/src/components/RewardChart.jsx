import React from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const val = payload[0].value;
    return (
      <div className="bg-slate-900 border border-slate-700 p-2 rounded shadow-lg">
        <p className="text-xs text-slate-400">Step {payload[0].payload.step}</p>
        <p className={`text-sm font-bold ${val >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
          Reward: {val.toFixed(2)}
        </p>
      </div>
    );
  }
  return null;
};

export default function RewardChart({ data }) {
  return (
    <div className="card h-72">
      <h2 className="section-title">Reward Dynamics</h2>
      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis dataKey="step" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
          <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={0} stroke="#64748b" strokeDasharray="3 3" />
          <Line
            type="monotone"
            dataKey="reward"
            stroke="#f59e0b"
            strokeWidth={2}
            dot={false}
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
