import React from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const getColor = (value) => {
  if (value < 40) return "#ef4444"; // Red
  if (value < 70) return "#eab308"; // Yellow
  return "#22c55e"; // Green
};

const CustomTooltip = ({ active, payload }) => {
  if (active && payload && payload.length) {
    const data = payload[0].payload;
    return (
      <div className="bg-slate-900 border border-slate-700 p-3 rounded-lg shadow-xl max-w-xs">
        <p className="text-slate-100 font-bold text-sm">{data.name}</p>
        <p className="text-xs mt-1" style={{ color: getColor(data.mood) }}>
          Mood: {data.mood.toFixed(1)}%
        </p>
        {data.performance && (
          <p className="text-slate-400 text-[10px] mt-0.5">
            Performance: {data.performance.toFixed(1)}%
          </p>
        )}
        {data.message && (
          <p className="text-slate-300 text-[10px] italic mt-2 leading-tight">
            "{data.message}"
          </p>
        )}
      </div>
    );
  }
  return null;
};

export default function EmployeeMoodChart({ data }) {
  return (
    <div className="card h-72">
      <h2 className="section-title">Department Insights</h2>
      <ResponsiveContainer width="100%" height="85%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
          <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
          <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} domain={[0, 100]} />
          <Tooltip content={<CustomTooltip />} cursor={{ fill: '#334155', opacity: 0.4 }} />
          <Bar dataKey="mood" radius={[4, 4, 0, 0]} isAnimationActive={true}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={getColor(entry.mood)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
