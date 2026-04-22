import React from "react";

import {
  Bar,
  BarChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

export default function EmployeeMoodChart({ data }) {
  return (
    <div className="card h-72">
      <h2 className="font-semibold mb-2">Employee Mood Panel</h2>
      <ResponsiveContainer width="100%" height="90%">
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#334155" />
          <XAxis dataKey="name" stroke="#94a3b8" />
          <YAxis stroke="#94a3b8" />
          <Tooltip />
          <Bar dataKey="mood" fill="#a78bfa" isAnimationActive={false} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
