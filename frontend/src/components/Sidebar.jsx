import React from "react";

export default function Sidebar() {
  const items = [
    { name: "Dashboard", icon: "📊", active: true },
    { name: "Overview", icon: "👁️" },
    { name: "Analytics", icon: "📈" },
    { name: "Decisions", icon: "⚖️" },
    { name: "Market", icon: "🌐" },
    { name: "Team", icon: "👥" },
    { name: "Reports", icon: "📄" },
    { name: "Settings", icon: "⚙️" },
  ];

  return (
    <div className="w-64 border-r border-slate-800 p-6 flex flex-col h-screen sticky top-0 hidden md:flex">
      <div className="flex items-center gap-3 mb-10 px-2">
        <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white font-black text-xl shadow-lg shadow-indigo-500/30">
          A
        </div>
        <div>
          <div className="font-black text-xl tracking-tighter">ATLAS</div>
          <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest -mt-1">AI CEO Dashboard</div>
        </div>
      </div>

      <nav className="flex-1 space-y-1">
        {items.map((item) => (
          <div
            key={item.name}
            className={`sidebar-item ${item.active ? "active" : ""}`}
          >
            <span className="text-xl">{item.icon}</span>
            <span className="font-semibold text-sm">{item.name}</span>
          </div>
        ))}
      </nav>

      <div className="mt-auto pt-6 border-t border-slate-800">
        <div className="flex items-center gap-3 px-2">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <div className="text-xs font-bold text-slate-400 uppercase tracking-widest">System Status</div>
        </div>
        <div className="text-[10px] text-slate-600 mt-1 px-2 font-mono">v2.4.1 (Online)</div>
      </div>
    </div>
  );
}
