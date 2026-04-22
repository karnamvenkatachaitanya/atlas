import React, { useEffect, useMemo, useState } from "react";

import DecisionLog from "./components/DecisionLog";
import EmployeeMoodChart from "./components/EmployeeMoodChart";
import EventFeed from "./components/EventFeed";
import LeaderboardPanel from "./components/LeaderboardPanel";
import RevenueCashChart from "./components/RevenueCashChart";
import RewardChart from "./components/RewardChart";
import StatsGrid from "./components/StatsGrid";
import { api } from "./services/api";
import { connectWS } from "./services/ws";

const MAX_POINTS = 120;

export default function App() {
  const [state, setState] = useState(null);
  const [mode, setMode] = useState("startup");
  const [done, setDone] = useState(false);
  const [episodeId, setEpisodeId] = useState(null);
  const [events, setEvents] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [rewards, setRewards] = useState([]);
  const [history, setHistory] = useState([]);
  const [leaderboard, setLeaderboard] = useState([]);

  useEffect(() => {
    boot();
    const ws = connectWS((data) => {
      if (data.type === "state_update") {
        const payload = data.payload;
        setState(payload.state);
        setEpisodeId(payload.episode_id ?? null);
        setDecisions((d) =>
          [{ day: payload.day, phase: payload.phase, action: payload.action }, ...d].slice(0, 30),
        );
        setHistory((h) =>
          [
            ...h,
            { step: h.length + 1, revenue: payload.state.revenue, cash: payload.state.cash_balance },
          ].slice(-MAX_POINTS),
        );
      }
      if (data.type === "market_event" && data.payload?.event) {
        setEvents((e) => [data.payload.event, ...e].slice(0, 30));
      }
      if (data.type === "reward_update") {
        setRewards((r) => [...r, { step: r.length + 1, reward: data.payload.reward }].slice(-MAX_POINTS));
      }
      if (data.type === "episode_done") {
        setDone(true);
        loadLeaderboard();
      }
    });
    return () => ws.close();
  }, []);

  async function boot() {
    const current = await api.getState();
    setState(current.data.state);
    setEpisodeId(current.data.episode_id);
    await loadLeaderboard();
  }

  async function loadLeaderboard() {
    const lb = await api.leaderboard();
    setLeaderboard(lb.data || []);
  }

  async function onReset() {
    const resetRes = await api.reset(mode);
    setState(resetRes.data.state);
    setEpisodeId(resetRes.data.episode_id);
    setEvents([]);
    setDecisions([]);
    setRewards([]);
    setHistory([]);
    setDone(false);
    await loadLeaderboard();
  }

  async function onReplay(id) {
    const replayRes = await api.replay(id);
    const steps = replayRes.data || [];
    setEvents([]);
    setDecisions([]);
    setRewards([]);
    setHistory([]);
    setDone(false);

    for (let i = 0; i < steps.length; i += 1) {
      const step = steps[i];
      // Keep replay smooth and readable for demo mode.
      await new Promise((resolve) => setTimeout(resolve, 200));
      setState(step.state);
      setDecisions((d) => [{ day: step.day, phase: step.phase, action: step.action }, ...d].slice(0, 30));
      setRewards((r) => [...r, { step: r.length + 1, reward: step.reward }].slice(-MAX_POINTS));
      setHistory((h) =>
        [...h, { step: h.length + 1, revenue: step.state.revenue, cash: step.state.cash_balance }].slice(
          -MAX_POINTS,
        ),
      );
      if (step.event?.name) {
        setEvents((e) => [step.event.name, ...e].slice(0, 30));
      }
    }
    setDone(true);
  }

  function exportCsv() {
    const rows = ["step,revenue,cash"];
    history.forEach((row) => rows.push(`${row.step},${row.revenue},${row.cash}`));
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `atlas_metrics_episode_${episodeId || "latest"}.csv`;
    anchor.click();
  }

  const mood = useMemo(() => {
    if (!state) return [];
    return [
      { name: "Engineering", mood: state.employee_morale - 2 },
      { name: "Sales", mood: state.employee_morale + 2 },
      { name: "HR", mood: state.employee_morale + 1 },
      { name: "Finance", mood: state.employee_morale - 1 },
      { name: "CS", mood: state.employee_morale + 0.5 },
    ];
  }, [state]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <h1 className="text-3xl font-bold mb-6">ATLAS: Multi-Agent Startup Management Simulation</h1>

      <div className="flex flex-wrap gap-2 mb-6">
        <select
          className="bg-slate-900 border border-slate-700 rounded px-3 py-2"
          value={mode}
          onChange={(e) => setMode(e.target.value)}
        >
          <option value="startup">Startup</option>
          <option value="crisis">Crisis</option>
          <option value="growth">Growth</option>
        </select>
        <button className="bg-indigo-600 rounded px-4 py-2" onClick={onReset}>
          Restart Simulation
        </button>
        <button className="bg-emerald-600 rounded px-4 py-2" onClick={exportCsv}>
          Download CSV
        </button>
        {episodeId && (
          <a
            className="bg-fuchsia-600 rounded px-4 py-2"
            href={api.investorReport(episodeId)}
            target="_blank"
          >
            Investor PDF
          </a>
        )}
      </div>

      <StatsGrid state={state} />

      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <RevenueCashChart data={history} />
        <RewardChart data={rewards} />
        <EmployeeMoodChart data={mood} />
        <EventFeed events={events} />
        <DecisionLog decisions={decisions} done={done} />
        <LeaderboardPanel rows={leaderboard} onReplay={onReplay} onPdf={api.investorReport} />
      </div>
    </div>
  );
}
