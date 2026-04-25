import React, { useEffect, useMemo, useRef, useState } from "react";
import { api } from "./services/api";
import { connectWS } from "./services/ws";

import StatsGrid from "./components/StatsGrid";
import RevenueCashChart from "./components/RevenueCashChart";
import RewardChart from "./components/RewardChart";
import RLTrainingPanel from "./components/RLTrainingPanel";
import EmployeeMoodChart from "./components/EmployeeMoodChart";
import EventFeed from "./components/EventFeed";
import ToolCallsPanel from "./components/ToolCallsPanel";
import NegotiationPanel from "./components/NegotiationPanel";
import DecisionLog from "./components/DecisionLog";

import ActionPanel from "./components/ActionPanel";

const MAX_POINTS = 50;

export default function App() {
  const [state, setState] = useState(null);
  const [history, setHistory] = useState([]);
  const [rewards, setRewards] = useState([]);
  const [rlEpisodeRewards, setRlEpisodeRewards] = useState([]);
  const [done, setDone] = useState(false);
  const [mode, setMode] = useState("startup");
  const [episodeId, setEpisodeId] = useState(null);
  const [leaderboard, setLeaderboard] = useState([]);
  const [events, setEvents] = useState([]);
  const [decisions, setDecisions] = useState([]);
  const [toolCalls, setToolCalls] = useState([]);
  const [negotiation, setNegotiation] = useState(null);
  const [deptReports, setDeptReports] = useState([]);
  const [inbox, setInbox] = useState("");
  const [reactions, setReactions] = useState([]);
  const [autoPlay, setAutoPlay] = useState(false);
  // Guard against processing the same step twice (manual step gets both
  // an axios response AND a WS broadcast).
  const lastStepRef = useRef(null);

  function processFrame(payload) {
    // Deduplicate: skip if we already processed this exact step
    const stepKey = `${payload.episode_id}-${payload.day}-${payload.phase}-${payload.action}`;
    if (lastStepRef.current === stepKey) return;
    lastStepRef.current = stepKey;

    setState(payload.state);
    setEpisodeId(payload.episode_id ?? null);
    setToolCalls((payload.tool_calls || []).slice(0, 10));
    setNegotiation(payload.negotiation || null);
    setDeptReports((payload.dept_reports || []).slice(0, 10));
    setInbox(payload.inbox || "");
    setReactions(payload.reactions || []);

    // Append chart data
    setHistory((h) =>
      [...h, { step: h.length + 1, revenue: payload.state.revenue, cash: payload.state.cash_balance }].slice(
        -MAX_POINTS,
      ),
    );
    setRewards((r) =>
      [...r, { step: r.length + 1, reward: payload.reward ?? 0 }].slice(-MAX_POINTS),
    );

    // Append decision log
    setDecisions((d) =>
      [{ day: payload.day, phase: payload.phase, action: payload.action, rationale: payload.rationale, state: payload.state, episode_id: payload.episode_id }, ...d].slice(0, 30),
    );

    // Append events
    if (payload.event && payload.event.name) {
      setEvents((e) => [payload.event.name, ...e].slice(0, 30));
    }

    setDone(payload.done);
  }

  useEffect(() => {
    onReset();
    loadLeaderboard();
    const ws = connectWS((data) => {
      if (data.type === "state_update") {
        processFrame(data.payload);
      }
      if (data.type === "rl_episode_complete") {
        setRlEpisodeRewards((prev) => [...prev, data.payload.total_reward].slice(-20));
      }
    });
    return () => ws.close();
  }, []);

  async function loadLeaderboard() {
    try {
      const lb = await api.leaderboard();
      setLeaderboard(lb.data || []);
    } catch (err) {
      console.warn("Leaderboard load failed:", err);
    }
  }

  async function onReset() {
    try {
      const resetRes = await api.reset(mode);
      setState(resetRes.data.state);
      setEpisodeId(resetRes.data.episode_id);
      setEvents([]);
      setDecisions([]);
      setRewards([]);
      setHistory([]);
      setToolCalls([]);
      setNegotiation(null);
      setDeptReports([]);
      setInbox("");
      setReactions([]);
      setDone(false);
      lastStepRef.current = null;
      await api.toggleAuto(autoPlay);
      await loadLeaderboard();
    } catch (err) {
      console.error("Reset failed:", err);
    }
  }

  async function toggleAutoPlay() {
    const next = !autoPlay;
    setAutoPlay(next);
    await api.toggleAuto(next);
  }

  async function handleManualAction(actionIdx) {
    if (autoPlay || done) return;
    try {
      // The backend broadcasts the frame over WS, so processFrame
      // will be called via the WS listener. No need to process here.
      await api.step(actionIdx);
    } catch (err) {
      console.error("Manual action failed:", err);
    }
  }

  async function onReplay(id) {
    try {
      const replayRes = await api.replay(id);
      const steps = replayRes.data || [];
      setEvents([]);
      setDecisions([]);
      setRewards([]);
      setHistory([]);
      setDone(false);
      lastStepRef.current = null;

      for (let i = 0; i < steps.length; i += 1) {
        const step = steps[i];
        await new Promise((resolve) => setTimeout(resolve, 150));
        setState(step.state);
        setReactions(step.reactions || []);
        setDecisions((d) => [{ day: step.day, phase: step.phase, action: step.action }, ...d].slice(0, 30));
        setRewards((r) => [...r, { step: r.length + 1, reward: step.reward }].slice(-MAX_POINTS));
        setHistory((h) =>
          [...h, { step: h.length + 1, revenue: step.state.revenue, cash: step.state.cash_balance }].slice(
            -MAX_POINTS,
          ),
        );
        if (step.event && step.event.name) {
          setEvents((e) => [step.event.name, ...e].slice(0, 30));
        }
      }
      setDone(true);
    } catch (err) {
      console.error("Replay failed:", err);
    }
  }

  const mood = useMemo(() => {
    if (reactions && reactions.length > 0) {
      return reactions.map((r) => {
        // r can be a dict {role, happiness, performance, message} or a plain string
        if (typeof r === "string") {
          return { name: r.replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()), mood: 70, message: r, performance: 0 };
        }
        const role = r?.role || "Unknown";
        return {
          name: String(role).replace(/_/g, " ").replace(/\b\w/g, (l) => l.toUpperCase()),
          mood: r?.happiness ?? 70,
          message: r?.message || "",
          performance: r?.performance ?? 0,
        };
      });
    }
    if (!state) return [];
    const m = state.employee_morale ?? 70;
    return [
      { name: "Engineering", mood: Math.min(100, Math.max(0, m - 2)) },
      { name: "Sales", mood: Math.min(100, Math.max(0, m + 3)) },
      { name: "HR", mood: Math.min(100, Math.max(0, m + 1)) },
      { name: "Finance", mood: Math.min(100, Math.max(0, m - 4)) },
      { name: "CS", mood: Math.min(100, Math.max(0, m + 0.5)) },
    ];
  }, [state, reactions]);

  function exportCsv() {
    const rows = ["step,revenue,cash"];
    history.forEach((row) => rows.push(`${row.step},${row.revenue},${row.cash}`));
    const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const anchor = document.createElement("a");
    anchor.href = URL.createObjectURL(blob);
    anchor.download = `atlas_metrics_episode_${episodeId || "latest"}.csv`;
    anchor.click();
  }

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 p-6">
      <div className="hero-gradient rounded-3xl p-6 md:p-8 border border-slate-700/70 shadow-2xl">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <div className="section-title">OpenEnv Hackathon Demo</div>
            <h1 className="text-3xl md:text-4xl font-black leading-tight">
              ATLAS Command Center
            </h1>
            <p className="text-slate-300 mt-2 max-w-3xl">
              High-fidelity startup simulation showcasing RL training and multi-agent alignment.
            </p>
          </div>
          <div className="glass-panel px-4 py-3 min-w-[220px]">
            <div className="text-xs text-slate-400 uppercase tracking-wider">Episode</div>
            <div className="text-lg font-bold text-white">{episodeId || "Not started"}</div>
            <div className={`mt-2 text-sm font-semibold ${done ? "text-emerald-300" : "text-amber-300"}`}>
              {done ? "Completed" : "Running"}
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel p-4 mt-5">
        <div className="flex flex-wrap gap-2">
          <select
            className="bg-slate-900 border border-slate-700 rounded-xl px-3 py-2"
            value={mode}
            onChange={(e) => setMode(e.target.value)}
          >
            <option value="startup">Startup</option>
            <option value="crisis">Crisis</option>
            <option value="growth">Growth</option>
            <option value="procedural">Procedural</option>
          </select>
          <button className="bg-indigo-600 rounded-xl px-4 py-2 font-semibold hover:bg-indigo-500" onClick={onReset}>
            Restart Simulation
          </button>
          <button className="bg-emerald-600 rounded-xl px-4 py-2 hover:bg-emerald-500 font-semibold" onClick={exportCsv}>
            Download CSV
          </button>
          <button
            className={`rounded-xl px-4 py-2 font-bold ${autoPlay ? "bg-amber-600 hover:bg-amber-500 text-white" : "bg-slate-700 hover:bg-slate-600 text-slate-300"}`}
            onClick={toggleAutoPlay}
          >
            {autoPlay ? "AI CEO (Auto)" : "Manual Mode"}
          </button>
          {episodeId && (
            <a
              className="bg-fuchsia-600 rounded-xl px-4 py-2 font-semibold hover:bg-fuchsia-500"
              href={api.investorReport(episodeId)}
              target="_blank"
            >
              Investor PDF
            </a>
          )}
        </div>
      </div>

      <div className="mt-4">
        <StatsGrid state={state} />
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-4">
        <RevenueCashChart data={history} state={state} />
        <RewardChart data={rewards} />
        <RLTrainingPanel episodeRewards={rlEpisodeRewards} />
        <EmployeeMoodChart data={mood} />

        <div className="card h-72 overflow-hidden flex flex-col">
          <h2 className="font-semibold text-rose-300 mb-2">⚠️ Market Events</h2>
          <EventFeed events={events} />
        </div>

        <ToolCallsPanel toolCalls={toolCalls} />

        <div className="card flex flex-col h-72">
          <h2 className="text-xl font-bold mb-3 text-indigo-300">Department Inbox</h2>
          <div className="flex-1 overflow-y-auto space-y-2 text-sm font-mono whitespace-pre-wrap">
            {inbox ? <span className="text-slate-300">{inbox}</span> : <span className="text-slate-500 italic">No messages.</span>}
          </div>
        </div>

        <NegotiationPanel negotiation={negotiation} deptReports={deptReports} />
        <DecisionLog decisions={decisions} done={done} />
      </div>
      <ActionPanel disabled={done || autoPlay} onAction={handleManualAction} />
    </div>
  );
}
