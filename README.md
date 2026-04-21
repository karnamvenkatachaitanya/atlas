# ATLAS: Multi-Agent Startup Management Simulation

ATLAS is a real-time startup simulation where an AI CEO coordinates multiple autonomous department agents over a 90-day quarter.

## Stack

- Python 3.11
- FastAPI + WebSocket backend
- SQLite (SQLAlchemy)
- React + Tailwind + Recharts dashboard
- Gym/OpenEnv-compatible environment interface
- Training baseline script for before-vs-after reward curves
- Docker support

## Project Structure

```text
atlas/
 ├── backend/
 ├── frontend/
 ├── env/
 ├── agents/
 ├── training/
 ├── data/
 ├── docker/
 ├── README.md
 ├── requirements.txt
 ├── run_backend.ps1
 └── run_frontend.ps1
```

## Run Locally (Windows PowerShell)

### Backend

```powershell
.\run_backend.ps1
```

### Frontend

```powershell
.\run_frontend.ps1
```

Open:

- Frontend: http://localhost:5173
- Backend docs: http://localhost:8000/docs

## Simulation Features

- 90-day loop with morning/afternoon/evening phases
- CEO action space including hiring, product, sales, finance, and culture decisions
- Dynamic crisis and market events
- Agent reactions from:
  - Engineering Manager
  - Sales Lead
  - HR Recruiter
  - Finance Officer
  - Customer Success
- Reward emitted each step

## Live WebSocket Events

- `state_update`
- `market_event`
- `reward_update`
- `episode_done`

## API Endpoints

- `POST /api/reset` with `{ "preset": "startup|crisis|growth" }`
- `POST /api/step` with `{ "action_idx": number }`
- `GET /api/state`
- `GET /api/leaderboard`
- `GET /api/replay/{episode_id}`
- `GET /api/investor-report/{episode_id}`

## Extra Demo Features

- Scenario presets (Startup / Crisis / Growth)
- Leaderboard (episodes ranked by reward)
- Replay previous quarter
- Downloadable metrics CSV (frontend action)
- AI commentary panel
- Investor PDF report generator

## Training Script

Run:

```powershell
python training\train.py
```

Outputs:

- Random vs trained-like average reward
- Reward curve image: `training/reward_curve.png`

## OpenEnv Compliance

ATLAS now includes an explicit OpenEnv adapter:

- `AtlasOpenEnv` in `env/startup_env.py` subclasses `openenv.env.Env`
- Implements required loop methods:
  - `reset()`
  - `step(action)`
  - `render()`
- Exposes:
  - `action_space`
  - `observation_space` (as alias to state space for Gym/tooling compatibility)

Quick check:

```powershell
.\.venv\Scripts\python.exe training\check_openenv.py
```

Expected output includes:

- `OpenEnv adapter reset ok`
- several `step=...` lines
- `OpenEnv adapter check passed.`

## 3-Minute Demo Flow

1. Open dashboard and pick a scenario preset.
2. Restart simulation and show live decision/event updates.
3. Highlight revenue, runway, morale, and crisis charts.
4. Open leaderboard and replay a previous episode.
5. Download CSV and open investor PDF.
6. Run training script and show improved reward curve.
