# ATLAS: Enterprise Autonomous Agent Simulation

ATLAS is an industry-grade, real-time autonomous agent simulation environment. It models a complex startup ecosystem where multiple AI agents (Engineering, Sales, HR) interact, negotiate, and autonomously execute state-mutating actions based on a continuous stream of synthetic market data.

## Features

- **Real-Time Event-Driven Architecture:** Powered by `asyncio`, FastAPI, and Redis (optional). 1 simulation day advances every 1 real-time second.
- **Explainable AI (XAI):** Agents broadcast their internal thought rationales in real-time to the React Dashboard.
- **Multi-Agent Negotiation Protocols:** Departmental agents identify conflicting proposals and negotiate compromises before final execution.
- **Continuous Synthetic Data Streaming:** Hugging Face LLMs dynamically generate and inject macro-economic shifts, competitor announcements, and viral events into the environment.
- **Human Feedback (RLHF) Loop:** The UI includes direct preference capture (👍/👎) that saves directly to `data/human_preferences.json`, ready for DPO fine-tuning.
- **QLoRA 7B Support:** Pre-configured scripts for 4-bit fine-tuning of 7B parameter models via `trl`, `peft`, and `bitsandbytes`.

## Architecture Overview

```text
                  +-----------------------+
                  |  Synthetic Streamer   | (Hugging Face API)
                  +-----------+-----------+
                              | (market_events)
                              v
                      +----------------+
(state_update) <----- |   Event Broker | -----> (agent_actions)
                      +----------------+
                        ^            |
                        |            v
+------------------+    |     +------------------+
| Realtime Engine  |----+     | Async Agents     | (Hugging Face API)
| (SQLite WAL DB)  |          | (Sales, HR, Eng) |
+------------------+          +------------------+
          |
          v (WebSocket Streaming)
+------------------+
| React Dashboard  |
+------------------+
```

## Setup & Execution

### 1. Requirements
Install dependencies:
```bash
pip install -r requirements.txt
cd frontend && npm install && cd ..
```

### 2. Configuration
Create a `.env` file in the root directory (never commit this):
```env
HUGGINGFACE_API_KEY=your_hf_token_here
ATLAS_USE_REDIS=0
# REDIS_URL=redis://localhost:6379  # Uncomment if ATLAS_USE_REDIS=1
```

### 3. Run the Simulation
Run the frontend (in one terminal):
```bash
cd frontend
npm run dev
```

Run the backend engine (in another terminal):
```bash
python backend/realtime_main.py
```

Open `http://localhost:5173` to view the ATLAS Command Center.

## Training & RLHF

- **Prepare Data:** Process your UI feedback into a DPO dataset using `python training/human_feedback.py`.
- **SFT/PPO/DPO:** Run `ATLAS_USE_QLORA=1 python training/trl_ppo_rl.py` to launch the 4-bit fine-tuning pipeline for 7B models.
