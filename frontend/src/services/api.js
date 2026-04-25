import axios from "axios";

const API = import.meta.env.VITE_API_BASE_URL || (window.location.origin + "/api");

export const api = {
  getState: () => axios.get(`${API}/state`),
  reset: (preset) => axios.post(`${API}/reset`, { preset }),
  leaderboard: () => axios.get(`${API}/leaderboard`),
  replay: (episodeId) => axios.get(`${API}/replay/${episodeId}`),
  investorReport: (episodeId) => `${API}/investor-report/${episodeId}`,
  toolsList: () => axios.get(`${API}/tools/list`),
  toolsCall: (tool, args) => axios.post(`${API}/tools/call`, { tool, args }),
  step: (action_idx) => axios.post(`${API}/step`, { action_idx }),
  toggleAuto: (auto_play) => axios.post(`${API}/toggle-auto`, { auto_play }),
  feedback: (data) => axios.post(`${API}/feedback`, data),
};
