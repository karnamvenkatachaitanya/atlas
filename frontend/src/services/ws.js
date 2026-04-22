const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS = import.meta.env.VITE_WS_BASE_URL || `${protocol}//${window.location.host}/ws`;

export function connectWS(onEvent) {
  const ws = new WebSocket(WS);
  ws.onmessage = (msg) => onEvent(JSON.parse(msg.data));
  return ws;
}
