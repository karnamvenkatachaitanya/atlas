const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS = import.meta.env.VITE_WS_BASE_URL || `${protocol}//${window.location.host}/ws`;

export function connectWS(onEvent) {
  let ws = null;
  let closedByUser = false;
  let reconnectTimer = null;

  const connect = () => {
    ws = new WebSocket(WS);
    ws.onmessage = (msg) => onEvent(JSON.parse(msg.data));
    ws.onclose = () => {
      if (closedByUser) return;
      reconnectTimer = setTimeout(connect, 1000);
    };
  };

  connect();

  return {
    close: () => {
      closedByUser = true;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (ws) ws.close();
    },
  };
}
