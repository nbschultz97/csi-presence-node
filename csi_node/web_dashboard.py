"""Web-based real-time dashboard for Vantage CSI presence detection.

Provides a browser-based visualization that works on any platform.
No dependencies beyond the Python standard library + the existing stack.

Usage:
    python -m csi_node.web_dashboard                    # Live mode
    python -m csi_node.web_dashboard --replay data/sample_csi.b64
    python -m csi_node.web_dashboard --port 8080

Then open http://localhost:8088 in a browser.
"""
from __future__ import annotations

import argparse
import json
import time
import threading
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from collections import deque
from typing import Optional

import yaml
import numpy as np

from . import utils
from . import replay as replay_mod
from . import preprocessing
from .presence import AdaptivePresenceDetector, PresenceState
from .pose_classifier import extract_features


# In-memory state shared between pipeline thread and HTTP server
_dashboard_state = {
    "current": PresenceState().to_dict(),
    "history": {"timestamps": [], "confidence": [], "energy_ratio": [],
                "variance_ratio": [], "spectral_ratio": [], "movement_intensity": []},
    "calibration": {"calibrated": False, "baseline_energy": 0, "baseline_variance": 0},
    "log": [],
    "started": False,
    "error": None,
}
_state_lock = threading.Lock()
_detector: Optional[AdaptivePresenceDetector] = None

# Store raw JSONL entries for the log panel
_log_entries: deque = deque(maxlen=200)


DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vantage — Through-Wall Presence Detection</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e17; color: #e0e0e0; }
  .header { background: linear-gradient(135deg, #1a1f2e, #0d1117); padding: 16px 24px;
    border-bottom: 1px solid #30363d; display: flex; align-items: center; gap: 16px; }
  .header h1 { font-size: 1.4rem; font-weight: 600; }
  .header h1 span { color: #58a6ff; }
  .header .status { margin-left: auto; font-size: 0.85rem; color: #8b949e; }
  .grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; padding: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 12px; padding: 20px; }
  .card h2 { font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; color: #8b949e; margin-bottom: 12px; }
  .big-indicator { text-align: center; padding: 32px 20px; }
  .big-indicator .status-ring {
    width: 160px; height: 160px; border-radius: 50%; margin: 0 auto 16px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.3rem; font-weight: 700; transition: all 0.5s ease;
  }
  .status-clear .status-ring { border: 4px solid #238636; color: #3fb950; background: rgba(35,134,54,0.1); }
  .status-detected .status-ring { border: 4px solid #da3633; color: #f85149; background: rgba(218,54,51,0.15);
    animation: pulse 1.5s ease-in-out infinite; }
  @keyframes pulse { 0%,100% { box-shadow: 0 0 0 0 rgba(248,81,73,0.4); } 50% { box-shadow: 0 0 0 20px rgba(248,81,73,0); } }
  .confidence-bar { height: 8px; background: #21262d; border-radius: 4px; margin-top: 12px; overflow: hidden; }
  .confidence-fill { height: 100%; border-radius: 4px; transition: width 0.3s, background 0.3s; }
  .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  .metric { background: #0d1117; border-radius: 8px; padding: 12px; }
  .metric .label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }
  .metric .value { font-size: 1.4rem; font-weight: 600; font-variant-numeric: tabular-nums; }
  .chart-area { position: relative; }
  canvas { width: 100%; height: 180px; display: block; }
  .full-width { grid-column: 1 / -1; }
  .log-area { max-height: 220px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.78rem; line-height: 1.5; background: #0d1117; border-radius: 8px; padding: 8px; }
  .log-line { white-space: nowrap; }
  .log-line.present { color: #f85149; }
  .log-line.clear { color: #3fb950; }
  .movement-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    font-weight: 600; margin-left: 8px; }
  .movement-moving { background: rgba(218,54,51,0.2); color: #f85149; }
  .movement-stationary { background: rgba(56,139,253,0.2); color: #58a6ff; }
  .movement-breathing { background: rgba(163,113,247,0.2); color: #a371f7; }
  .calibration-banner { background: rgba(210,153,34,0.15); border: 1px solid #9e6a03; border-radius: 8px;
    padding: 12px 16px; margin: 16px; text-align: center; color: #d29922; display: none; }
  .btn { background: #238636; color: white; border: none; padding: 8px 16px; border-radius: 6px;
    cursor: pointer; font-size: 0.85rem; font-weight: 500; }
  .btn:hover { background: #2ea043; }
  .btn-outline { background: transparent; border: 1px solid #30363d; color: #c9d1d9; }
  .btn-outline:hover { background: #21262d; }
</style>
</head>
<body>

<div class="header">
  <h1>🎯 <span>VANTAGE</span> — Through-Wall Presence Detection</h1>
  <div class="status" id="conn-status">Connecting...</div>
</div>

<div class="calibration-banner" id="cal-banner">
  ⚠️ Not calibrated — detection uses adaptive thresholds. For best results, run calibration in an empty room.
  <button class="btn" onclick="startCalibration()" style="margin-left:12px">Calibrate Now (30s)</button>
</div>

<div class="grid">
  <!-- Main presence indicator -->
  <div class="card big-indicator status-clear" id="main-indicator">
    <h2>Presence</h2>
    <div class="status-ring" id="status-ring">CLEAR</div>
    <div style="font-size:0.85rem;color:#8b949e;margin-top:8px" id="method-label">—</div>
    <div class="confidence-bar"><div class="confidence-fill" id="conf-fill" style="width:0%;background:#238636"></div></div>
    <div style="font-size:0.8rem;color:#8b949e;margin-top:6px"><span id="conf-pct">0</span>% confidence</div>
  </div>

  <!-- Metrics -->
  <div class="card">
    <h2>Detection Metrics</h2>
    <div class="metrics">
      <div class="metric">
        <div class="label">Energy Ratio</div>
        <div class="value" id="m-energy">0.0</div>
      </div>
      <div class="metric">
        <div class="label">Variance Ratio</div>
        <div class="value" id="m-variance">0.0</div>
      </div>
      <div class="metric">
        <div class="label">Spectral Ratio</div>
        <div class="value" id="m-spectral">0.0</div>
      </div>
      <div class="metric">
        <div class="label">Packets/sec</div>
        <div class="value" id="m-pps">0</div>
      </div>
    </div>
  </div>

  <!-- Movement & Direction -->
  <div class="card">
    <h2>Movement & Position</h2>
    <div class="metrics">
      <div class="metric">
        <div class="label">Movement</div>
        <div class="value" id="m-movement">—</div>
      </div>
      <div class="metric">
        <div class="label">Intensity</div>
        <div class="value" id="m-intensity">0.0</div>
      </div>
      <div class="metric">
        <div class="label">Direction</div>
        <div class="value" id="m-direction">CENTER</div>
      </div>
      <div class="metric">
        <div class="label">Distance</div>
        <div class="value" id="m-distance">—</div>
      </div>
    </div>
  </div>

  <!-- Confidence chart -->
  <div class="card full-width chart-area">
    <h2>Detection History (60s)</h2>
    <canvas id="chart" height="180"></canvas>
  </div>

  <!-- Log -->
  <div class="card full-width">
    <h2>Event Log</h2>
    <div class="log-area" id="log-area"></div>
  </div>
</div>

<script>
const POLL_MS = 500;
let chart, ctx;
let histData = { confidence: [], energy: [], variance: [] };
const MAX_POINTS = 120;

function initChart() {
  const c = document.getElementById('chart');
  c.width = c.offsetWidth * 2;
  c.height = 360;
  ctx = c.getContext('2d');
}

function drawChart() {
  if (!ctx) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);

  // Grid
  ctx.strokeStyle = '#21262d';
  ctx.lineWidth = 1;
  for (let y = 0; y <= 1; y += 0.25) {
    const py = H - y * H;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }

  // Threshold line
  ctx.strokeStyle = '#da363388';
  ctx.setLineDash([6,4]);
  const ty = H - 0.5 * H;
  ctx.beginPath(); ctx.moveTo(0, ty); ctx.lineTo(W, ty); ctx.stroke();
  ctx.setLineDash([]);

  function drawLine(data, color) {
    if (data.length < 2) return;
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - Math.min(data[i], 1.5) / 1.5 * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  drawLine(histData.confidence, '#58a6ff');
  drawLine(histData.energy, '#3fb950');
  drawLine(histData.variance, '#d29922');

  // Legend
  ctx.font = '20px sans-serif';
  const items = [['Confidence','#58a6ff'],['Energy','#3fb950'],['Variance','#d29922']];
  let lx = 10;
  items.forEach(([label, color]) => {
    ctx.fillStyle = color;
    ctx.fillRect(lx, 8, 16, 16);
    ctx.fillStyle = '#8b949e';
    ctx.fillText(label, lx + 22, 22);
    lx += ctx.measureText(label).width + 40;
  });
}

function updateUI(data) {
  const c = data.current;
  const ind = document.getElementById('main-indicator');
  const ring = document.getElementById('status-ring');

  if (c.present) {
    ind.className = 'card big-indicator status-detected';
    ring.textContent = 'DETECTED';
  } else {
    ind.className = 'card big-indicator status-clear';
    ring.textContent = 'CLEAR';
  }

  const confPct = Math.round(c.confidence * 100);
  document.getElementById('conf-pct').textContent = confPct;
  const fill = document.getElementById('conf-fill');
  fill.style.width = confPct + '%';
  fill.style.background = c.present ? '#da3633' : '#238636';

  document.getElementById('method-label').textContent = c.method !== 'none' ? 'via ' + c.method : '—';
  document.getElementById('m-energy').textContent = c.energy_ratio.toFixed(2);
  document.getElementById('m-variance').textContent = c.variance_ratio.toFixed(2);
  document.getElementById('m-spectral').textContent = c.spectral_ratio.toFixed(2);
  document.getElementById('m-pps').textContent = Math.round(c.packets_per_sec);
  document.getElementById('m-movement').textContent = c.movement.toUpperCase() || '—';
  document.getElementById('m-intensity').textContent = c.movement_intensity.toFixed(2);
  document.getElementById('m-direction').textContent = c.direction.toUpperCase();
  document.getElementById('m-distance').textContent = c.distance_m > 0 ? c.distance_m.toFixed(1) + 'm' : '—';

  document.getElementById('cal-banner').style.display = c.calibrated ? 'none' : 'block';

  // Update history chart
  const h = data.history;
  if (h && h.confidence) {
    histData.confidence = h.confidence.slice(-MAX_POINTS);
    histData.energy = h.energy_ratio.slice(-MAX_POINTS);
    histData.variance = h.variance_ratio.slice(-MAX_POINTS);
  }
  drawChart();

  // Update log
  if (data.log && data.log.length) {
    const area = document.getElementById('log-area');
    const atBottom = area.scrollTop >= area.scrollHeight - area.clientHeight - 40;
    // Only add new entries
    const existing = area.children.length;
    for (let i = existing; i < data.log.length; i++) {
      const e = data.log[i];
      const div = document.createElement('div');
      div.className = 'log-line ' + (e.presence ? 'present' : 'clear');
      let text = e.timestamp + '  ' + (e.presence ? '🔴 PRESENT' : '🟢 CLEAR');
      text += '  conf=' + (e.confidence * 100).toFixed(0) + '%';
      if (e.movement && e.movement !== 'none') text += '  ' + e.movement;
      if (e.direction && e.direction !== 'center') text += '  dir=' + e.direction;
      div.textContent = text;
      area.appendChild(div);
    }
    if (atBottom) area.scrollTop = area.scrollHeight;
  }

  document.getElementById('conn-status').textContent =
    'Live • ' + Math.round(c.packets_per_sec) + ' pkt/s' +
    (c.calibrated ? ' • Calibrated' : ' • Uncalibrated');
}

async function poll() {
  try {
    const resp = await fetch('/api/state');
    if (resp.ok) {
      const data = await resp.json();
      updateUI(data);
    }
  } catch (e) {
    document.getElementById('conn-status').textContent = 'Disconnected';
  }
  setTimeout(poll, POLL_MS);
}

async function startCalibration() {
  try {
    await fetch('/api/calibrate', { method: 'POST' });
    document.getElementById('cal-banner').innerHTML =
      '⏳ Calibrating... please ensure room is empty. This takes ~30 seconds.';
  } catch(e) { }
}

window.addEventListener('load', () => {
  initChart();
  window.addEventListener('resize', initChart);
  poll();
});
</script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    """HTTP handler for the dashboard."""

    def log_message(self, format, *args):
        pass  # Suppress default logging

    def do_GET(self):
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode('utf-8'))
        elif self.path == '/api/state':
            with _state_lock:
                data = dict(_dashboard_state)
                data['log'] = list(_log_entries)
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Cache-Control', 'no-cache')
            self.end_headers()
            self.wfile.write(json.dumps(data).encode('utf-8'))
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path == '/api/calibrate':
            global _detector
            if _detector:
                _detector.calibrate_start()
                # Auto-finish after collecting enough samples
                threading.Timer(30.0, _finish_calibration).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"calibrating"}')
        else:
            self.send_error(404)


def _finish_calibration():
    global _detector
    if _detector:
        success = _detector.calibrate_finish()
        if success:
            cal_path = Path(__file__).resolve().parent.parent / 'data' / 'calibration.json'
            cal_path.parent.mkdir(parents=True, exist_ok=True)
            _detector.save_calibration(cal_path)
            print(f"[dashboard] Calibration saved to {cal_path}", file=sys.stderr)


def _pipeline_thread(
    replay_path: Optional[str] = None,
    log_path: Optional[str] = None,
    speed: float = 1.0,
    auto_calibrate: bool = True,
):
    """Run the CSI pipeline and feed the detector."""
    global _detector

    cfg_path = Path(__file__).resolve().parent / 'config.yaml'
    cfg = yaml.safe_load(open(cfg_path))

    _detector = AdaptivePresenceDetector(
        energy_threshold_factor=cfg.get('energy_threshold_factor', 2.5),
        variance_threshold_factor=cfg.get('variance_threshold_factor', 3.0),
        sample_rate_hz=cfg.get('sample_rate_hz', 30.0),
    )

    # Try to load existing calibration
    cal_path = Path(__file__).resolve().parent.parent / 'data' / 'calibration.json'
    if cal_path.exists():
        if _detector.load_calibration(cal_path):
            print(f"[dashboard] Loaded calibration from {cal_path}", file=sys.stderr)
    elif auto_calibrate:
        print("[dashboard] No calibration found — auto-calibrating from first 100 frames", file=sys.stderr)
        _detector.auto_calibrate(100)

    # Load baseline if available
    baseline = None
    baseline_path = cfg.get('baseline_file', '')
    if baseline_path and Path(baseline_path).exists():
        try:
            baseline = np.load(baseline_path)['mean']
        except Exception:
            pass

    buffer = deque()
    frame_count = 0

    def process_packet(pkt):
        nonlocal frame_count
        buffer.append(pkt)
        # Keep buffer bounded
        while len(buffer) > 300:
            buffer.popleft()

        csi = pkt.get('csi')
        if csi is None or csi.size == 0:
            return

        # Subtract baseline if available
        amps = csi.copy()
        if baseline is not None and baseline.shape == csi.shape:
            amps = amps - baseline

        rssi = pkt.get('rssi')
        state = _detector.update(amps, rssi=rssi, timestamp=pkt.get('ts', time.time()))

        frame_count += 1
        if frame_count % 15 == 0:  # Update dashboard at ~2Hz
            with _state_lock:
                dash = _detector.get_dashboard_data()
                _dashboard_state['current'] = dash['current']
                _dashboard_state['history'] = dash['history']
                _dashboard_state['calibration'] = dash['calibration']
                _dashboard_state['started'] = True

                if state.present or frame_count % 30 == 0:
                    _log_entries.append({
                        'timestamp': time.strftime('%H:%M:%S'),
                        'presence': state.present,
                        'confidence': state.confidence,
                        'movement': state.movement,
                        'direction': state.direction,
                        'method': state.method,
                    })

    try:
        if replay_path:
            print(f"[dashboard] Replaying {replay_path} at {speed}x", file=sys.stderr)
            for pkt in replay_mod.replay(replay_path, speed):
                process_packet(pkt)
        elif log_path:
            # Tail a log file
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler
            from .pipeline import CSILogHandler
            from threading import Event

            log_p = Path(log_path)
            if not log_p.exists():
                print(f"[dashboard] Waiting for log file: {log_p}", file=sys.stderr)
                for _ in range(50):
                    if log_p.exists():
                        break
                    time.sleep(0.2)

            stop = Event()
            handler = CSILogHandler(log_p, buffer, lambda: process_packet(buffer[-1]) if buffer else None)
            observer = Observer()
            observer.schedule(handler, str(log_p.parent), recursive=False)
            observer.start()
            try:
                while not stop.is_set():
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass
            finally:
                observer.stop()
                observer.join()
        else:
            # Default: tail the configured log file
            log_file = cfg.get('log_file', './data/csi_raw.log')
            print(f"[dashboard] Tailing {log_file}", file=sys.stderr)
            _pipeline_thread(log_path=log_file)

    except Exception as e:
        with _state_lock:
            _dashboard_state['error'] = str(e)
        print(f"[dashboard] Pipeline error: {e}", file=sys.stderr)


def run_dashboard(
    port: int = 8088,
    replay_path: Optional[str] = None,
    log_path: Optional[str] = None,
    speed: float = 1.0,
):
    """Start the web dashboard with pipeline."""
    # Start pipeline in background
    t = threading.Thread(
        target=_pipeline_thread,
        args=(replay_path, log_path, speed),
        daemon=True,
    )
    t.start()

    # Start HTTP server
    server = HTTPServer(('0.0.0.0', port), DashboardHandler)
    print(f"\n  🎯 Vantage Dashboard: http://localhost:{port}\n", file=sys.stderr)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[dashboard] Shutting down.", file=sys.stderr)
        server.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Vantage Web Dashboard")
    parser.add_argument('--port', type=int, default=8088, help='HTTP port (default: 8088)')
    parser.add_argument('--replay', type=str, default=None, help='Replay a capture file')
    parser.add_argument('--log', type=str, default=None, help='Tail a CSI log file')
    parser.add_argument('--speed', type=float, default=1.0, help='Replay speed')
    args = parser.parse_args()

    run_dashboard(
        port=args.port,
        replay_path=args.replay,
        log_path=args.log,
        speed=args.speed,
    )


if __name__ == '__main__':
    main()
