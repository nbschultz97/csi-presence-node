"""Web-based real-time dashboard for Vantage CSI presence detection.

Provides a browser-based visualization that works on any platform.
Uses Server-Sent Events (SSE) for real-time streaming updates.
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
    "simulate": False,
    "recording": False,
    "record_count": 0,
    "calibration_progress": 0.0,
    "zone_heatmap": [],
}
_state_lock = threading.Lock()
_detector: Optional[AdaptivePresenceDetector] = None

# Store raw JSONL entries for the log panel
_log_entries: deque = deque(maxlen=200)

# SSE clients
_sse_clients: list = []
_sse_lock = threading.Lock()

# Recording state
_recording = False
_record_file = None
_record_count = 0
_record_label = "unknown"


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
  .header .badges { margin-left: auto; display: flex; gap: 8px; align-items: center; }
  .badge { font-size: 0.75rem; padding: 3px 10px; border-radius: 12px; font-weight: 600; }
  .badge-live { background: rgba(35,134,54,0.2); color: #3fb950; border: 1px solid #238636; }
  .badge-sim { background: rgba(88,166,255,0.1); color: #58a6ff; border: 1px solid #1f6feb; }
  .badge-rec { background: rgba(218,54,51,0.2); color: #f85149; border: 1px solid #da3633; animation: blink 1s infinite; }
  @keyframes blink { 50% { opacity: 0.5; } }
  .header .status { font-size: 0.8rem; color: #8b949e; }
  .toolbar { background: #161b22; border-bottom: 1px solid #21262d; padding: 8px 24px; display: flex; gap: 8px; align-items: center; }
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
  .two-thirds { grid-column: span 2; }
  .log-area { max-height: 220px; overflow-y: auto; font-family: 'Cascadia Code', 'Fira Code', monospace;
    font-size: 0.78rem; line-height: 1.5; background: #0d1117; border-radius: 8px; padding: 8px; }
  .log-line { white-space: nowrap; }
  .log-line.present { color: #f85149; }
  .log-line.clear { color: #3fb950; }
  .movement-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; margin-left: 8px; }
  .movement-moving { background: rgba(218,54,51,0.2); color: #f85149; }
  .movement-stationary { background: rgba(56,139,253,0.2); color: #58a6ff; }
  .movement-breathing { background: rgba(163,113,247,0.2); color: #a371f7; }
  .calibration-banner { background: rgba(210,153,34,0.15); border: 1px solid #9e6a03; border-radius: 8px;
    padding: 12px 16px; margin: 16px; text-align: center; color: #d29922; display: none; }
  .cal-progress { height: 4px; background: #21262d; border-radius: 2px; margin-top: 8px; overflow: hidden; }
  .cal-progress-fill { height: 100%; background: #d29922; border-radius: 2px; transition: width 0.5s; width: 0%; }
  .btn { background: #238636; color: white; border: none; padding: 6px 14px; border-radius: 6px;
    cursor: pointer; font-size: 0.8rem; font-weight: 500; }
  .btn:hover { background: #2ea043; }
  .btn-outline { background: transparent; border: 1px solid #30363d; color: #c9d1d9; }
  .btn-outline:hover { background: #21262d; }
  .btn-danger { background: #da3633; }
  .btn-danger:hover { background: #f85149; }
  .btn-sm { padding: 4px 10px; font-size: 0.75rem; }
  /* Subcarrier heatmap */
  .heatmap-container { display: flex; gap: 1px; align-items: flex-end; height: 80px; background: #0d1117; border-radius: 8px; padding: 8px; overflow: hidden; }
  .heatmap-bar { flex: 1; min-width: 2px; border-radius: 1px 1px 0 0; transition: height 0.3s, background 0.3s; }
  /* Zone visualization */
  .zone-viz { position: relative; background: #0d1117; border-radius: 8px; height: 200px; overflow: hidden; }
  .zone-sensor { position: absolute; bottom: 10px; left: 50%; transform: translateX(-50%);
    width: 36px; height: 36px; background: #238636; border-radius: 50%; display: flex; align-items: center;
    justify-content: center; font-size: 14px; z-index: 2; border: 2px solid #3fb950; }
  .zone-wall { position: absolute; top: 50%; left: 0; right: 0; height: 4px; background: #8b949e;
    transform: translateY(-50%); z-index: 1; }
  .zone-wall-label { position: absolute; top: 50%; right: 8px; transform: translateY(-50%);
    font-size: 0.65rem; color: #8b949e; z-index: 2; background: #0d1117; padding: 0 4px; }
  .zone-target { position: absolute; width: 28px; height: 28px; border-radius: 50%; display: flex;
    align-items: center; justify-content: center; font-size: 12px; transition: all 0.5s ease;
    z-index: 3; }
  .zone-target.active { background: rgba(218,54,51,0.3); border: 2px solid #f85149; }
  .zone-target.inactive { background: rgba(33,38,45,0.5); border: 2px dashed #30363d; opacity: 0.3; }
  .zone-cone { position: absolute; bottom: 38px; left: 50%; transform-origin: bottom center;
    width: 0; height: 0; opacity: 0.1; z-index: 0; }
  /* Uptime counter */
  .uptime { font-variant-numeric: tabular-nums; }
</style>
</head>
<body>

<div class="header">
  <h1>🎯 <span>VANTAGE</span> — Through-Wall Presence Detection</h1>
  <div class="badges">
    <span class="badge badge-live" id="badge-live" style="display:none">● LIVE</span>
    <span class="badge badge-sim" id="badge-sim" style="display:none">◉ SIMULATION</span>
    <span class="badge badge-rec" id="badge-rec" style="display:none">⏺ REC</span>
  </div>
  <div class="status" id="conn-status">Connecting...</div>
</div>

<div class="toolbar" id="toolbar">
  <button class="btn btn-sm" onclick="startCalibration()">📐 Calibrate</button>
  <button class="btn btn-sm btn-outline" onclick="toggleRecording()" id="rec-btn">⏺ Record Data</button>
  <select class="btn btn-sm btn-outline" onchange="setProfile(this.value)" id="profile-select" style="appearance:auto">
    <option value="default">Profile: Default</option>
    <option value="through_wall">Profile: Through-Wall</option>
    <option value="same_room">Profile: Same Room</option>
    <option value="high_sensitivity">Profile: High Sensitivity</option>
  </select>
  <span style="margin-left:auto;font-size:0.75rem;color:#8b949e" class="uptime" id="uptime">00:00:00</span>
</div>

<div class="calibration-banner" id="cal-banner">
  ⚠️ Not calibrated — detection uses adaptive thresholds. For best results, run calibration in an empty room.
  <div class="cal-progress" id="cal-progress" style="display:none"><div class="cal-progress-fill" id="cal-fill"></div></div>
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

  <!-- Zone visualization -->
  <div class="card">
    <h2>Spatial View</h2>
    <div class="zone-viz" id="zone-viz">
      <div class="zone-wall"></div>
      <div class="zone-wall-label">WALL</div>
      <div class="zone-sensor">📡</div>
      <div class="zone-target inactive" id="zone-target" style="top:25%;left:50%;transform:translate(-50%,-50%)">🚶</div>
    </div>
  </div>

  <!-- Subcarrier heatmap -->
  <div class="card two-thirds">
    <h2>Subcarrier Energy Heatmap</h2>
    <div class="heatmap-container" id="heatmap"></div>
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
let ctx;
let histData = { confidence: [], energy: [], variance: [], spectral: [] };
const MAX_POINTS = 120;
let startTime = Date.now();
let evtSource = null;
let heatmapBars = [];
let logCount = 0;

function initChart() {
  const c = document.getElementById('chart');
  c.width = c.offsetWidth * 2;
  c.height = 360;
  ctx = c.getContext('2d');
}

function initHeatmap() {
  const container = document.getElementById('heatmap');
  container.innerHTML = '';
  heatmapBars = [];
  for (let i = 0; i < 52; i++) {
    const bar = document.createElement('div');
    bar.className = 'heatmap-bar';
    bar.style.height = '4px';
    bar.style.background = '#21262d';
    container.appendChild(bar);
    heatmapBars.push(bar);
  }
}

function updateHeatmap(zones) {
  if (!zones || !zones.length) return;
  const n = Math.min(zones.length, heatmapBars.length);
  for (let i = 0; i < n; i++) {
    const v = Math.min(zones[i], 1.0);
    const h = Math.max(4, v * 72);
    heatmapBars[i].style.height = h + 'px';
    if (v > 0.7) heatmapBars[i].style.background = '#f85149';
    else if (v > 0.4) heatmapBars[i].style.background = '#d29922';
    else if (v > 0.15) heatmapBars[i].style.background = '#58a6ff';
    else heatmapBars[i].style.background = '#21262d';
  }
}

function drawChart() {
  if (!ctx) return;
  const W = ctx.canvas.width, H = ctx.canvas.height;
  ctx.clearRect(0, 0, W, H);
  ctx.strokeStyle = '#21262d'; ctx.lineWidth = 1;
  for (let y = 0; y <= 1; y += 0.25) {
    const py = H - y * H;
    ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(W, py); ctx.stroke();
  }
  ctx.strokeStyle = '#da363388'; ctx.setLineDash([6,4]);
  ctx.beginPath(); ctx.moveTo(0, H*0.5); ctx.lineTo(W, H*0.5); ctx.stroke();
  ctx.setLineDash([]);

  function drawLine(data, color, alpha) {
    if (data.length < 2) return;
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.globalAlpha = alpha || 1;
    ctx.beginPath();
    for (let i = 0; i < data.length; i++) {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - Math.min(data[i], 1.5) / 1.5 * H;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke(); ctx.globalAlpha = 1;
  }
  // Fill under confidence
  if (histData.confidence.length > 1) {
    ctx.fillStyle = 'rgba(88,166,255,0.08)';
    ctx.beginPath();
    ctx.moveTo(0, H);
    for (let i = 0; i < histData.confidence.length; i++) {
      const x = (i / (MAX_POINTS - 1)) * W;
      const y = H - Math.min(histData.confidence[i], 1.5) / 1.5 * H;
      ctx.lineTo(x, y);
    }
    ctx.lineTo(((histData.confidence.length-1) / (MAX_POINTS-1)) * W, H);
    ctx.fill();
  }
  drawLine(histData.confidence, '#58a6ff');
  drawLine(histData.energy, '#3fb950', 0.7);
  drawLine(histData.variance, '#d29922', 0.7);
  drawLine(histData.spectral, '#a371f7', 0.5);

  ctx.font = '20px sans-serif';
  const items = [['Confidence','#58a6ff'],['Energy','#3fb950'],['Variance','#d29922'],['Spectral','#a371f7']];
  let lx = 10;
  items.forEach(([label, color]) => {
    ctx.fillStyle = color; ctx.fillRect(lx, 8, 16, 16);
    ctx.fillStyle = '#8b949e'; ctx.fillText(label, lx + 22, 22);
    lx += ctx.measureText(label).width + 40;
  });
}

function updateZoneViz(c) {
  const tgt = document.getElementById('zone-target');
  if (c.present) {
    tgt.className = 'zone-target active';
    let leftPct = 50;
    if (c.direction === 'left') leftPct = 30;
    else if (c.direction === 'right') leftPct = 70;
    let topPct = 25 - Math.min(c.confidence, 1) * 5;
    tgt.style.left = leftPct + '%';
    tgt.style.top = topPct + '%';
    tgt.textContent = c.movement === 'moving' ? '🏃' : c.movement === 'breathing' ? '🧘' : '🚶';
  } else {
    tgt.className = 'zone-target inactive';
    tgt.textContent = '🚶';
    tgt.style.left = '50%'; tgt.style.top = '25%';
  }
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
  document.getElementById('m-movement').textContent = (c.movement || 'none').toUpperCase();
  document.getElementById('m-intensity').textContent = c.movement_intensity.toFixed(2);
  document.getElementById('m-direction').textContent = (c.direction || 'center').toUpperCase();
  document.getElementById('m-distance').textContent = c.distance_m > 0 ? c.distance_m.toFixed(1) + 'm' : '—';

  document.getElementById('cal-banner').style.display = c.calibrated ? 'none' : 'block';
  updateZoneViz(c);

  // Badges
  document.getElementById('badge-sim').style.display = data.simulate ? '' : 'none';
  document.getElementById('badge-live').style.display = data.simulate ? 'none' : '';
  document.getElementById('badge-rec').style.display = data.recording ? '' : 'none';

  // Calibration progress
  if (data.calibration_progress > 0 && data.calibration_progress < 1) {
    document.getElementById('cal-progress').style.display = '';
    document.getElementById('cal-fill').style.width = (data.calibration_progress * 100) + '%';
    document.getElementById('cal-banner').style.display = '';
    document.getElementById('cal-banner').firstChild.textContent =
      '⏳ Calibrating... ' + Math.round(data.calibration_progress * 100) + '% — ensure room is empty';
  } else {
    document.getElementById('cal-progress').style.display = 'none';
  }

  // Heatmap
  if (data.zone_heatmap) updateHeatmap(data.zone_heatmap);

  // History
  const h = data.history;
  if (h && h.confidence) {
    histData.confidence = h.confidence.slice(-MAX_POINTS);
    histData.energy = h.energy_ratio.slice(-MAX_POINTS);
    histData.variance = h.variance_ratio.slice(-MAX_POINTS);
    histData.spectral = (h.spectral_ratio || []).slice(-MAX_POINTS);
  }
  drawChart();

  document.getElementById('conn-status').textContent =
    Math.round(c.packets_per_sec) + ' pkt/s' +
    (c.calibrated ? ' • Calibrated' : '');
}

function addLogEntry(e) {
  const area = document.getElementById('log-area');
  const atBottom = area.scrollTop >= area.scrollHeight - area.clientHeight - 40;
  const div = document.createElement('div');
  div.className = 'log-line ' + (e.presence ? 'present' : 'clear');
  let text = e.timestamp + '  ' + (e.presence ? '🔴 PRESENT' : '🟢 CLEAR');
  text += '  conf=' + (e.confidence * 100).toFixed(0) + '%';
  if (e.movement && e.movement !== 'none') text += '  [' + e.movement + ']';
  if (e.method && e.method !== 'none') text += '  via:' + e.method;
  div.textContent = text;
  area.appendChild(div);
  // Trim log area to 200 entries
  while (area.children.length > 200) area.removeChild(area.firstChild);
  if (atBottom) area.scrollTop = area.scrollHeight;
}

function connectSSE() {
  if (evtSource) evtSource.close();
  evtSource = new EventSource('/api/stream');
  evtSource.onmessage = function(e) {
    try {
      const data = JSON.parse(e.data);
      updateUI(data);
    } catch(err) {}
  };
  evtSource.addEventListener('log', function(e) {
    try { addLogEntry(JSON.parse(e.data)); } catch(err) {}
  });
  evtSource.onerror = function() {
    document.getElementById('conn-status').textContent = 'Reconnecting...';
    setTimeout(() => { if (evtSource.readyState === 2) connectSSE(); }, 2000);
  };
  evtSource.onopen = function() {
    document.getElementById('conn-status').textContent = 'Connected';
  };
}

// Fallback to polling if SSE not available
async function pollFallback() {
  try {
    const resp = await fetch('/api/state');
    if (resp.ok) updateUI(await resp.json());
  } catch(e) {}
  setTimeout(pollFallback, 500);
}

async function startCalibration() {
  try { await fetch('/api/calibrate', { method: 'POST' }); } catch(e) {}
}

async function toggleRecording() {
  try {
    const resp = await fetch('/api/record', { method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({label: prompt('Label for recording (e.g. empty, present, walking):', 'present') || 'unknown'})
    });
    const d = await resp.json();
    document.getElementById('rec-btn').textContent = d.recording ? '⏹ Stop Recording' : '⏺ Record Data';
  } catch(e) {}
}

async function setProfile(p) {
  try { await fetch('/api/profile', { method: 'POST',
    headers: {'Content-Type':'application/json'},
    body: JSON.stringify({profile: p})
  }); } catch(e) {}
}

function updateUptime() {
  const s = Math.floor((Date.now() - startTime) / 1000);
  const h = String(Math.floor(s/3600)).padStart(2,'0');
  const m = String(Math.floor((s%3600)/60)).padStart(2,'0');
  const ss = String(s%60).padStart(2,'0');
  document.getElementById('uptime').textContent = h+':'+m+':'+ss;
}

window.addEventListener('load', () => {
  initChart(); initHeatmap();
  window.addEventListener('resize', () => { initChart(); initHeatmap(); });
  // Try SSE first, fall back to polling
  if (typeof EventSource !== 'undefined') connectSSE();
  else pollFallback();
  setInterval(updateUptime, 1000);
  setInterval(drawChart, 500);
});
</script>
</body>
</html>"""


DETECTION_PROFILES = {
    "default": {
        "energy_threshold_factor": 2.5,
        "variance_threshold_factor": 3.0,
        "spectral_threshold_factor": 2.0,
        "presence_threshold": 0.5,
        "ema_alpha": 0.3,
    },
    "through_wall": {
        "energy_threshold_factor": 1.8,
        "variance_threshold_factor": 2.0,
        "spectral_threshold_factor": 1.5,
        "presence_threshold": 0.35,
        "ema_alpha": 0.25,
    },
    "same_room": {
        "energy_threshold_factor": 3.5,
        "variance_threshold_factor": 4.0,
        "spectral_threshold_factor": 3.0,
        "presence_threshold": 0.6,
        "ema_alpha": 0.35,
    },
    "high_sensitivity": {
        "energy_threshold_factor": 1.5,
        "variance_threshold_factor": 1.5,
        "spectral_threshold_factor": 1.2,
        "presence_threshold": 0.3,
        "ema_alpha": 0.2,
    },
}


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
        elif self.path == '/api/stream':
            self._handle_sse()
        else:
            self.send_error(404)

    def _handle_sse(self):
        """Server-Sent Events endpoint for real-time streaming."""
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'keep-alive')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        try:
            while True:
                with _state_lock:
                    data = dict(_dashboard_state)
                payload = f"data: {json.dumps(data)}\n\n"
                self.wfile.write(payload.encode('utf-8'))
                self.wfile.flush()
                time.sleep(0.5)
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    def do_POST(self):
        if self.path == '/api/calibrate':
            global _detector
            if _detector:
                _detector.calibrate_start()
                with _state_lock:
                    _dashboard_state['calibration_progress'] = 0.01
                threading.Timer(30.0, _finish_calibration).start()
                # Progress updates
                for i in range(1, 30):
                    threading.Timer(float(i), _update_cal_progress, args=(i/30.0,)).start()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"calibrating"}')
        elif self.path == '/api/record':
            self._handle_record()
        elif self.path == '/api/profile':
            self._handle_profile()
        else:
            self.send_error(404)

    def _handle_record(self):
        """Toggle data recording for training data collection."""
        global _recording, _record_file, _record_count, _record_label
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}

        if _recording:
            # Stop recording
            _recording = False
            if _record_file:
                _record_file.close()
                _record_file = None
            with _state_lock:
                _dashboard_state['recording'] = False
            result = {"recording": False, "frames_saved": _record_count}
        else:
            # Start recording
            _record_label = body.get('label', 'unknown')
            ts = time.strftime('%Y%m%d_%H%M%S')
            rec_dir = Path(__file__).resolve().parent.parent / 'data' / 'recordings'
            rec_dir.mkdir(parents=True, exist_ok=True)
            rec_path = rec_dir / f'{_record_label}_{ts}.jsonl'
            _record_file = open(rec_path, 'w')
            _record_count = 0
            _recording = True
            with _state_lock:
                _dashboard_state['recording'] = True
                _dashboard_state['record_count'] = 0
            result = {"recording": True, "path": str(rec_path)}
            print(f"[dashboard] Recording to {rec_path}", file=sys.stderr)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(result).encode('utf-8'))

    def _handle_profile(self):
        """Switch detection profile."""
        global _detector
        content_len = int(self.headers.get('Content-Length', 0))
        body = json.loads(self.rfile.read(content_len)) if content_len else {}
        profile_name = body.get('profile', 'default')
        profile = DETECTION_PROFILES.get(profile_name, DETECTION_PROFILES['default'])

        if _detector:
            _detector.energy_threshold_factor = profile['energy_threshold_factor']
            _detector.variance_threshold_factor = profile['variance_threshold_factor']
            _detector.spectral_threshold_factor = profile['spectral_threshold_factor']
            _detector.presence_threshold = profile['presence_threshold']
            _detector.ema_alpha = profile['ema_alpha']
            print(f"[dashboard] Switched to profile: {profile_name}", file=sys.stderr)

        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({"profile": profile_name}).encode('utf-8'))


def _update_cal_progress(progress: float):
    with _state_lock:
        _dashboard_state['calibration_progress'] = progress


def _finish_calibration():
    global _detector
    if _detector:
        success = _detector.calibrate_finish()
        with _state_lock:
            _dashboard_state['calibration_progress'] = 0.0
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
    simulate: bool = False,
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
        global _record_count
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

        # Record raw data if recording is active
        if _recording and _record_file:
            try:
                rec_entry = {
                    'ts': pkt.get('ts', time.time()),
                    'csi': csi.tolist() if hasattr(csi, 'tolist') else list(csi),
                    'rssi': rssi,
                    'label': _record_label,
                    'presence': state.present,
                    'confidence': state.confidence,
                }
                _record_file.write(json.dumps(rec_entry) + '\n')
                _record_count += 1
            except Exception:
                pass

        frame_count += 1
        if frame_count % 15 == 0:  # Update dashboard at ~2Hz
            # Compute per-subcarrier energy for heatmap
            zone_heatmap = []
            if len(buffer) >= 5:
                recent = np.array([p['csi'].flatten() for p in list(buffer)[-30:]
                                   if p.get('csi') is not None and p['csi'].size > 0])
                if recent.size > 0:
                    sub_var = np.var(recent, axis=0)
                    max_var = np.max(sub_var) if np.max(sub_var) > 0 else 1.0
                    zone_heatmap = (sub_var / max_var).tolist()

            with _state_lock:
                dash = _detector.get_dashboard_data()
                _dashboard_state['current'] = dash['current']
                _dashboard_state['history'] = dash['history']
                _dashboard_state['calibration'] = dash['calibration']
                _dashboard_state['started'] = True
                _dashboard_state['zone_heatmap'] = zone_heatmap
                _dashboard_state['record_count'] = _record_count

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
        if simulate:
            from .simulator import CSISimulator
            print("[dashboard] Running in SIMULATION mode — synthetic CSI data", file=sys.stderr)
            with _state_lock:
                _dashboard_state['simulate'] = True
            sim = CSISimulator()
            for pkt in sim.stream(loop=True, realtime=True):
                process_packet(pkt)
        elif replay_path:
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
    simulate: bool = False,
):
    """Start the web dashboard with pipeline."""
    # Start pipeline in background
    t = threading.Thread(
        target=_pipeline_thread,
        args=(replay_path, log_path, speed, True, simulate),
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
    parser.add_argument('--simulate', action='store_true', help='Use synthetic CSI data (no hardware needed)')
    args = parser.parse_args()

    run_dashboard(
        port=args.port,
        replay_path=args.replay,
        log_path=args.log,
        speed=args.speed,
        simulate=args.simulate,
    )


if __name__ == '__main__':
    main()
