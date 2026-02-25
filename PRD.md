# Product Requirements Document

> **Vantage** — Passive WiFi CSI Through-Wall Human Detection System

**Version:** 1.0  
**Status:** MVP Development  
**Classification:** UNCLASSIFIED

---

## 1. Product Vision

Vantage is a passive WiFi Channel State Information (CSI) system that detects human presence through walls — at **10× lower cost** than existing radar alternatives.

| Capability | Vantage | Lumineye Lux / Lynx | Origin Wireless |
|---|---|---|---|
| **Price point** | $5–10K | $60–85K | N/A (consumer) |
| **Sensing method** | Passive WiFi CSI | Active UWB radar | Active WiFi sensing |
| **RF emission** | Zero (covert) | Active transmitter | Active transmitter |
| **Target market** | Military / LE / FR | Military / LE | Smart home |
| **Networked / mesh** | Yes (roadmap) | No | No |
| **ATAK integration** | Native CoT | Limited | None |

Vantage exploits ambient WiFi signals — it transmits nothing, making it undetectable by RF scanners.

---

## 2. Target Users

| Segment | Use Case |
|---|---|
| **Military — SOF / Infantry** | Compound clearing, building assessment, perimeter security |
| **Law Enforcement — SWAT / Tactical** | Hostage rescue, warrant service, active shooter |
| **First Responders** | Search and rescue, disaster response, firefighter safety |

---

## 3. Core Functional Requirements

### 3.1 Presence Detection
- Detect human presence through interior walls (drywall, wood-frame)
- Binary output: occupied / unoccupied per zone
- AdaptivePresenceDetector with multi-method fusion (energy, variance, spectral)

### 3.2 Direction Estimation
- Classify target bearing as Left / Center / Right relative to sensor
- Multi-chain RSSI comparison for coarse angular resolution

### 3.3 Distance Estimation
- Estimate range to target using path-loss model with per-environment calibration
- Configurable path-loss exponent and TX power parameters

### 3.4 Pose Classification
- Classify detected subject as standing, sitting, walking, or prone
- RandomForest classifier with 14-feature extraction pipeline

### 3.5 Through-Wall Operation
- Operate through standard interior construction materials
- Environment profile manager for site-specific calibration

---

## 4. Non-Functional Requirements

| Requirement | Target |
|---|---|
| **Processing latency** | < 100 ms end-to-end |
| **Compute** | Edge-only — no cloud dependency |
| **RF signature** | Zero emission (passive receive only) |
| **Covertness** | Undetectable by RF scanners or spectrum analyzers |
| **Reliability** | Crash-safe logging, automatic WiFi recovery |
| **Portability** | Linux (Ubuntu 22.04+), Python 3.10+ |
| **Testability** | Demo/simulation mode without hardware |

---

## 5. Integration Requirements

### 5.1 ATAK / Cursor-on-Target
- Real-time CoT messages for ATAK overlay
- Detection events with bearing and range data
- Compatible with TAK Server and mesh networking

### 5.2 Data Streaming
- UDP broadcast for integration with external C2 systems
- JSON-formatted event stream
- Server-Sent Events (SSE) for web clients

### 5.3 Web Dashboard
- Real-time visualization with zone map and subcarrier heatmap
- Detection profile management
- Data recording and replay capability

### 5.4 API
- JSON API for programmatic integration
- Offline replay from recorded CSI data

---

## 6. Hardware Requirements

### Minimum Configuration
- **WiFi Adapter:** Intel AX210 (WiFi 6E) with CSI-capable firmware
- **Compute:** Any x86_64 Linux system (laptop, NUC, SBC)
- **OS:** Ubuntu 22.04+ or equivalent
- **Dependencies:** FeitCSI kernel module for CSI extraction

### Target Form Factors

| Form Factor | Description | Use Case |
|---|---|---|
| **Tripod Node** | AX210 + NUC/Pi on tripod mount | Fixed perimeter, building assessment |
| **Handheld** | Ruggedized tablet or handheld compute | Dismounted patrol, room clearing |
| **UxS Payload** | Miniaturized for drone/robot integration | Aerial/ground robotic ISR |

---

## 7. MVP Success Criteria

The minimum viable product demonstration must achieve:

- [x] **Presence detection** through one standard interior wall (drywall/wood-frame)
- [x] **Real-time web dashboard** with live detection visualization
- [x] **ATAK integration** with Cursor-on-Target output
- [x] **Demo mode** functional without CSI hardware
- [x] **Direction estimation** (Left / Center / Right)
- [x] **Distance estimation** with calibration workflow

---

## 8. Out of Scope (MVP)

The following are explicitly deferred beyond the MVP milestone:

- Multi-person tracking and counting
- Centimeter-accurate ranging
- Metal or reinforced concrete wall penetration
- Multi-node mesh fusion
- Production packaging / hardened enclosure
- FIPS / MIL-STD certification

---

## 9. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Synthetic-only training data | Reduced real-world accuracy | Phase 1 real-data collection campaign |
| Wall material variability | Detection range fluctuation | Environment profile calibration system |
| WiFi environment dependency | No ambient WiFi = no sensing | Document operational requirements; future: dedicated TX node |
| Intel AX210 firmware changes | CSI extraction breakage | Pin firmware version; FeitCSI compatibility layer |

---

## 10. References

- [FeitCSI](https://github.com/mfeit-internet2/feitcsi) — Intel AX210 CSI extraction
- [ATAK](https://tak.gov) — Team Awareness Kit
- [IEEE 802.11 CSI](https://en.wikipedia.org/wiki/Channel_state_information) — Channel State Information overview
