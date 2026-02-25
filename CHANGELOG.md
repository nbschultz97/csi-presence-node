# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
- Environment profile manager for multi-site calibration
- Through-wall demo mode with ATAK dashboard integration and narration panel
- Enhanced web dashboard with SSE streaming, zone visualization, subcarrier heatmap, detection profiles, and data recording
- Demo simulation mode — full system evaluation without CSI hardware
- 64-second demo replay log for quick demonstrations
- Web dashboard with multi-method AdaptivePresenceDetector (energy 50%, variance 35%, spectral 15%)
- 112+ unit tests boosting coverage from 45% to 86%
- Comprehensive demo guide (DEMO.md)

### Changed
- Updated demo documentation with dashboard features, recording workflow, and through-wall tips

### Fixed
- Resolved `utcnow` deprecation warning
- Fixed test path-length bug on Windows
- Updated web dashboard tests for through-wall parameter

### Security
- Added sensitive file patterns to `.gitignore`

---

## [0.5.0] — Signal Pipeline & ML Upgrade

### Added
- Full 14-feature extractor with signal conditioning pipeline
- Hampel filter + Butterworth bandpass signal conditioning module
- RandomForest classifier with StandardScaler (replaced toy model)
- Movement classification integrated into pipeline
- PresenceDetector wired into main pipeline

### Changed
- Default radio config to 2.4 GHz channel 1 @ 20 MHz for through-wall optimization

### Fixed
- Baseline shape mismatch in `compute_window`

---

## [0.4.0] — GUI, Live Runner & FeitCSI Integration

### Added
- GUI application with live tracking window, through-wall preset, and calibration status
- One-command live runner: FeitCSI `.dat` → JSONL → pipeline
- `.dat` capture path with streaming converter
- RSSI estimation from CSI with TX power and path-loss calibration
- Distance calibration and threshold editor in GUI
- Capture Baseline action in GUI Tools menu
- FeitCSI pipeline integration with JSON logging

### Changed
- Improved pre/postflight checks and passwordless sudo support
- WiFi autoconnect with robust recovery (wpa_supplicant + NetworkManager)

---

## [0.3.0] — Pose Classification & TUI

### Added
- Pose classifier (standing, sitting, walking, prone)
- Curses-based TUI for terminal operation
- Offline replay from recorded CSI data
- Auto setup script and JSON output pipeline

### Changed
- Converted pose estimator input to torch tensor
- Flattened amplitude window for WiPose compatibility

---

## [0.2.0] — ATAK Integration & Setup

### Added
- ATAK Cursor-on-Target output integration
- UDP streaming for external C2 integration
- Guided setup wizard for first-time users
- Crash-safe per-run logging
- Doctor utility for deployment diagnostics
- AoA and range calibration pipeline with UI support
- Hardware validation script and deployment guide

### Fixed
- Critical `path_loss_exponent` bug
- Removed deprecated `multi_class` parameter from LogisticRegression

---

## [0.1.0] — Foundation

### Added
- CSI presence detection pipeline with baseline subtraction
- FeitCSI binary integration for Intel AX210 CSI capture
- Direction estimation via multi-chain RSSI comparison
- Orchestration scripts for capture → baseline → pipeline workflow
- Log rotation handling and CSI log freshness checks
- Single-chain RSSI fallback handling
- Virtual environment detection in installer
- Channel-to-frequency conversion utility
- Unit tests for presence/direction output, RSSI parsing, and error handling

### Fixed
- Numerous edge cases: missing log files, mismatched dimensions, empty/invalid CSI packets, single-chain RSSI

---

## [0.0.1] — Initial Commit

### Added
- Initial repository structure
- CSI presence pipeline and baseline tooling
