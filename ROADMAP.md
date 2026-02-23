# Roadmap

> **Vantage** — Passive WiFi CSI Through-Wall Human Detection

## Current Capabilities

- **Signal Pipeline:** Hampel filter + Butterworth bandpass + 14-feature extractor
- **CSI Capture:** FeitCSI integration for Intel AX210 hardware
- **Detection:** AdaptivePresenceDetector with multi-method fusion (energy 50%, variance 35%, spectral 15%)
- **Output:** ATAK Cursor-on-Target, UDP streaming, GUI/TUI, real-time web dashboard with SSE
- **Demo Mode:** Full simulation without hardware — ideal for evaluation and integration testing
- **Calibration:** Environment profile manager for multi-site deployment
- **Deep Learning:** CNN-LSTM model architecture implemented (training-ready, pending real CSI data)
- **Quality:** 819 tests passing across 170+ commits

---

## Phase 1: Real-World Validation *(Near-term)*

Transition from synthetic to real-world performance.

- [ ] Collect real CSI datasets across multiple environments and wall types
- [ ] Retrain RandomForest classifier on real through-wall capture data
- [ ] Validate per-environment distance calibration accuracy
- [ ] Field test through drywall, wood-frame, and concrete walls
- [ ] Establish baseline detection performance metrics (Pd, Pfa, range)

## Phase 2: ML Model Upgrade *(Medium-term)*

Replace classical ML with deep learning for improved accuracy.

- [x] CNN-LSTM architecture for temporal pattern recognition ✅ *(Feb 2026)*
- [ ] Multi-person tracking and counting (currently single-target)
- [ ] Improved pose classification with real training data
- [ ] ONNX export for optimized edge inference
- [ ] Model versioning and A/B testing framework

## Phase 3: Platform Maturity *(Longer-term)*

Scale from single-node to networked sensing platform.

- [ ] Multi-node mesh fusion — multiple sensors, fused operational picture
- [ ] Hardware-in-the-loop CI/CD testing
- [ ] Raspberry Pi and Jetson Nano/Orin deployment optimization
- [ ] Plugin architecture for custom detection algorithms
- [ ] AoA (Angle of Arrival) estimation via multi-antenna processing
- [ ] Doppler-based micro-movement and breathing detection

## Phase 4: Production Readiness

Harden for operational deployment.

- [ ] Encrypted telemetry and secure transport
- [ ] OTA firmware and model updates
- [ ] Fleet management dashboard
- [ ] MIL-STD environmental and EMI compliance testing
- [ ] FIPS 140 cryptographic compliance evaluation

---

## Contributing

We welcome contributions at any phase. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines, or open an issue to discuss a feature before submitting a PR.
