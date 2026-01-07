# VANTAGE System - Hardware Requirements

## Critical Hardware Dependencies

### ⚠️ MANDATORY: Intel AX210 WiFi 6E NIC

**VANTAGE ONLY works with Intel AX210** - this is non-negotiable.

**Why?**
- FeitCSI (the CSI extraction tool) only supports Intel AX210
- AX210 exposes raw Channel State Information (CSI) via iwlwifi debugfs
- Other WiFi chips (Realtek, Broadcom, Qualcomm) do NOT provide CSI access

**Verify you have AX210:**
```bash
lspci | grep -i "AX210\|wireless"
# Expected output: "Intel Corporation Wi-Fi 6 AX210"
```

**If you see anything else** (e.g., "Intel Wireless-AC 9260", "Realtek RTL8822CE", etc.):
- **STOP**: This system will NOT work
- **Required**: Replace with Intel AX210 (PCIe or M.2/NGFF CNVio2 form factor)

### ⚠️ MANDATORY: Dual Antennas

**Direction detection requires 2 antennas.**

**Verify dual-antenna operation:**
```bash
# Start capture, then check RSSI values
iw dev wlan0 station dump
# Should show 2 chains with different RSSI values:
# signal:  -45 [-50, -40] dBm
#          ^    ^    ^
#          |    |    +-- Chain 1 (antenna 2)
#          |    +------- Chain 0 (antenna 1)
#          +------------ Average
```

**If you only see one value:**
- Check physical antenna connections
- Direction will always show "center" (unusable)

---

## Hardware Architecture

```
┌─────────────────────────────────────────────────────┐
│              Physical Layer (Hardware)              │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │  Intel AX210 WiFi 6E NIC                  │   │
│   │  - 802.11ax (WiFi 6)                      │   │
│   │  - 2.4 GHz / 5 GHz / 6 GHz bands          │   │
│   │  - MIMO: 2x2 (2 spatial streams)          │   │
│   │  - MU-MIMO capable                        │   │
│   │  - PCIe interface (or M.2 CNVio2)         │   │
│   │  - Device ID: 8086:2725                   │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
│                   │ (PHY registers via PCIe)        │
│                   │                                 │
│   ┌───────────────▼───────────────────────────┐   │
│   │  iwlwifi Kernel Driver                    │   │
│   │  - Part of Linux mainline kernel          │   │
│   │  - Loads firmware: iwlwifi-ty-*.ucode     │   │
│   │  - Exposes debugfs interface              │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
│                   │ (debugfs filesystem)            │
│                   │                                 │
│   ┌───────────────▼───────────────────────────┐   │
│   │  /sys/kernel/debug/iwlwifi/               │   │
│   │  - PHY registers                          │   │
│   │  - CSI matrix data                        │   │
│   │  - Driver debug info                      │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
└───────────────────┼─────────────────────────────────┘
                    │
                    │ (read by FeitCSI)
                    │
┌───────────────────▼─────────────────────────────────┐
│           Software Layer (VANTAGE)                  │
├─────────────────────────────────────────────────────┤
│                                                     │
│   ┌───────────────────────────────────────────┐   │
│   │  FeitCSI Binary                           │   │
│   │  - Reads iwlwifi debugfs                  │   │
│   │  - Extracts CSI matrices                  │   │
│   │  - Requires: cap_net_admin, cap_net_raw   │   │
│   │  - Outputs: .dat binary stream            │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
│                   ▼                                 │
│   ┌───────────────────────────────────────────┐   │
│   │  csi_raw.dat (binary CSI stream)          │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
│                   ▼                                 │
│   ┌───────────────────────────────────────────┐   │
│   │  Python Pipeline (csi_node/)              │   │
│   │  - Parses CSI                             │   │
│   │  - Variance/PCA presence detection        │   │
│   │  - RSSI-based distance/direction          │   │
│   │  - Pose classification                    │   │
│   └───────────────┬───────────────────────────┘   │
│                   │                                 │
│                   ▼                                 │
│   ┌───────────────────────────────────────────┐   │
│   │  presence_log.jsonl (Intelligence)        │   │
│   │  - Presence, pose, direction, distance    │   │
│   │  - JSON Lines format                      │   │
│   │  - ATAK integration ready                 │   │
│   └───────────────────────────────────────────┘   │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## Form Factors

### PCIe (Desktop/Server)
- **Form Factor**: Full-height/half-height PCIe card
- **Interface**: PCIe 3.0 x1
- **Antennas**: External via RP-SMA connectors
- **Recommended for**: Lab testing, fixed installations

### M.2 (Laptop/Embedded)
- **Form Factor**: M.2 2230 (22mm x 30mm)
- **Interface**: M.2 Key E (CNVio2 or PCIe)
- **Antennas**: Internal via MHF4/U.FL connectors
- **Recommended for**: Portable deployments, tactical ISR

**WARNING**: Some laptops have CNVio (Integrated RF) which requires specific Intel host controller. Verify compatibility before purchasing.

---

## Verified Compatible Systems

### Tested Working:
- **Dell Precision 5570** (M.2 AX210)
- **ThinkPad X1 Carbon Gen 9** (M.2 AX210)
- **Intel NUC 11** (M.2 AX210)
- **Generic PCIe AX210 cards** (desktop)

### Known Issues:
- **CNVio vs CNVio2**: AX210 requires CNVio2 host interface
  - **11th gen Intel Core and newer**: CNVio2 (compatible)
  - **10th gen Intel Core and older**: CNVio only (NOT compatible without adapter)
- **AMD Ryzen systems**: Use PCIe AX210 cards (not M.2 CNVio2)

**Before purchasing AX210:**
```bash
# Check your system's M.2 slot capabilities
lspci | grep -i "network\|wireless"
# Should show PCIe or CNVio2 compatible slot
```

---

## Alternative CSI Hardware (NOT SUPPORTED)

### Why not use:

**Intel Wi-Fi 5300 (legacy CSI research NIC)?**
- ❌ EOL, no longer manufactured
- ❌ Only 802.11n (WiFi 4), no modern bands
- ❌ Requires custom firmware patching (IWL5300 CSI Tool)
- ❌ No through-wall capability (low power)

**Atheros AR9380/AR9580 (ath9k CSI Tool)?**
- ❌ Only 802.11n
- ❌ Requires ath9k driver patching
- ❌ Limited to 2.4 GHz
- ❌ Poor CSI quality for presence detection

**Raspberry Pi with WiFi dongles?**
- ❌ No CSI access (consumer WiFi chips don't expose CSI)
- ❌ Only RSSI available (not sufficient for pose detection)

**SDR (Software Defined Radio) like USRP/HackRF?**
- ❌ Can decode WiFi, but CSI extraction is complex
- ❌ High cost ($1000+ for USRP)
- ❌ Power hungry, not field-deployable
- ✅ **Research use only**, not tactical

**Espressif ESP32 (ESP-IDF CSI)?**
- ❌ Low power, short range (~5m)
- ❌ Only 2.4 GHz
- ❌ Poor CSI quality (single antenna, limited bandwidth)
- ✅ **Good for learning**, not for VANTAGE use case

---

## Cost & Sourcing

### Intel AX210 Pricing (as of 2025):
- **M.2 module**: $15-30 USD (eBay, AliExpress)
- **PCIe card**: $25-50 USD (includes antennas)
- **OEM pulls**: $10-20 USD (from laptop upgrades)

### Recommended Vendors:
- **Amazon**: Search "Intel AX210 M.2" or "Intel AX210 PCIe"
- **Newegg**: OEM and retail packaged versions
- **eBay**: OEM pulls (cheaper, no warranty)

**WARNING**: Avoid counterfeit "AX210" cards (especially from AliExpress). Verify:
```bash
lspci -nn | grep -i wireless
# Should show: [8086:2725]  <- Intel vendor/device ID
```

### Antenna Sourcing:
- **M.2 antennas**: MHF4 (IPEX4) pigtails with adhesive mount
- **PCIe antennas**: RP-SMA 2.4/5 GHz dual-band
- **Cost**: $5-15 USD per pair

**Through-wall use**: Consider higher-gain antennas (5-7 dBi) for extended range.

---

## Power Requirements

### Intel AX210 Power Draw:
- **Idle**: ~0.5 W
- **RX (receive)**: ~1.5 W
- **TX (transmit)**: ~3.5 W (not used in VANTAGE)

**VANTAGE passive mode**: ~1.5 W (RX only, no transmit)

### System Power Budget:
- **AX210**: 1.5 W
- **Host CPU (idle)**: 5-15 W (depends on CPU)
- **Total**: ~10-20 W for embedded system

**Battery operation**: Intel NUC + AX210 can run 8+ hours on 100 Wh battery.

---

## Thermal Considerations

### AX210 Operating Temperature:
- **Commercial**: 0°C to 70°C
- **Extended**: -20°C to 85°C (industrial variants)

**Passive cooling**: AX210 generates minimal heat (~1.5 W). No heatsink required for M.2 in most laptops.

**High ambient**: For vehicles/outdoor, ensure airflow or active cooling above 50°C ambient.

---

## OS and Kernel Requirements

### Linux Kernel:
- **Minimum**: 5.10+ (iwlwifi AX210 support added)
- **Recommended**: 5.15+ (stable iwlwifi)
- **Tested**: 5.18, 6.1, 6.5

**Check your kernel:**
```bash
uname -r
# Example output: 6.1.0-kali9-amd64
```

### Distributions:
- ✅ **Kali Linux** (2022.1+)
- ✅ **Ubuntu** (20.04+)
- ✅ **Debian** (11+)
- ✅ **Arch Linux** (current)
- ⚠️ **CentOS/RHEL** (requires backported iwlwifi)
- ❌ **Windows** (FeitCSI is Linux-only)
- ❌ **macOS** (no iwlwifi driver)

---

## Firmware

### iwlwifi Firmware:
AX210 requires firmware: `iwlwifi-ty-a0-gf-a0-*.ucode`

**Check firmware loaded:**
```bash
dmesg | grep -i "iwlwifi.*firmware"
# Expected: "iwlwifi 0000:XX:00.0: loaded firmware version: ..."
```

**Update firmware (if missing):**
```bash
sudo apt-get install firmware-iwlwifi  # Debian/Ubuntu
# OR
sudo apt-get install linux-firmware    # Includes all Intel firmware
```

**Manual firmware install:**
```bash
wget https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/tree/iwlwifi-ty-a0-gf-a0-77.ucode
sudo cp iwlwifi-ty-a0-gf-a0-*.ucode /lib/firmware/
sudo modprobe -r iwlwifi && sudo modprobe iwlwifi
```

---

## Validation Before Deployment

### Quick Hardware Test:
```bash
# Run hardware validation script
./scripts/validate_hardware.sh

# Expected output:
# [PASS] Intel AX210 detected
# [PASS] iwlwifi driver loaded
# [PASS] debugfs mounted
# [PASS] iwlwifi debugfs interface accessible
# [PASS] FeitCSI binary found
# [PASS] FeitCSI has required capabilities
```

### Manual Checks:
```bash
# 1. Verify AX210 present
lspci | grep AX210

# 2. Verify iwlwifi loaded
lsmod | grep iwlwifi

# 3. Verify debugfs accessible
ls /sys/kernel/debug/iwlwifi/

# 4. Verify FeitCSI binary
which feitcsi
getcap /usr/local/bin/feitcsi

# 5. Test CSI capture (10 seconds)
./scripts/demo.sh
```

**If any check fails**: See DEPLOY.md troubleshooting section.

---

## Limitations

### Hardware Constraints:
1. **Single-sensor system**: Cannot triangulate position (requires 3+ sensors)
2. **Dual-antenna only**: Direction is left/center/right (not 360° bearing)
3. **Through-wall attenuation**: 1 wall OK, 2+ walls marginal
4. **Range**: ~10m line-of-sight, ~5m through-wall (2.4 GHz)
5. **Bandwidth**: 20 MHz minimum for usable CSI

### Environmental Limitations:
1. **High WiFi density**: Performance degrades with many APs/clients
2. **Metal structures**: Block/reflect signals (Faraday cage effect)
3. **Water**: Absorbs 2.4 GHz (humid environments degrade signal)
4. **Moving objects**: Fans, doors, curtains cause false positives

### Pose Detection Limitations:
1. **Single target**: Cannot distinguish multiple people
2. **Motion required**: Stationary targets fade (not radar)
3. **No human/object classification**: Responds to any reflector
4. **Pose accuracy**: Dependent on training data (current model is toy)

---

## Future Hardware Considerations

### Upcoming Intel WiFi 7 (BE200):
- 802.11be (WiFi 7)
- 6 GHz primary band
- 320 MHz channels
- **Unknown**: CSI access via debugfs (not yet verified)

**Recommendation**: Stick with AX210 until FeitCSI adds BE200 support.

### Multiple Sensors (Distributed VANTAGE):
- **Concept**: 3+ AX210 sensors for triangulation
- **Challenge**: Time synchronization (requires GPS or PTP)
- **Benefit**: True 2D/3D position (not just direction)
- **Status**: Not implemented in current version

---

## Hardware Procurement Checklist

Before field deployment, procure:

- [ ] Intel AX210 NIC (M.2 or PCIe)
- [ ] 2x WiFi antennas (MHF4 or RP-SMA)
- [ ] Host system (laptop/NUC/embedded PC)
- [ ] Linux USB drive (Kali/Ubuntu live boot for testing)
- [ ] Backup AX210 (in case of hardware failure)
- [ ] Antenna extension cables (if remote mounting needed)
- [ ] Power supply/battery (for field use)
- [ ] Ruggedized case (for tactical deployment)

**Budget estimate**: $100-300 USD for complete hardware setup.

---

**Last Updated**: 2026-01-07
**VANTAGE System - Hardware Requirements v1.0**
