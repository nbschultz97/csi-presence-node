#!/usr/bin/env bash
# Hardware validation script for VANTAGE deployment
# Checks all hardware and driver requirements before field deployment

set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

PASS=0
WARN=0
FAIL=0

echo "=========================================="
echo "VANTAGE Hardware Validation"
echo "=========================================="
echo ""

check_pass() {
    echo -e "${GREEN}[PASS]${NC} $1"
    ((PASS++))
}

check_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    ((WARN++))
}

check_fail() {
    echo -e "${RED}[FAIL]${NC} $1"
    ((FAIL++))
}

# 1. Check for Intel AX210
echo "1. Checking for Intel AX210 hardware..."
if command -v lspci &>/dev/null; then
    if lspci | grep -qi "AX210"; then
        check_pass "Intel AX210 detected"
    else
        check_fail "Intel AX210 NOT detected - required for CSI capture"
        echo "       Run: lspci | grep -i wireless"
    fi
else
    check_warn "lspci not available, cannot verify AX210 presence"
fi

# 2. Check iwlwifi driver loaded
echo ""
echo "2. Checking iwlwifi kernel driver..."
if command -v lsmod &>/dev/null; then
    if lsmod | grep -q "^iwlwifi"; then
        check_pass "iwlwifi driver loaded"
    else
        check_fail "iwlwifi driver NOT loaded"
        echo "       Fix: sudo modprobe iwlwifi"
    fi
else
    check_warn "lsmod not available, cannot verify iwlwifi"
fi

# 3. Check debugfs mounted
echo ""
echo "3. Checking debugfs filesystem..."
if mount | grep -q "debugfs"; then
    check_pass "debugfs mounted"
else
    check_fail "debugfs NOT mounted - required for CSI extraction"
    echo "       Fix: sudo mount -t debugfs debugfs /sys/kernel/debug"
fi

# 4. Check iwlwifi debugfs interface
echo ""
echo "4. Checking iwlwifi debugfs interface..."
if [[ -e /sys/kernel/debug/iwlwifi ]]; then
    check_pass "iwlwifi debugfs interface accessible"
elif [[ -e /sys/kernel/debug/ieee80211 ]]; then
    if find /sys/kernel/debug/ieee80211 -name "iwlwifi" -type d 2>/dev/null | grep -q .; then
        check_warn "iwlwifi debugfs found under ieee80211/ (needs symlink)"
        echo "       Fix: sudo ./scripts/preflight_root.sh"
    else
        check_fail "iwlwifi debugfs interface NOT found"
        echo "       This usually means AX210 is not present or driver not loaded"
    fi
else
    check_fail "iwlwifi debugfs interface NOT accessible"
    echo "       Fix: sudo ./scripts/preflight_root.sh"
fi

# 5. Check FeitCSI binary
echo ""
echo "5. Checking FeitCSI binary..."
FEITCSI_PATH="/usr/local/bin/feitcsi"
if [[ -x "$FEITCSI_PATH" ]]; then
    check_pass "FeitCSI binary found at $FEITCSI_PATH"
elif command -v feitcsi &>/dev/null; then
    check_pass "FeitCSI binary found in PATH: $(command -v feitcsi)"
else
    check_fail "FeitCSI binary NOT found"
    echo "       Fix: cd FeitCSI && make && sudo make install"
    echo "       Or run: ./setup.sh"
fi

# 6. Check FeitCSI capabilities
echo ""
echo "6. Checking FeitCSI capabilities..."
if [[ -x "$FEITCSI_PATH" ]]; then
    if command -v getcap &>/dev/null; then
        CAPS=$(getcap "$FEITCSI_PATH" 2>/dev/null)
        if echo "$CAPS" | grep -q "cap_net_admin.*cap_net_raw"; then
            check_pass "FeitCSI has required capabilities"
        else
            check_fail "FeitCSI missing required capabilities"
            echo "       Fix: sudo setcap cap_net_admin,cap_net_raw+eip $FEITCSI_PATH"
        fi
    else
        check_warn "getcap not available, cannot verify capabilities"
    fi
else
    check_warn "FeitCSI not found, skipping capability check"
fi

# 7. Check rfkill status
echo ""
echo "7. Checking WiFi rfkill status..."
if command -v rfkill &>/dev/null; then
    if rfkill list wifi | grep -q "Soft blocked: yes"; then
        check_warn "WiFi is soft-blocked by rfkill"
        echo "       Fix: sudo rfkill unblock all"
    else
        check_pass "WiFi not blocked by rfkill"
    fi
else
    check_warn "rfkill not available, cannot check block status"
fi

# 8. Check Python environment
echo ""
echo "8. Checking Python environment..."
if [[ -d ".venv" ]]; then
    check_pass "Python virtualenv found (.venv)"
    if [[ -f ".venv/bin/activate" ]]; then
        # Check if virtualenv has required packages
        if .venv/bin/python -c "import numpy, scipy, sklearn" 2>/dev/null; then
            check_pass "Required Python packages installed"
        else
            check_warn "Some Python packages missing"
            echo "       Fix: source .venv/bin/activate && pip install -r requirements.txt"
        fi
    fi
else
    check_warn "Python virtualenv NOT found"
    echo "       Fix: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
fi

# 9. Check config file
echo ""
echo "9. Checking configuration..."
if [[ -f "csi_node/config.yaml" ]]; then
    check_pass "Config file found (csi_node/config.yaml)"

    # Check if calibrated
    if grep -q "^calibrated: true" csi_node/config.yaml; then
        check_pass "System marked as calibrated"

        # Check for valid path_loss_exponent
        PLE=$(grep "^path_loss_exponent:" csi_node/config.yaml | awk '{print $2}')
        if [[ -n "$PLE" ]]; then
            # Check if negative (invalid)
            if (( $(echo "$PLE < 0" | bc -l 2>/dev/null || echo 0) )); then
                check_warn "path_loss_exponent is negative ($PLE) - calibration may be invalid"
                echo "       Re-run: Tools → Calibrate Distance, or python -m csi_node.calibrate"
            elif (( $(echo "$PLE < 1.5 || $PLE > 6" | bc -l 2>/dev/null || echo 0) )); then
                check_warn "path_loss_exponent ($PLE) outside typical range (1.5-6.0)"
                echo "       Consider re-calibrating at known distances"
            fi
        fi
    else
        check_warn "System NOT calibrated - distance estimates will be inaccurate"
        echo "       Fix: Run calibration wizard or python -m csi_node.calibrate"
    fi

    # Check baseline
    if grep -q "^baseline_file:" csi_node/config.yaml; then
        BASELINE=$(grep "^baseline_file:" csi_node/config.yaml | awk '{print $2}')
        if [[ -f "$BASELINE" ]]; then
            check_pass "Baseline file exists ($BASELINE)"
        else
            check_warn "Baseline file configured but not found: $BASELINE"
            echo "       Fix: python -m csi_node.baseline --duration 60"
        fi
    else
        check_warn "No baseline configured - presence detection may have false positives"
        echo "       Fix: python -m csi_node.baseline --duration 60"
    fi
else
    check_fail "Config file NOT found (csi_node/config.yaml)"
fi

# 10. Check data directory
echo ""
echo "10. Checking data directory..."
if [[ -d "data" ]]; then
    check_pass "Data directory exists"
else
    check_warn "Data directory not found (will be created on first run)"
fi

# Summary
echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}PASS:${NC} $PASS"
echo -e "${YELLOW}WARN:${NC} $WARN"
echo -e "${RED}FAIL:${NC} $FAIL"
echo ""

if [[ $FAIL -gt 0 ]]; then
    echo -e "${RED}CRITICAL FAILURES DETECTED${NC}"
    echo "System is NOT ready for deployment. Fix critical issues above."
    echo "Recommended: Run ./scripts/preflight_root.sh with sudo"
    exit 1
elif [[ $WARN -gt 0 ]]; then
    echo -e "${YELLOW}WARNINGS DETECTED${NC}"
    echo "System may work but with degraded performance or accuracy."
    echo "Recommended: Address warnings before field deployment."
    exit 0
else
    echo -e "${GREEN}ALL CHECKS PASSED${NC}"
    echo "System is ready for deployment."
    exit 0
fi
