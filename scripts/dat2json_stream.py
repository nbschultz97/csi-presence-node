#!/usr/bin/env python3
import argparse
import json
import math
import os
import struct
import sys
import time


def read_exact(f, n):
    """Read exactly n bytes from file f, waiting for more if needed."""
    buf = b""
    while len(buf) < n:
        chunk = f.read(n - len(buf))
        if not chunk:
            time.sleep(0.05)
            continue
        buf += chunk
    return buf


def try_read(f, n):
    """Try to read up to n bytes; return bytes read (may be empty)."""
    pos = f.tell()
    data = f.read(n)
    if len(data) < n:
        # not enough data yet; rewind and let caller retry later
        f.seek(pos)
    return data


def parse_frame(field: bytes):
    """Parse one FeitCSI .dat frame payload into a magnitude array 2x56.

    This matches FeitCSI's simple 2-RX x 56-subcarrier demo format used by
    parse_csi.py bundled with the FeitCSI source.
    """
    if len(field) < 18:
        return None
    # count = struct.unpack('<H', field[14:16])[0]  # unused here
    csi_raw = field[18:]
    needed = 2 * 2 * 56 * 2  # rx * subc * (i,q) int8
    if len(csi_raw) < needed:
        return None
    mags = []
    for i in range(2 * 56):
        r = struct.unpack('<b', csi_raw[2 * i : 2 * i + 1])[0]
        im = struct.unpack('<b', csi_raw[2 * i + 1 : 2 * i + 2])[0]
        mags.append((r * r + im * im) ** 0.5)
    # reshape to 2 x 56
    ch0 = mags[:56]
    ch1 = mags[56:112]
    return [ch0, ch1]


def _rssi_from_csi(csi2x56):
    """Compute per-chain relative RSSI (dB) from 2x56 magnitudes.

    Uses 20*log10(RMS amplitude) per chain. Absolute scale is arbitrary; use the
    calibration utility to align distance estimates for your environment.
    """
    try:
        ch0, ch1 = csi2x56
        # RMS amplitude per chain
        eps = 1e-12
        def rms(vals):
            return math.sqrt(sum(v * v for v in vals) / max(len(vals), 1))
        a0 = rms(ch0)
        a1 = rms(ch1)
        rssi0 = 20.0 * math.log10(max(a0, eps))
        rssi1 = 20.0 * math.log10(max(a1, eps))
        # Optional global offset to shift into a dBm-like range
        off = float(os.environ.get("DAT_RSSI_OFFSET", "0"))
        return [rssi0 + off, rssi1 + off]
    except Exception:
        return [-40.0, -40.0]


def stream(in_path: str, out_path: str, out2_path: str | None = None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    # Wait for input file to appear
    start = time.time()
    while not os.path.exists(in_path):
        time.sleep(0.1)
        if time.time() - start > 12:
            print(f"[error] input file not found: {in_path}", file=sys.stderr)
            sys.exit(2)
    # Open input for reading; don't seek to end so we catch frames from start
    with open(in_path, 'rb') as f_in, open(out_path, 'a', buffering=1) as f_out:
        f_out2 = None
        if out2_path:
            try:
                os.makedirs(os.path.dirname(out2_path), exist_ok=True)
                f_out2 = open(out2_path, 'a', buffering=1)
            except Exception:
                f_out2 = None
        while True:
            # Read 2-byte length prefix if available
            hdr = try_read(f_in, 2)
            if len(hdr) < 2:
                time.sleep(0.05)
                continue
            (field_len,) = struct.unpack('<H', hdr)
            field = try_read(f_in, field_len)
            if len(field) < field_len:
                time.sleep(0.05)
                continue
            csi = parse_frame(field)
            if csi is None:
                continue
            pkt = {
                'ts': time.time(),
                'rssi': _rssi_from_csi(csi),
                'csi': csi,
            }
            line = json.dumps(pkt) + '\n'
            f_out.write(line)
            if f_out2:
                try:
                    f_out2.write(line)
                except Exception:
                    pass


def main():
    ap = argparse.ArgumentParser(description='Stream FeitCSI .dat to JSONL')
    ap.add_argument('--in', dest='in_path', required=True, help='input .dat path')
    ap.add_argument('--out', dest='out_path', required=True, help='output .log path (JSONL)')
    ap.add_argument('--out2', dest='out2_path', default=None, help='optional second .log path (JSONL)')
    args = ap.parse_args()
    stream(args.in_path, args.out_path, args.out2_path)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
