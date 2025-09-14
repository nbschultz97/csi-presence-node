from __future__ import annotations
"""Simple Tkinter GUI to run CSI capture and pipeline.

Features
- Live mode: starts FeitCSI capture via scripts/10_csi_capture.sh, then the
  Python pipeline and streams JSON output into the UI.
- Replay mode: runs the pipeline on a selected .log or .b64 file.

The GUI avoids the curses TUI; it shows logs inline. Pose can be toggled.

Notes
- FeitCSI may require elevated permissions unless capabilities are set.
- If capture fails, errors from the script/binary are shown in the log area.
"""

import os
import sys
import threading
import queue
import subprocess
from pathlib import Path
from tkinter import Tk, StringVar, IntVar, DoubleVar, BooleanVar, END, Menu, simpledialog, Toplevel
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
import shutil
import time

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover - runtime requirement
    yaml = None


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
DEFAULT_CFG = REPO_ROOT / "csi_node" / "config.yaml"
DATA_DIR = REPO_ROOT / "data"


class ProcessLogger:
    def __init__(self, name: str, proc: subprocess.Popen, out_q: queue.Queue):
        self.name = name
        self.proc = proc
        self.q = out_q
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        if self.proc.stdout:
            t = threading.Thread(target=self._pump, args=(self.proc.stdout,))
            t.daemon = True
            t.start()
            self._threads.append(t)
        if self.proc.stderr:
            t = threading.Thread(target=self._pump, args=(self.proc.stderr,))
            t.daemon = True
            t.start()
            self._threads.append(t)

    def _pump(self, stream) -> None:
        for line in iter(stream.readline, ""):
            if not line:
                break
            self.q.put((self.name, line.rstrip("\n")))

    def wait(self) -> int:
        try:
            return self.proc.wait()
        except Exception:
            return -1

    def stop(self) -> None:
        try:
            if self.proc.poll() is None:
                self.proc.terminate()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=0.5)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass


class App:
    def __init__(self, root: Tk):
        self.root = root
        root.title("CSI Presence Node")
        root.geometry("920x640")

        # Load defaults from config
        cfg = {}
        if yaml and DEFAULT_CFG.exists():
            try:
                cfg = yaml.safe_load(open(DEFAULT_CFG)) or {}
            except Exception:
                cfg = {}

        self.mode = StringVar(value="live")
        self.channel = IntVar(value=int(cfg.get("channel", 36)))
        self.width = IntVar(value=int(cfg.get("bandwidth", 80)))
        self.coding = StringVar(value="BCC")
        self.wifi_device = StringVar(value="")
        self.pose = BooleanVar(value=False)
        # When enabled, skip Python FeitCSI interface and use .dat → JSONL path
        self.dat_mode = BooleanVar(value=False)
        self.replay_path = StringVar(value=str((REPO_ROOT / "data" / "sample_csi.b64")))
        self.speed = DoubleVar(value=1.0)
        self.output_file = StringVar(value=str(Path(cfg.get("output_file", "data/presence_log.jsonl"))))
        self.autoscroll = BooleanVar(value=True)
        self.pwless_status = StringVar(value="Passwordless: unknown")
        self.calib_status = StringVar(value="Calibrated: unknown")
        # Dat-mode RSSI offset to align derived RSSI to dBm-like scale
        self.dat_rssi_offset = DoubleVar(value=float(cfg.get("dat_rssi_offset", -60.0)))
        # Optional window override (e.g., Through‑Wall preset)
        self._window_override: float | None = None
        # Tracking window state
        self._tracking_win: Toplevel | None = None
        self._last_entry: dict | None = None

        self.cap_plog: ProcessLogger | None = None
        self.conv_plog: ProcessLogger | None = None
        self.pipe_plog: ProcessLogger | None = None
        self.out_q: queue.Queue = queue.Queue()
        self._workflow_thread: threading.Thread | None = None
        self._prev_active_cons: list[tuple[str, str]] = []
        self._wifi_devs: list[str] = []
        # When true, prefer sudo -n (passwordless) over pkexec for privileged commands
        self._prefer_pwless_sudo: bool = False

        self._build_ui()
        self._schedule_pump()

    def _build_ui(self) -> None:
        # Menu bar (Tools -> Diagnostics / Setup Passwordless)
        menubar = Menu(self.root)
        tools = Menu(menubar, tearoff=0)
        tools.add_command(label="Diagnostics", command=self.run_diagnostics)
        tools.add_separator()
        tools.add_command(label="Setup Passwordless sudo…", command=self.setup_passwordless)
        tools.add_command(label="Fix Wi‑Fi Profile…", command=self.fix_wifi_profile)
        tools.add_separator()
        tools.add_command(label="Calibrate Distance…", command=self.calibrate_distance)
        tools.add_command(label="Edit Thresholds…", command=self.edit_thresholds)
        tools.add_command(label="Through‑Wall Preset", command=self.apply_through_wall_preset)
        tools.add_separator()
        tools.add_command(label="Show Tracking Window", command=self.show_tracking_window)
        menubar.add_cascade(label="Tools", menu=tools)
        self.root.config(menu=menubar)

        frm = ttk.Frame(self.root)
        frm.pack(fill="x", padx=10, pady=10)

        # Mode selection
        mode_frame = ttk.LabelFrame(frm, text="Mode")
        mode_frame.pack(fill="x", pady=5)
        ttk.Radiobutton(mode_frame, text="Live (FeitCSI)", variable=self.mode, value="live").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Radiobutton(mode_frame, text="Replay", variable=self.mode, value="replay").grid(row=0, column=1, sticky="w", padx=6, pady=4)

        # Live settings
        live_frame = ttk.LabelFrame(frm, text="Live Settings")
        live_frame.pack(fill="x", pady=5)
        ttk.Label(live_frame, text="Device").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        self.dev_combo = ttk.Combobox(live_frame, textvariable=self.wifi_device, values=[], width=16, state="readonly")
        self.dev_combo.grid(row=0, column=1, sticky="w", padx=6, pady=4)
        ttk.Button(live_frame, text="Refresh", command=self._populate_wifi_devices).grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Label(live_frame, text="Channel").grid(row=0, column=3, sticky="w", padx=6, pady=4)
        ttk.Entry(live_frame, textvariable=self.channel, width=8).grid(row=0, column=4, sticky="w", padx=6, pady=4)
        ttk.Label(live_frame, text="Width (MHz)").grid(row=0, column=5, sticky="w", padx=6, pady=4)
        ttk.Entry(live_frame, textvariable=self.width, width=8).grid(row=0, column=6, sticky="w", padx=6, pady=4)
        ttk.Label(live_frame, text="Coding").grid(row=0, column=7, sticky="w", padx=6, pady=4)
        ttk.Combobox(live_frame, textvariable=self.coding, values=["BCC", "LDPC"], width=10, state="readonly").grid(row=0, column=8, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(live_frame, text="Pose", variable=self.pose).grid(row=0, column=9, sticky="w", padx=6, pady=4)
        ttk.Checkbutton(live_frame, text="Dat mode", variable=self.dat_mode).grid(row=0, column=10, sticky="w", padx=6, pady=4)
        ttk.Label(live_frame, text="RSSI offset").grid(row=0, column=11, sticky="w", padx=6, pady=4)
        ttk.Entry(live_frame, textvariable=self.dat_rssi_offset, width=8).grid(row=0, column=12, sticky="w", padx=6, pady=4)

        # Replay settings
        rep_frame = ttk.LabelFrame(frm, text="Replay Settings")
        rep_frame.pack(fill="x", pady=5)
        ttk.Label(rep_frame, text="File").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(rep_frame, textvariable=self.replay_path, width=60).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(rep_frame, text="Browse…", command=self._browse).grid(row=0, column=2, sticky="w", padx=6, pady=4)
        ttk.Label(rep_frame, text="Speed").grid(row=0, column=3, sticky="w", padx=6, pady=4)
        ttk.Entry(rep_frame, textvariable=self.speed, width=8).grid(row=0, column=4, sticky="w", padx=6, pady=4)

        # Output path
        out_frame = ttk.Frame(frm)
        out_frame.pack(fill="x", pady=5)
        ttk.Label(out_frame, text="Output JSONL").grid(row=0, column=0, sticky="w", padx=6, pady=4)
        ttk.Entry(out_frame, textvariable=self.output_file, width=60).grid(row=0, column=1, sticky="we", padx=6, pady=4)
        ttk.Button(out_frame, text="Open", command=self._open_output).grid(row=0, column=2, sticky="w", padx=6, pady=4)

        # Controls
        ctrl = ttk.Frame(frm)
        ctrl.pack(fill="x", pady=5)
        self.btn_start = ttk.Button(ctrl, text="Start", command=self.start)
        self.btn_start.grid(row=0, column=0, padx=6)
        self.btn_stop = ttk.Button(ctrl, text="Stop", command=self.stop, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=6)
        ttk.Checkbutton(ctrl, text="Autoscroll", variable=self.autoscroll).grid(row=0, column=2, padx=6)
        ttk.Button(ctrl, text="Copy Selected", command=self._copy_selected).grid(row=0, column=3, padx=6)
        ttk.Button(ctrl, text="Copy All", command=self._copy_all).grid(row=0, column=4, padx=6)
        ttk.Button(ctrl, text="Save Log…", command=self._save_log).grid(row=0, column=5, padx=6)
        ttk.Button(ctrl, text="Clear", command=self._clear_log).grid(row=0, column=6, padx=6)
        ttk.Label(ctrl, textvariable=self.pwless_status).grid(row=0, column=7, padx=12)
        ttk.Label(ctrl, textvariable=self.calib_status).grid(row=0, column=8, padx=12)

        # Log output
        log_frame = ttk.LabelFrame(self.root, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        self.log = ScrolledText(log_frame, height=24, undo=True, maxundo=1000)
        self.log.pack(fill="both", expand=True)
        # Context menu for copy/save
        self._menu = Menu(self.log, tearoff=0)
        self._menu.add_command(label="Copy", command=self._copy_selected)
        self._menu.add_command(label="Copy All", command=self._copy_all)
        self._menu.add_separator()
        self._menu.add_command(label="Save Log…", command=self._save_log)
        self._menu.add_command(label="Clear", command=self._clear_log)
        self.log.bind("<Button-3>", self._show_menu)
        # Useful keybindings
        self.log.bind("<Control-a>", self._select_all)
        self.log.bind("<Control-A>", self._select_all)
        self._append("INFO", f"Repo: {REPO_ROOT}")
        self._append("INFO", f"Scripts: {SCRIPTS_DIR}")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        # Populate and check states
        self._populate_wifi_devices()
        self._update_pwless_status()
        self._update_calibration_status()

    def _append(self, tag: str, line: str) -> None:
        self.log.insert(END, f"[{tag}] {line}\n")
        if self.autoscroll.get():
            self.log.see(END)

    def _show_menu(self, event) -> None:
        try:
            self._menu.tk_popup(event.x_root, event.y_root)
        finally:
            self._menu.grab_release()

    def _select_all(self, event=None):  # type: ignore[no-redef]
        self.log.tag_add("sel", "1.0", END)
        return "break"

    def _copy_selected(self) -> None:
        try:
            text = self.log.get("sel.first", "sel.last")
        except Exception:
            # No selection; copy last line
            text = self.log.get("end-2l", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._append("INFO", "Copied to clipboard")

    def _copy_all(self) -> None:
        text = self.log.get("1.0", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._append("INFO", "All log text copied")

    def _save_log(self) -> None:
        text = self.log.get("1.0", "end-1c")
        if not text:
            messagebox.showinfo("Save Log", "Nothing to save.")
            return
        path = filedialog.asksaveasfilename(
            title="Save log",
            defaultextension=".txt",
            filetypes=[("Text files", ".txt"), ("All files", "*.*")],
        )
        if path:
            try:
                Path(path).write_text(text)
                self._append("INFO", f"Saved log to {path}")
            except Exception as exc:
                messagebox.showerror("Save Log", f"Failed to save log: {exc}")

    def _clear_log(self) -> None:
        self.log.delete("1.0", END)

    def _browse(self) -> None:
        path = filedialog.askopenfilename(title="Select capture file", filetypes=[
            ("Capture files", ".log .b64 .ftm"), ("All files", "*.*")
        ])
        if path:
            self.replay_path.set(path)

    def _open_output(self) -> None:
        path = Path(self.output_file.get())
        if not path.exists():
            messagebox.showinfo("Open output", "Output file not found yet.")
            return
        try:
            # Best effort: open with xdg-open if available
            subprocess.Popen(["xdg-open", str(path)])
        except Exception:
            self._append("INFO", f"Output at: {path}")

    def _schedule_pump(self) -> None:
        try:
            while True:
                name, line = self.out_q.get_nowait()
                self._append(name, line)
                # Try to parse pipeline JSON lines and update tracking
                if name == "PIPE" and line.startswith("{") and "\"presence\"" in line:
                    try:
                        import json as _json
                        obj = _json.loads(line)
                        self._last_entry = obj
                        self._update_tracking_window(obj)
                    except Exception:
                        pass
        except queue.Empty:
            pass
        self.root.after(100, self._schedule_pump)

    def start(self) -> None:
        if self.pipe_plog or self.cap_plog or self._workflow_thread:
            return
        self.btn_start.config(state="disabled")
        self.btn_stop.config(state="normal")
        self._workflow_thread = threading.Thread(target=self._run_workflow, daemon=True)
        self._workflow_thread.start()

    def _run_workflow(self) -> None:
        mode = self.mode.get()
        pose_flag = "--pose" if self.pose.get() else None
        out = self.output_file.get()

        if mode == "live":
            self._append("AUTO", "Running preflight: wifi off, rfkill unblock, driver, regdomain, caps…")
            self._preflight()

            # Prefer direct FeitCSI Python interface (no file tail). If the
            # pipeline exits immediately (module missing or permission issue),
            # fall back to file-based capture.
            if self.dat_mode.get():
                ok = self._start_dat_jsonl_capture()
                if not ok:
                    self._append("FEIT", "Dat mode start failed. See logs above.")
                    self.stop()
                    return
                # Wait until log file exists and is non-empty
                if not self._wait_for_file(REPO_ROOT / "data" / "csi_raw.log", 10.0):
                    self._append("FEIT", "Log not created after capture start; aborting.")
                    self.stop()
                    return
                # Start pipeline to tail file (unprivileged)
                pipe_cmd = [sys.executable, str(REPO_ROOT / "run.py"), "--out", out]
                if self._window_override:
                    pipe_cmd += ["--window", str(self._window_override)]
                if pose_flag:
                    pipe_cmd.append(pose_flag)
                self._append("PIPE", f"Starting pipeline (file): {' '.join(pipe_cmd)}")
                try:
                    pipe_proc = subprocess.Popen(
                        pipe_cmd,
                        cwd=str(REPO_ROOT),
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        bufsize=1,
                    )
                    self.pipe_plog = ProcessLogger("PIPE", pipe_proc, self.out_q)
                    self.pipe_plog.start()
                except Exception as exc:
                    self._append("PIPE", f"Failed to start pipeline: {exc}")
                return
            dev = self.wifi_device.get().strip()
            if not dev:
                # Try to populate and pick first available
                self._populate_wifi_devices()
                dev = self.wifi_device.get().strip()
            if not dev:
                self._append("PIPE", "No Wi‑Fi device selected/detected; cannot start live pipeline.")
                self.stop()
                return

            # Build candidate Python interpreters (venv, then system)
            py_venv = sys.executable
            py_sys = shutil.which("python3") or py_venv

            def _start_iface(py_exec: str, privileged: bool) -> subprocess.Popen:
                cmd = [py_exec, str(REPO_ROOT / "run.py"), "--iface", dev, "--out", out]
                if pose_flag:
                    cmd.append(pose_flag)
                tag_cmd = " ".join(cmd)
                if privileged:
                    if shutil.which("pkexec"):
                        full = ["pkexec"] + cmd
                    else:
                        full = ["sudo", "-n"] + cmd
                    self._append("PIPE", f"Starting pipeline (iface, root): {tag_cmd}")
                else:
                    full = cmd
                    self._append("PIPE", f"Starting pipeline (iface): {tag_cmd}")
                return subprocess.Popen(
                    full,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )

            def _stability_wait(proc: subprocess.Popen, secs: float = 2.0) -> bool:
                time.sleep(secs)
                return proc.poll() is None

            # 1) Try venv Python with root (most likely to have deps + permissions)
            try:
                pipe_proc = _start_iface(py_venv, privileged=True)
                self.pipe_plog = ProcessLogger("PIPE", pipe_proc, self.out_q)
                self.pipe_plog.start()
                if _stability_wait(pipe_proc):
                    # Running fine
                    pass
                else:
                    rc = pipe_proc.returncode
                    self._append("PIPE", f"Pipeline (iface, venv, root) exited early (rc={rc})")
                    # Capture stderr to detect missing module quickly
                    try:
                        _, err = pipe_proc.communicate(timeout=0.5)
                    except Exception:
                        err = ""
                    try:
                        self.pipe_plog.stop()
                    except Exception:
                        pass
                    self.pipe_plog = None

                    need_sys = ("No module named 'feitcsi'" in err) or ("CSIExtractor not available" in err)
                    if need_sys and py_sys != py_venv:
                        # 2) Try system python with root
                        self._append("PIPE", "Retrying with system Python (root)…")
                        pipe_proc = _start_iface(py_sys, privileged=True)
                        self.pipe_plog = ProcessLogger("PIPE", pipe_proc, self.out_q)
                        self.pipe_plog.start()
                        if not _stability_wait(pipe_proc):
                            rc = pipe_proc.returncode
                            self._append("PIPE", f"Pipeline (iface, system, root) exited early (rc={rc}); falling back to file capture…")
                            try:
                                self.pipe_plog.stop()
                            except Exception:
                                pass
                            self.pipe_plog = None
                            raise RuntimeError("iface_root_failed")
                    elif not need_sys:
                        self._append("PIPE", "Pipeline failed for another reason; falling back to file capture…")
                        raise RuntimeError("iface_root_failed")
                    # else: system Python started and is running
            except Exception:
                # Fallback to .dat → JSONL capture path
                ok = self._start_dat_jsonl_capture()
                if not ok:
                    self._append("FEIT", "All capture attempts failed. See logs above.")
                    self.stop()
                    return
                # Wait until log file exists and is non-empty
                if not self._wait_for_file(REPO_ROOT / "data" / "csi_raw.log", 10.0):
                    self._append("FEIT", "Log not created after capture start; aborting.")
                    self.stop()
                    return
                # Rebuild pipeline command to tail file (unprivileged is fine here)
                pipe_cmd = [py_venv, str(REPO_ROOT / "run.py"), "--out", out]
                if pose_flag:
                    pipe_cmd.append(pose_flag)
                self._append("PIPE", f"Starting pipeline (file): {' '.join(pipe_cmd)}")
                pipe_proc = subprocess.Popen(
                    pipe_cmd,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self.pipe_plog = ProcessLogger("PIPE", pipe_proc, self.out_q)
                self.pipe_plog.start()
        else:
            rep = self.replay_path.get()
            speed = str(self.speed.get())
            pipe_cmd = [sys.executable, str(REPO_ROOT / "run.py"), "--replay", rep, "--speed", speed, "--out", out]
            if pose_flag:
                pipe_cmd.append(pose_flag)
            self._append("PIPE", f"Starting pipeline: {' '.join(pipe_cmd)}")
            try:
                pipe_proc = subprocess.Popen(
                    pipe_cmd,
                    cwd=str(REPO_ROOT),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                )
                self.pipe_plog = ProcessLogger("PIPE", pipe_proc, self.out_q)
                self.pipe_plog.start()
            except Exception as exc:
                self._append("PIPE", f"Failed to start pipeline: {exc}")

    def stop(self) -> None:
        if self.pipe_plog:
            self._append("PIPE", "Stopping pipeline…")
            self.pipe_plog.stop()
            self.pipe_plog = None
        if self.conv_plog:
            self._append("CONV", "Stopping converter…")
            try:
                self.conv_plog.stop()
            finally:
                self.conv_plog = None
        if self.cap_plog:
            self._append("FEIT", "Stopping capture…")
            self.cap_plog.stop()
            self.cap_plog = None
        # Clean up FeitCSI-created interfaces if present to restore normal state
        self._cleanup_feit_ifaces()
        # Bring networking back up and try to reconnect saved Wi‑Fi
        self._restart_networking()
        self._reconnect_previous_connections()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._workflow_thread = None

    def _cleanup_feit_ifaces(self) -> None:
        """Remove FeitCSImon/FeitCSIap interfaces if they linger after capture."""
        for dev in ("FeitCSImon", "FeitCSIap"):
            self._run_cmd(["ip", "link", "set", dev, "down"], "AUTO", privileged=True)
            # `iw dev <iface> del` removes the interface when supported
            self._run_cmd(["iw", "dev", dev, "del"], "AUTO", privileged=True)

    def _start_dat_jsonl_capture(self) -> bool:
        """Start FeitCSI writing .dat and launch dat→JSONL converter.

        Tries the requested width, then falls back to 80 → 40 → 20. If the
        channel is > 14 (5 GHz) and all widths fail, try channel 1 (2.4 GHz)
        with the same fallback widths.
        """
        if self.cap_plog:
            try:
                self.cap_plog.stop()
            finally:
                self.cap_plog = None
        if self.conv_plog:
            try:
                self.conv_plog.stop()
            finally:
                self.conv_plog = None

        ch = int(self.channel.get())
        requested = int(self.width.get()) if self.width.get() > 0 else 20
        dat_path = DATA_DIR / "csi_raw.dat"
        jsonl_path = DATA_DIR / "csi_raw.log"
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            for p in (dat_path, jsonl_path):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
        except Exception:
            pass

        def _try_once(channel: int, width: int) -> bool:
            freq = self._channel_to_freq(channel)
            cmd = [
                "/usr/local/bin/feitcsi",
                "-f", str(freq),
                "-w", str(width),
                "-o", str(dat_path),
                "-v",
            ]
            full_cmd = cmd
            if self._prefer_pwless_sudo and shutil.which("sudo"):
                full_cmd = ["sudo", "-n"] + cmd
            elif shutil.which("pkexec"):
                full_cmd = ["pkexec"] + cmd
            elif shutil.which("sudo"):
                full_cmd = ["sudo"] + cmd
            self._append("FEIT", f"Attempting (dat): {' '.join(full_cmd)}")
            try:
                proc = subprocess.Popen(full_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            except FileNotFoundError:
                self._append("FEIT", "FeitCSI binary not found at /usr/local/bin/feitcsi")
                return False
            self.cap_plog = ProcessLogger("FEIT", proc, self.out_q)
            self.cap_plog.start()
            # Wait for .dat to become non-empty
            for _ in range(24):
                try:
                    if dat_path.exists() and dat_path.stat().st_size > 0:
                        self._append("FEIT", f".dat active: {dat_path}")
                        return True
                except Exception:
                    pass
                time.sleep(0.5)
            # Stop this attempt if no data
            try:
                self.cap_plog.stop()
            except Exception:
                pass
            self.cap_plog = None
            self._append("FEIT", f".dat not created or empty: {dat_path}")
            return False

        tried = []
        widths = []
        for w in (requested, 80, 40, 20):
            if w > 0 and w not in widths:
                widths.append(w)

        # First try requested channel with fallback widths
        for w in widths:
            if _try_once(ch, w):
                break
            tried.append((ch, w))
        else:
            # If on 5 GHz, try 2.4 GHz channel 1 with width fallbacks
            if ch > 14:
                for w in widths:
                    if _try_once(1, w):
                        ch = 1  # reflect the channel actually used
                        break
                    tried.append((1, w))
                else:
                    self._append("FEIT", f"All attempts failed: {tried}")
                    return False
            else:
                self._append("FEIT", f"All attempts failed: {tried}")
                return False

        # Start converter
        conv = [sys.executable, str(REPO_ROOT / "scripts" / "dat2json_stream.py"), "--in", str(dat_path), "--out", str(jsonl_path)]
        self._append("CONV", f"Starting converter: {' '.join(conv)}")
        try:
            env = os.environ.copy()
            env["DAT_RSSI_OFFSET"] = str(self.dat_rssi_offset.get())
            cproc = subprocess.Popen(conv, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
            self.conv_plog = ProcessLogger("CONV", cproc, self.out_q)
            self.conv_plog.start()
        except Exception as exc:
            self._append("CONV", f"Failed to start converter: {exc}")
        # Confirm JSON log starts writing
        for _ in range(40):
            try:
                if jsonl_path.exists() and jsonl_path.stat().st_size > 0:
                    self._append("CONV", f"JSON log active: {jsonl_path}")
                    break
            except Exception:
                pass
            time.sleep(0.25)
        return True

    def run_diagnostics(self) -> None:
        t = threading.Thread(target=self._run_diagnostics, daemon=True)
        t.start()

    def _run_and_capture(self, cmd: list[str], tag: str, privileged: bool = False, timeout: float | None = None) -> tuple[int, str]:
        full_cmd = cmd
        if privileged:
            # Prefer passwordless sudo when configured; otherwise prefer pkexec
            # (GUI prompt). Fallback to sudo without -n as a last resort.
            if getattr(self, "_prefer_pwless_sudo", False) and shutil.which("sudo"):
                full_cmd = ["sudo", "-n"] + cmd
            elif shutil.which("pkexec"):
                full_cmd = ["pkexec"] + cmd
            elif shutil.which("sudo"):
                full_cmd = ["sudo"] + cmd
        self._append(tag, f"$ {' '.join(full_cmd)}")
        combined_lines: list[str] = []
        try:
            proc = subprocess.Popen(full_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            for line in (out or "").splitlines():
                self._append(tag, line)
                combined_lines.append(line)
            for line in (err or "").splitlines():
                self._append(tag, line)
                combined_lines.append(line)
            combined = "\n".join(combined_lines)
            if combined:
                combined += "\n"
            return proc.returncode, combined
        except Exception as exc:
            msg = f"Command failed: {exc}\n"
            self._append(tag, msg.rstrip())
            return -1, msg

    def _run_diagnostics(self) -> None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_path = DATA_DIR / f"diagnostics-{ts}.txt"
        self._append("DIAG", f"Collecting diagnostics to {out_path}…")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        sections: list[tuple[str, list[str], bool]] = [
            ("uname -a", ["uname", "-a"], False),
            ("iw dev", ["iw", "dev"], False),
            ("ip -o link show", ["ip", "-o", "link", "show"], False),
            ("rfkill list", ["rfkill", "list"], False),
            ("nmcli radio all", ["nmcli", "radio", "all"], False),
            ("nmcli dev status", ["nmcli", "dev", "status"], False),
            ("which feitcsi", ["which", "feitcsi"], False),
            ("ls -l /usr/local/bin/feitcsi", ["ls", "-l", "/usr/local/bin/feitcsi"], False),
            ("getcap feitcsi", ["getcap", "/usr/local/bin/feitcsi"], False),
            ("lsmod (wireless)", ["lsmod"], False),
            ("modinfo iwlwifi (truncated)", ["modinfo", "iwlwifi"], False),
            ("iw reg get", ["iw", "reg", "get"], False),
        ]
        logs_to_tail = [
            ("tail -n 200 data/feitcsi.log", ["tail", "-n", "200", str(DATA_DIR / "feitcsi.log")]),
            ("tail -n 200 data/presence_log.jsonl", ["tail", "-n", "200", str(DATA_DIR / "presence_log.jsonl")]),
        ]
        # Run commands and collect output
        output_chunks: list[str] = []
        for title, cmd, needs_priv in sections:
            rc, txt = self._run_and_capture(cmd, "DIAG", privileged=needs_priv)
            if title.startswith("lsmod"):
                # filter busy output
                txt = "\n".join([line for line in txt.splitlines() if any(k in line for k in ("iwlwifi", "mac80211", "cfg80211"))]) + "\n"
            if title.startswith("modinfo"):
                txt = "\n".join(txt.splitlines()[:80]) + "\n"
            output_chunks.append(f"==== {title} ====\n{txt}")
        for title, cmd in logs_to_tail:
            if Path(cmd[-1]).exists():
                _, txt = self._run_and_capture(cmd, "DIAG")
                output_chunks.append(f"==== {title} ====\n{txt}")
        # dmesg (privileged on some systems)
        _, txt = self._run_and_capture(["dmesg", "-T"], "DIAG", privileged=True)
        filt = [line for line in txt.splitlines() if any(k in line.lower() for k in ("iwlwifi", "nl80211", "mac80211", "feitcsi"))]
        output_chunks.append("==== dmesg -T (filtered) ====\n" + "\n".join(filt[-200:]) + "\n")
        try:
            out_path.write_text("\n".join(output_chunks))
            self._append("DIAG", f"Diagnostics saved to {out_path}")
            try:
                subprocess.Popen(["xdg-open", str(out_path)])
            except Exception:
                pass
        except Exception as exc:
            messagebox.showerror("Diagnostics", f"Failed to save diagnostics: {exc}")

    def _run_cmd(self, cmd: list[str], tag: str, privileged: bool = False, timeout: float | None = None) -> int:
        full_cmd = cmd
        if privileged:
            # Prefer sudo -n if passwordless is configured; otherwise pkexec
            if self._prefer_pwless_sudo and shutil.which("sudo"):
                full_cmd = ["sudo", "-n"] + cmd
            elif shutil.which("pkexec"):
                full_cmd = ["pkexec"] + cmd
            elif shutil.which("sudo"):
                full_cmd = ["sudo", "-n"] + cmd
        self._append(tag, f"$ {' '.join(full_cmd)}")
        try:
            proc = subprocess.Popen(full_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                out, err = proc.communicate(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                out, err = proc.communicate()
            for line in (out or "").splitlines():
                self._append(tag, line)
            for line in (err or "").splitlines():
                self._append(tag, line)
            return proc.returncode
        except Exception as exc:
            self._append(tag, f"Command failed: {exc}")
            return -1

    def _preflight(self) -> None:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Save previously active Wi‑Fi connections for later restore
        self._snapshot_active_connections()
        # 1) Disconnect Wi‑Fi devices (keep interface present)
        self._disconnect_wifi_devices()
        # 2–6) Combined privileged steps in one call to reduce prompts
        self._run_root_preflight()
        # 7) Verify iwlwifi debugfs exists (unprivileged existence check)
        try:
            if Path("/sys/kernel/debug/iwlwifi").exists():
                self._append("AUTO", "iwlwifi debugfs present.")
            else:
                self._append("AUTO", "iwlwifi debugfs not found after mount/link. Ensure FeitCSI iwlwifi module is loaded.")
        except Exception:
            self._append("AUTO", "iwlwifi debugfs check inconclusive (permission). Proceeding…")

    @staticmethod
    def _channel_to_freq(ch: int) -> int:
        if 1 <= ch <= 13:
            return 2407 + ch * 5
        if ch == 14:
            return 2484
        return 5000 + ch * 5

    @staticmethod
    def _wait_for_file(path: Path, timeout: float) -> bool:
        end = time.time() + timeout
        while time.time() < end:
            try:
                if path.exists() and path.stat().st_size > 0:
                    return True
            except Exception:
                pass
            time.sleep(0.25)
        return False

    def _restart_networking(self) -> None:
        self._append("AUTO", "Re-enabling Wi‑Fi and restarting NetworkManager…")
        # Unblock rfkill regardless
        self._run_cmd(["rfkill", "unblock", "all"], "AUTO", privileged=True)
        # Try systemd restart, fallback to service; also restart wpa_supplicant
        rc = self._run_cmd(["systemctl", "restart", "NetworkManager"], "AUTO", privileged=True)
        if rc != 0:
            self._run_cmd(["service", "NetworkManager", "restart"], "AUTO", privileged=True)
        rc = self._run_cmd(["systemctl", "restart", "wpa_supplicant"], "AUTO", privileged=True)
        if rc != 0:
            self._run_cmd(["service", "wpa_supplicant", "restart"], "AUTO", privileged=True)
        # Ensure networking/wifi are on
        self._run_cmd(["nmcli", "networking", "on"], "AUTO")
        self._run_cmd(["nmcli", "radio", "wifi", "on"], "AUTO")
        # Put previously used wifi devices back under NetworkManager control
        for dev in self._wifi_devs:
            self._run_cmd(["nmcli", "dev", "set", dev, "managed", "yes"], "AUTO", privileged=True)
            # Bring the link up in case it was left down
            self._run_cmd(["ip", "link", "set", dev, "up"], "AUTO", privileged=True)
        # Trigger a scan to force the stack into an available state
        self._run_cmd(["nmcli", "device", "wifi", "rescan"], "AUTO")
        # Wait for devices to become available again; if not, reload driver
        available = self._wait_wifi_available(timeout=8.0)
        if not available:
            self._append("AUTO", "Wi‑Fi still unavailable; reloading iwlwifi driver…")
            # Best effort: remove and reload iwlwifi/iwlmvm
            self._run_cmd(["modprobe", "-r", "iwlmvm"], "AUTO", privileged=True)
            self._run_cmd(["modprobe", "-r", "iwlwifi"], "AUTO", privileged=True)
            self._run_cmd(["modprobe", "iwlwifi"], "AUTO", privileged=True)
            # Ensure rfkill/networking/radio are up again
            self._run_cmd(["rfkill", "unblock", "all"], "AUTO", privileged=True)
            self._run_cmd(["nmcli", "networking", "on"], "AUTO")
            self._run_cmd(["nmcli", "radio", "wifi", "on"], "AUTO")
            self._wait_wifi_available(timeout=8.0)

    def _wait_wifi_available(self, dev: str | None = None, timeout: float = 10.0) -> bool:
        """Wait until Wi‑Fi device is not 'unavailable' according to nmcli.

        If ``dev`` is None, use the selected device or any Wi‑Fi device found.
        Returns True if available; False if timeout.
        """
        import time as _t
        # Pick a device
        if not dev:
            dev = self.wifi_device.get().strip()
            if not dev:
                devs = self._detect_wifi_devices()
                dev = devs[0] if devs else ""
        end = _t.time() + timeout
        while _t.time() < end:
            rc, txt = self._run_and_capture(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev", "status"], "AUTO")
            if rc == 0:
                for line in txt.splitlines():
                    parts = line.split(":")
                    if len(parts) >= 3 and parts[0] == dev and parts[1] in ("wifi", "802-11-wireless"):
                        state = parts[2]
                        if "unavailable" not in state:
                            return True
            _t.sleep(0.5)
        return False

    def _snapshot_active_connections(self) -> None:
        rc, txt = self._run_and_capture([
            "nmcli", "-t", "-f", "NAME,UUID,TYPE", "connection", "show", "--active"
        ], "AUTO")
        cons: list[tuple[str, str]] = []
        for line in txt.splitlines():
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            name, uuid, ctype = parts[0], parts[1], parts[2]
            ctype_norm = ctype.strip().lower()
            if ctype_norm in ("wifi", "802-11-wireless") or "wireless" in ctype_norm:
                cons.append((name, uuid))
        self._prev_active_cons = cons
        if cons:
            names = ", ".join(n for n, _ in cons)
            self._append("AUTO", f"Saved active Wi‑Fi connections: {names}")
        else:
            self._append("AUTO", "No active Wi‑Fi connections to restore later")

    def _reconnect_previous_connections(self) -> None:
        if not self._prev_active_cons:
            self._append("AUTO", "No saved Wi‑Fi connections to reconnect")
            return
        for name, uuid in self._prev_active_cons:
            self._append("AUTO", f"Reconnecting to '{name}' ({uuid})…")
            rc = self._run_cmd(["nmcli", "connection", "up", "uuid", uuid], "AUTO")
            if rc != 0:
                # Attempt to bring up by id; if that fails due to interface
                # binding, clear the binding and retry.
                rc2 = self._run_cmd(["nmcli", "connection", "up", "id", name], "AUTO")
                if rc2 != 0:
                    self._maybe_fix_nm_interface_binding(name)
                    self._run_cmd(["nmcli", "connection", "up", "id", name], "AUTO")
                uuid = ""  # fallback used id; uuid may not match
            self._ensure_autoconnect(name, uuid)
        self._prev_active_cons = []

    def _maybe_fix_nm_interface_binding(self, name: str) -> None:
        """If the connection is pinned to the wrong interface (e.g., eth0), clear it.

        This addresses errors like:
        "device eth0 not available because profile is not compatible with device (mismatching interface name)"
        """
        self._append("AUTO", f"Checking interface binding for '{name}'…")
        rc, out = self._run_and_capture(["nmcli", "-g", "connection.interface-name", "connection", "show", "id", name], "AUTO")
        if rc == 0:
            iface = (out or "").strip()
            if iface and iface != "wlan0":
                self._append("AUTO", f"Clearing mismatched interface-name '{iface}' on '{name}'")
                self._run_cmd(["nmcli", "connection", "modify", "id", name, "connection.interface-name", ""], "AUTO")

    def _ensure_autoconnect(self, name: str, uuid: str) -> None:
        """Force NetworkManager to autoconnect to the restored Wi‑Fi profile."""
        self._append("AUTO", f"Ensuring autoconnect for '{name}'…")
        if uuid:
            rc = self._run_cmd([
                "nmcli", "connection", "modify", "uuid", uuid, "connection.autoconnect", "yes"
            ], "AUTO")
            if rc != 0:
                self._run_cmd([
                    "nmcli", "connection", "modify", "id", name, "connection.autoconnect", "yes"
                ], "AUTO")
            rc = self._run_cmd([
                "nmcli", "connection", "modify", "uuid", uuid, "connection.autoconnect-priority", "100"
            ], "AUTO")
            if rc != 0:
                self._run_cmd([
                    "nmcli", "connection", "modify", "id", name, "connection.autoconnect-priority", "100"
                ], "AUTO")
        else:
            # No UUID; best-effort by id only
            self._run_cmd([
                "nmcli", "connection", "modify", "id", name, "connection.autoconnect", "yes"
            ], "AUTO")
            self._run_cmd([
                "nmcli", "connection", "modify", "id", name, "connection.autoconnect-priority", "100"
            ], "AUTO")

    def _run_root_preflight(self) -> None:
        """Run all privileged preflight actions in one pkexec call.

        Falls back to per-step commands if pkexec is unavailable or fails.
        """
        script = REPO_ROOT / "scripts" / "preflight_root.sh"
        if script.exists():
            # Prefer a single pkexec prompt for the entire preflight script.
            # Per-step commands later use passwordless sudo when configured.
            if shutil.which("pkexec"):
                cmd = ["pkexec", "bash", str(script)]
            elif self._prefer_pwless_sudo and shutil.which("sudo"):
                cmd = ["sudo", "-n", "bash", str(script)]
            elif shutil.which("sudo"):
                cmd = ["sudo", "bash", str(script)]
            self._append("AUTO", f"$ {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                out, err = proc.communicate()
                for line in (out or "").splitlines():
                    self._append("AUTO", line)
                for line in (err or "").splitlines():
                    self._append("AUTO", line)
                if proc.returncode == 0:
                    return
                else:
                    self._append("AUTO", f"preflight_root.sh exited {proc.returncode}; falling back to per-step commands")
            except Exception as exc:
                self._append("AUTO", f"pkexec preflight failed: {exc}; falling back")

        # Fallback discrete steps (avoid shells to minimize prompts)
        self._run_cmd(["rfkill", "unblock", "all"], "AUTO", privileged=True)
        self._run_cmd(["modprobe", "iwlwifi"], "AUTO", privileged=True)
        self._run_cmd(["iw", "reg", "set", "US"], "AUTO", privileged=True)
        self._run_cmd(["mountpoint", "-q", "/sys/kernel/debug"], "AUTO")
        self._run_cmd(["mount", "-t", "debugfs", "debugfs", "/sys/kernel/debug"], "AUTO", privileged=True)
        # Try to link ieee80211/.../iwlwifi to /sys/kernel/debug/iwlwifi (find unprivileged, ln privileged)
        from pathlib import Path as _P
        if not _P("/sys/kernel/debug/iwlwifi").exists():
            rc, txt = self._run_and_capture(["find", "/sys/kernel/debug/ieee80211", "-maxdepth", "3", "-type", "d", "-name", "iwlwifi"], "AUTO")
            alt = txt.splitlines()[0].strip() if txt.strip() else ""
            if alt:
                self._run_cmd(["ln", "-s", alt, "/sys/kernel/debug/iwlwifi"], "AUTO", privileged=True)
        self._run_cmd(["/usr/sbin/setcap", "cap_net_admin,cap_net_raw+eip", "/usr/local/bin/feitcsi"], "AUTO", privileged=True)
        self._run_cmd(["setcap", "cap_net_admin,cap_net_raw+eip", "/usr/local/bin/feitcsi"], "AUTO", privileged=True)

    def _detect_wifi_devices(self) -> list[str]:
        # Gather devices from nmcli and supplement with iw dev to include monitor ifaces
        rc, txt = self._run_and_capture(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev", "status"], "AUTO")
        devs: list[str] = []
        for line in txt.splitlines():
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            dev, typ, state = parts[0], parts[1], parts[2]
            if typ.strip().lower() in ("wifi", "802-11-wireless"):
                # Exclude FeitCSI temporary and P2P interfaces from selection
                if dev.startswith("FeitCSI") or dev.startswith("p2p-dev-"):
                    continue
                devs.append(dev)
        # Also check iw for monitor/managed interfaces that nmcli may omit
        rc, txt = self._run_and_capture(["iw", "dev"], "AUTO")
        cur = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("Interface "):
                cur = line.split()[1]
            elif line.startswith("type ") and cur:
                typ = line.split()[1]
                # Only include managed/station; skip monitor and FeitCSI temporary ifaces
                if typ in ("managed", "station"):
                    if cur.startswith("FeitCSI") or cur.startswith("p2p-dev-"):
                        cur = None
                    else:
                        devs.append(cur)
                cur = None
        # Dedupe and prefer wlan* first for nicer UX
        uniq = list(dict.fromkeys(devs))
        uniq.sort(key=lambda d: (not d.startswith("wlan"), d))
        return uniq

    def _disconnect_wifi_devices(self) -> None:
        devs = self._detect_wifi_devices()
        self._wifi_devs = devs
        sel = self.wifi_device.get().strip()

        # Proactively disconnect ALL managed wifi devices so monitor capture can work
        rc, txt = self._run_and_capture(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev", "status"], "AUTO")
        to_disconnect: list[str] = []
        if rc == 0:
            for line in txt.splitlines():
                parts = line.split(":")
                if len(parts) >= 3:
                    dev, typ, state = parts[0], parts[1], parts[2]
                    if typ == "wifi" and dev and not dev.startswith("p2p-dev-"):
                        to_disconnect.append(dev)
        # If none reported by nmcli, fall back to best-effort using iw list
        if not to_disconnect:
            for d in devs:
                if d and not d.startswith("p2p-dev-") and d != "lo":
                    to_disconnect.append(d)

        if to_disconnect:
            names = ", ".join(to_disconnect)
            self._append("AUTO", f"Disconnecting wifi devices: {names}…")
            for d in to_disconnect:
                self._run_cmd(["nmcli", "dev", "disconnect", d], "AUTO")

        # Choose target (prefer the selected device if present)
        target = sel if sel in devs else (devs[0] if devs else "")
        if target:
            if sel and sel != target:
                self.wifi_device.set(target)
            # Ensure target interface is UP
            self._run_cmd(["ip", "link", "set", target, "up"], "AUTO", privileged=True)
        else:
            # Fallback: radio off to release stack
            self._append("AUTO", "No Wi‑Fi device found via nmcli/iw; turning radio off")
            self._run_cmd(["nmcli", "radio", "wifi", "off"], "AUTO")

    def _populate_wifi_devices(self) -> None:
        devs = self._detect_wifi_devices()
        try:
            self.dev_combo["values"] = devs
        except Exception:
            pass
        # Keep the current selection if it's still valid; otherwise choose first
        cur = self.wifi_device.get().strip()
        if devs:
            if cur in devs:
                # Do not overwrite user selection
                pass
            else:
                self.wifi_device.set(devs[0])

    def _update_pwless_status(self) -> None:
        ok = self._check_passwordless()
        self._prefer_pwless_sudo = bool(ok)
        self.pwless_status.set("Passwordless: OK" if ok else "Passwordless: not configured")

    def setup_passwordless(self) -> None:
        username = simpledialog.askstring("Passwordless sudo", "Enter your username:", initialvalue=os.getenv("USER", ""))
        if not username:
            return
        script = str(REPO_ROOT / "scripts" / "setup_passwordless.sh")
        cmd = ["pkexec", "bash", script, username]
        self._append("AUTO", f"Configuring passwordless sudo for {username}…")
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            for line in (out or "").splitlines():
                self._append("AUTO", line)
            for line in (err or "").splitlines():
                self._append("AUTO", line)
            if proc.returncode == 0:
                self._append("AUTO", "Passwordless sudo configured.")
                # Recheck status
                self._update_pwless_status()
            else:
                self._append("AUTO", f"Passwordless setup exited with code {proc.returncode}")
        except Exception as exc:
            self._append("AUTO", f"Failed to run setup: {exc}")

    def fix_wifi_profile(self) -> None:
        """Clear interface-name bindings on Wi‑Fi profiles and try reconnect.

        This fixes cases where a profile is pinned to the wrong interface
        (e.g., eth0) and prevents reconnection.
        """
        self._append("AUTO", "Fixing Wi‑Fi profile interface bindings…")
        # Get all Wi‑Fi connections
        rc, txt = self._run_and_capture(["nmcli", "-t", "-f", "NAME,TYPE", "connection", "show"], "AUTO")
        wifi_names: list[str] = []
        if rc == 0:
            for line in txt.splitlines():
                parts = line.split(":")
                if len(parts) >= 2 and parts[1].strip().lower() in ("wifi", "802-11-wireless"):
                    wifi_names.append(parts[0])
        changed = 0
        for name in wifi_names:
            rc, out = self._run_and_capture(["nmcli", "-g", "connection.interface-name", "connection", "show", "id", name], "AUTO")
            iface = (out or "").strip()
            if iface:
                self._append("AUTO", f"Clearing interface-name '{iface}' on '{name}'")
                self._run_cmd(["nmcli", "connection", "modify", "id", name, "connection.interface-name", ""], "AUTO")
                changed += 1
        if changed == 0:
            self._append("AUTO", "No Wi‑Fi profiles required changes.")
        # Attempt to reconnect previous Wi‑Fi connections
        self._reconnect_previous_connections()

    def edit_thresholds(self) -> None:
        """Edit processing thresholds and distance params in config.yaml."""
        try:
            cfg = yaml.safe_load(open(DEFAULT_CFG)) if (yaml and DEFAULT_CFG.exists()) else {}
        except Exception as exc:
            messagebox.showerror("Edit Thresholds", f"Failed to load config: {exc}")
            return
        def _ask_float(title: str, key: str, default: float) -> float | None:
            val = cfg.get(key, default)
            try:
                return simpledialog.askfloat(title, f"{key}", initialvalue=float(val))
            except Exception:
                return None
        updates = {}
        for (key, default) in (
            ("variance_threshold", 5.0),
            ("pca_threshold", 1.0),
            ("rssi_delta", 2.0),
            ("dat_rssi_offset", -60.0),
            ("tx_power_dbm", -40.0),
            ("path_loss_exponent", 2.0),
        ):
            v = _ask_float("Edit Thresholds", key, default)
            if v is None:
                continue
            updates[key] = float(v)
        if not updates:
            return
        try:
            cfg.update(updates)
            with open(DEFAULT_CFG, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            self._append("INFO", f"Updated config: {', '.join(f'{k}={v}' for k,v in updates.items())}")
            # Refresh calibration label if toggled
            self._update_calibration_status()
        except Exception as exc:
            messagebox.showerror("Edit Thresholds", f"Failed to write config: {exc}")

    def calibrate_distance(self) -> None:
        """Run distance calibration helper from the GUI."""
        # Choose two logs
        path1 = filedialog.askopenfilename(title="Select first log (at distance d1)", filetypes=[("JSONL", ".log"), ("All files", "*.*")])
        if not path1:
            return
        d1 = simpledialog.askfloat("Calibrate Distance", "Enter distance d1 (meters)", initialvalue=1.0)
        if not d1:
            return
        path2 = filedialog.askopenfilename(title="Select second log (at distance d2)", filetypes=[("JSONL", ".log"), ("All files", "*.*")])
        if not path2:
            return
        d2 = simpledialog.askfloat("Calibrate Distance", "Enter distance d2 (meters)", initialvalue=3.0)
        if not d2:
            return
        cmd = [sys.executable, "-m", "csi_node.calibrate", "--log1", path1, "--d1", str(d1), "--log2", path2, "--d2", str(d2), "--config", str(DEFAULT_CFG)]
        self._append("CAL", f"Running: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            out, err = proc.communicate()
            for line in (out or "").splitlines():
                self._append("CAL", line)
            for line in (err or "").splitlines():
                self._append("CAL", line)
            if proc.returncode == 0:
                self._append("CAL", "Calibration complete and written to config.")
                self._update_calibration_status()
            else:
                self._append("CAL", f"Calibration exited with code {proc.returncode}")
        except Exception as exc:
            messagebox.showerror("Calibrate Distance", f"Failed to run calibrator: {exc}")

    def _update_calibration_status(self) -> None:
        if not yaml or not DEFAULT_CFG.exists():
            self.calib_status.set("Calibrated: unknown")
            return
        try:
            cfg = yaml.safe_load(open(DEFAULT_CFG)) or {}
            calib = bool(cfg.get("calibrated", False))
            when = cfg.get("calibrated_at", "")
            self.calib_status.set(f"Calibrated: {'yes' if calib else 'no'} {when}")
        except Exception:
            self.calib_status.set("Calibrated: unknown")

    def apply_through_wall_preset(self) -> None:
        """Set parameters suited for through-wall demos."""
        try:
            # UI controls
            self.channel.set(1)
            self.width.set(20)
            self.dat_mode.set(True)
            self.dat_rssi_offset.set(-60.0)
            # Window override for steadier output
            self._window_override = 2.5
            # Persist rssi_delta in config for direction stability
            if yaml and DEFAULT_CFG.exists():
                cfg = yaml.safe_load(open(DEFAULT_CFG)) or {}
                cfg["rssi_delta"] = 3.5
                with open(DEFAULT_CFG, "w") as f:
                    yaml.safe_dump(cfg, f, sort_keys=False)
            self._append("INFO", "Applied Through‑Wall preset: ch1/20MHz, RSSI offset −60, rssi_delta=3.5, window=2.5s")
        except Exception as exc:
            self._append("INFO", f"Failed to apply preset: {exc}")

    def show_tracking_window(self) -> None:
        if self._tracking_win and self._tracking_win.winfo_exists():
            try:
                self._tracking_win.lift()
                return
            except Exception:
                pass
        win = Toplevel(self.root)
        win.title("Live Tracking")
        win.geometry("360x200")
        container = ttk.Frame(win)
        container.pack(fill="both", expand=True, padx=12, pady=12)
        self._trk_presence = StringVar(value="Presence: —")
        self._trk_direction = StringVar(value="Direction: —")
        self._trk_distance = StringVar(value="Distance: — m")
        self._trk_conf = StringVar(value="Confidence: —")
        ttk.Label(container, textvariable=self._trk_presence, font=("TkDefaultFont", 14, "bold")).pack(anchor="w", pady=4)
        ttk.Label(container, textvariable=self._trk_direction, font=("TkDefaultFont", 12)).pack(anchor="w", pady=4)
        ttk.Label(container, textvariable=self._trk_distance, font=("TkDefaultFont", 12)).pack(anchor="w", pady=4)
        ttk.Label(container, textvariable=self._trk_conf, font=("TkDefaultFont", 12)).pack(anchor="w", pady=4)
        self._tracking_win = win
        if self._last_entry:
            self._update_tracking_window(self._last_entry)

    def _update_tracking_window(self, obj: dict) -> None:
        try:
            if not (self._tracking_win and self._tracking_win.winfo_exists()):
                return
        except Exception:
            return
        pres = bool(obj.get("presence"))
        dirn = obj.get("direction", "?")
        dist = obj.get("distance_m", float("nan"))
        conf = obj.get("confidence", float("nan"))
        self._trk_presence.set(f"Presence: {'YES' if pres else 'NO'}")
        self._trk_direction.set(f"Direction: {dirn}")
        try:
            dtxt = f"{float(dist):.2f} m" if dist == dist else "—"
        except Exception:
            dtxt = "—"
        self._trk_distance.set(f"Distance: {dtxt}")
        try:
            ctxt = f"{float(conf):.2f}"
        except Exception:
            ctxt = "—"
        self._trk_conf.set(f"Confidence: {ctxt}")

    def _check_passwordless(self) -> bool:
        """Return True if `sudo -n` works for at least our needed commands.

        Some systems only allow a curated set of commands via sudoers. Prefer a
        generic `sudo -n true`, but if that fails, probe a shortlist of
        frequently used commands (rfkill, modprobe, iw, ip, systemctl, nmcli).
        """
        try:
            if shutil.which("sudo"):
                # Generic check
                proc = subprocess.run(["sudo", "-n", "true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                if proc.returncode == 0:
                    return True
                # Targeted probes
                candidates = [
                    ["rfkill", "--help"],
                    ["modprobe", "-h"],
                    ["iw", "--help"],
                    ["ip", "-V"],
                    ["systemctl", "--version"],
                    ["nmcli", "--help"],
                ]
                for cmd in candidates:
                    path = shutil.which(cmd[0])
                    if not path:
                        continue
                    try:
                        p = subprocess.run(["sudo", "-n", path] + cmd[1:], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
                        if p.returncode == 0:
                            return True
                    except Exception:
                        continue
        except Exception:
            pass
        return False

    def _start_capture_with_fallback(self) -> bool:
        # Stop any previous capture
        if self.cap_plog:
            try:
                self.cap_plog.stop()
            finally:
                self.cap_plog = None

        ch = int(self.channel.get())
        coding = self.coding.get().upper()
        requested = int(self.width.get()) if self.width.get() > 0 else 80
        widths: list[int] = []
        for w in (requested, 80, 40, 20):
            if w not in widths and w > 0:
                widths.append(w)
        freq = self._channel_to_freq(ch)
        self._append("FEIT", f"Interface prep complete. Trying widths: {widths}")

        log_path = DATA_DIR / "csi_raw.log"
        # Remove stale 0-byte files
        try:
            if log_path.exists() and log_path.stat().st_size == 0:
                log_path.unlink()
        except Exception:
            pass

        for w in widths:
            if self.cap_plog:
                try:
                    self.cap_plog.stop()
                finally:
                    self.cap_plog = None
            cmd = [
                "/usr/local/bin/feitcsi",
                "-f", str(freq),
                "-w", str(w),
                "--coding", coding,
                "-o", str(log_path),
                "-v",
            ]
            self._append("FEIT", f"Attempting: {' '.join(cmd)}")
            try:
                proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            except FileNotFoundError:
                # Fallback to script if binary not found
                cap_cmd = ["bash", str(SCRIPTS_DIR / "10_csi_capture.sh"), str(ch), str(w), coding]
                self._append("FEIT", f"Binary not found; using script: {' '.join(cap_cmd)}")
                proc = subprocess.Popen(cap_cmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
            self.cap_plog = ProcessLogger("FEIT", proc, self.out_q)
            self.cap_plog.start()
            # Wait up to 6s for log to appear. If it shows up but the process
            # exits immediately (e.g., debugfs permission denied), escalate to
            # a privileged attempt before declaring success.
            if self._wait_for_file(log_path, 6.0):
                # Give the process a brief window to prove it's stable.
                self._append("FEIT", f"Log detected: {log_path} (checking stability)")
                time.sleep(1.0)
                try:
                    rc = proc.poll()
                except Exception:
                    rc = None
                if rc is None:
                    # Still running: accept this attempt.
                    return True
                else:
                    self._append("FEIT", f"Capture exited early (rc={rc}); retrying with privileges…")
                    try:
                        self.cap_plog.stop()
                    finally:
                        self.cap_plog = None

            # Try elevated capture via pkexec/sudo (either no log or early exit)
            try:
                if shutil.which("sudo"):
                    pcmd = ["sudo", "-n"] + cmd
                else:
                    pcmd = ["pkexec"] + cmd
                self._append("FEIT", f"Attempting privileged: {' '.join(pcmd)}")
                pproc = subprocess.Popen(pcmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                # Replace logger with new proc
                try:
                    if self.cap_plog:
                        self.cap_plog.stop()
                except Exception:
                    pass
                self.cap_plog = ProcessLogger("FEIT", pproc, self.out_q)
                self.cap_plog.start()
                if self._wait_for_file(log_path, 6.0):
                    self._append("FEIT", f"Log detected (privileged): {log_path}")
                    # Brief stability check again
                    time.sleep(1.0)
                    if pproc.poll() is None:
                        return True
                    else:
                        self._append("FEIT", f"Privileged capture exited early (rc={pproc.returncode}); trying next width…")
            except Exception as exc:
                self._append("FEIT", f"Privileged attempt failed: {exc}")

            self._append("FEIT", f"No stable log after attempt at {w} MHz; restarting capture…")
            try:
                if self.cap_plog:
                    self.cap_plog.stop()
            finally:
                self.cap_plog = None
        return False

    def _on_close(self) -> None:
        try:
            self.stop()
        finally:
            self.root.destroy()


def main() -> None:
    root = Tk()
    # Prefer platform theme where available
    try:
        style = ttk.Style()
        if sys.platform.startswith("linux"):
            style.theme_use("clam")
    except Exception:
        pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
