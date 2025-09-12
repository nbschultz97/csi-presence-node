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
from tkinter import Tk, StringVar, IntVar, DoubleVar, BooleanVar, END, Menu, simpledialog
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
        self.replay_path = StringVar(value=str((REPO_ROOT / "data" / "sample_csi.b64")))
        self.speed = DoubleVar(value=1.0)
        self.output_file = StringVar(value=str(Path(cfg.get("output_file", "data/presence_log.jsonl"))))
        self.autoscroll = BooleanVar(value=True)
        self.pwless_status = StringVar(value="Passwordless: unknown")

        self.cap_plog: ProcessLogger | None = None
        self.pipe_plog: ProcessLogger | None = None
        self.out_q: queue.Queue = queue.Queue()
        self._workflow_thread: threading.Thread | None = None
        self._prev_active_cons: list[tuple[str, str]] = []
        self._wifi_devs: list[str] = []

        self._build_ui()
        self._schedule_pump()

    def _build_ui(self) -> None:
        # Menu bar (Tools -> Diagnostics / Setup Passwordless)
        menubar = Menu(self.root)
        tools = Menu(menubar, tearoff=0)
        tools.add_command(label="Diagnostics", command=self.run_diagnostics)
        tools.add_separator()
        tools.add_command(label="Setup Passwordless sudo…", command=self.setup_passwordless)
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
            ok = self._start_capture_with_fallback()
            if not ok:
                self._append("FEIT", "All capture attempts failed. See logs above.")
                self.stop()
                return

            # Wait until log file exists and is non-empty
            if not self._wait_for_file(REPO_ROOT / "data" / "csi_raw.log", 10.0):
                self._append("FEIT", "Log not created after capture start; aborting.")
                self.stop()
                return

            pipe_cmd = [sys.executable, str(REPO_ROOT / "run.py"), "--out", out]
            if pose_flag:
                pipe_cmd.append(pose_flag)
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
        if self.cap_plog:
            self._append("FEIT", "Stopping capture…")
            self.cap_plog.stop()
            self.cap_plog = None
        # Bring networking back up and try to reconnect saved Wi‑Fi
        self._restart_networking()
        self._reconnect_previous_connections()
        self.btn_start.config(state="normal")
        self.btn_stop.config(state="disabled")
        self._workflow_thread = None

    def run_diagnostics(self) -> None:
        t = threading.Thread(target=self._run_diagnostics, daemon=True)
        t.start()

    def _run_and_capture(self, cmd: list[str], tag: str, privileged: bool = False, timeout: float | None = None) -> tuple[int, str]:
        full_cmd = cmd
        if privileged:
            # Prefer sudo -n (works with passwordless sudo); fallback to pkexec
            if shutil.which("sudo"):
                full_cmd = ["sudo", "-n"] + cmd
            elif shutil.which("pkexec"):
                full_cmd = ["pkexec"] + cmd
            else:
                full_cmd = ["sudo", "-n"] + cmd
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
            if shutil.which("pkexec"):
                full_cmd = ["pkexec"] + cmd
            else:
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
        # 7) Verify iwlwifi debugfs exists (check as root to avoid PermissionError)
        dbg_path = "/sys/kernel/debug/iwlwifi"
        rc = self._run_cmd(["bash", "-lc", f"test -e {dbg_path}"], "AUTO", privileged=True)
        if rc != 0:
            self._append("AUTO", "iwlwifi debugfs not found after mount/link. Ensure FeitCSI iwlwifi module is loaded.")
        else:
            self._append("AUTO", "iwlwifi debugfs present.")

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
        # Try systemd restart, fallback to service
        rc = self._run_cmd(["systemctl", "restart", "NetworkManager"], "AUTO", privileged=True)
        if rc != 0:
            self._run_cmd(["service", "NetworkManager", "restart"], "AUTO", privileged=True)
        # Ensure networking/wifi are on
        self._run_cmd(["nmcli", "networking", "on"], "AUTO")
        self._run_cmd(["nmcli", "radio", "wifi", "on"], "AUTO")
        time.sleep(2.0)

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
                self._run_cmd(["nmcli", "connection", "up", "id", name], "AUTO")
        self._prev_active_cons = []

    def _run_root_preflight(self) -> None:
        """Run all privileged preflight actions in one pkexec call.

        Falls back to per-step commands if pkexec is unavailable or fails.
        """
        script = REPO_ROOT / "scripts" / "preflight_root.sh"
        if script.exists():
            cmd = ["sudo", "-n", "bash", str(script)] if shutil.which("sudo") else ["pkexec", "bash", str(script)]
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

        # Fallback discrete steps
        self._run_cmd(["rfkill", "unblock", "all"], "AUTO", privileged=True)
        self._run_cmd(["modprobe", "iwlwifi"], "AUTO", privileged=True)
        self._run_cmd(["iw", "reg", "set", "US"], "AUTO", privileged=True)
        self._run_cmd(["mountpoint", "-q", "/sys/kernel/debug"], "AUTO")
        self._run_cmd(["mount", "-t", "debugfs", "debugfs", "/sys/kernel/debug"], "AUTO", privileged=True)
        # Try to link ieee80211/.../iwlwifi to /sys/kernel/debug/iwlwifi
        self._run_cmd(["bash", "-lc", "test -d /sys/kernel/debug/iwlwifi || ln -s $(find /sys/kernel/debug/ieee80211 -maxdepth 3 -type d -name iwlwifi | head -n1) /sys/kernel/debug/iwlwifi 2>/dev/null || true"], "AUTO", privileged=True)
        self._run_cmd(["/usr/sbin/setcap", "cap_net_admin,cap_net_raw+eip", "/usr/local/bin/feitcsi"], "AUTO", privileged=True)
        self._run_cmd(["setcap", "cap_net_admin,cap_net_raw+eip", "/usr/local/bin/feitcsi"], "AUTO", privileged=True)

    def _detect_wifi_devices(self) -> list[str]:
        # Prefer nmcli (stable output). Fallback to iw if needed.
        rc, txt = self._run_and_capture(["nmcli", "-t", "-f", "DEVICE,TYPE,STATE", "dev", "status"], "AUTO")
        devs: list[str] = []
        for line in txt.splitlines():
            if not line or ":" not in line:
                continue
            parts = line.split(":")
            if len(parts) < 3:
                continue
            dev, typ, state = parts[0], parts[1], parts[2]
            if typ.strip().lower() == "wifi":
                devs.append(dev)
        if devs:
            return devs
        # Fallback to iw dev parsing
        rc, txt = self._run_and_capture(["iw", "dev"], "AUTO")
        cur = None
        for line in txt.splitlines():
            line = line.strip()
            if line.startswith("Interface "):
                cur = line.split()[1]
            elif line.startswith("type ") and cur:
                typ = line.split()[1]
                if typ in ("managed", "station"):
                    devs.append(cur)
                cur = None
        return list(dict.fromkeys(devs))  # dedupe

    def _disconnect_wifi_devices(self) -> None:
        devs = self._detect_wifi_devices()
        self._wifi_devs = devs
        sel = self.wifi_device.get().strip()
        target = sel if sel in devs else (devs[0] if devs else "")
        if target:
            if sel and sel != target:
                self.wifi_device.set(target)
            self._append("AUTO", f"Disconnecting device {target}…")
            self._run_cmd(["nmcli", "dev", "disconnect", target], "AUTO")
            # Ensure interface stays up
            self._run_cmd(["ip", "link", "set", target, "up"], "AUTO", privileged=True)
        else:
            # Fallback: radio off
            self._append("AUTO", "No Wi‑Fi device found via nmcli/iw; falling back to radio off")
            self._run_cmd(["nmcli", "radio", "wifi", "off"], "AUTO")

    def _populate_wifi_devices(self) -> None:
        devs = self._detect_wifi_devices()
        try:
            self.dev_combo["values"] = devs
        except Exception:
            pass
        if devs and not self.wifi_device.get():
            self.wifi_device.set(devs[0])

    def _update_pwless_status(self) -> None:
        ok = self._check_passwordless()
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

    def _check_passwordless(self) -> bool:
        # Try a harmless allowed command via sudo -n; if it succeeds, we’re good
        candidates = ["/usr/sbin/rfkill", "/usr/bin/mount", "/usr/sbin/iw"]
        for path in candidates:
            if Path(path).exists():
                rc = self._run_cmd([path, "--help"], "AUTO", privileged=True)
                if rc == 0:
                    return True
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

            # Wait up to 6s for log to appear, else try privileged capture then retry next width
            if self._wait_for_file(log_path, 6.0):
                self._append("FEIT", f"Log detected: {log_path}")
                return True
            else:
                # Try elevated capture via pkexec/sudo
                try:
                    if shutil.which("sudo"):
                        pcmd = ["sudo", "-n"] + cmd
                    else:
                        pcmd = ["pkexec"] + cmd
                    self._append("FEIT", f"No log at {w} MHz; attempting privileged: {' '.join(pcmd)}")
                    pproc = subprocess.Popen(pcmd, cwd=str(REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1)
                    # Replace logger with new proc
                    try:
                        self.cap_plog.stop()
                    except Exception:
                        pass
                    self.cap_plog = ProcessLogger("FEIT", pproc, self.out_q)
                    self.cap_plog.start()
                    if self._wait_for_file(log_path, 6.0):
                        self._append("FEIT", f"Log detected (privileged): {log_path}")
                        return True
                except Exception as exc:
                    self._append("FEIT", f"Privileged attempt failed: {exc}")
                self._append("FEIT", f"No log after attempt at {w} MHz; restarting capture…")
                try:
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
