"""Windows wrapper for run.py / self_check.py (does NOT modify the organizers' files).

The organizers' harness uses Linux-only calls (os.setsid, os.killpg, os.getuid,
signal.SIGKILL). This file defines harmless stand-ins, then runs the normal harness.

Usage:
    python run_windows.py --adapter adapters.mine:MySolver --out report.json
    python run_windows.py --self-check --adapter adapters.mine:MySolver
"""
import os
import signal
import sys

# Top level on purpose: the harness's child process re-imports this file, so the
# stand-ins must exist there too.
if not hasattr(os, "setsid"):
    os.setsid = lambda: None
if not hasattr(os, "getuid"):
    os.getuid = lambda: 1000
if not hasattr(signal, "SIGKILL"):
    signal.SIGKILL = signal.SIGTERM
if not hasattr(os, "killpg"):
    os.killpg = lambda pid, sig: os.kill(pid, signal.SIGTERM)

import _paths  # noqa: F401,E402

if __name__ == "__main__":
    import metrics
    if "--self-check" in sys.argv:
        sys.argv.remove("--self-check")
        from benchkit.cli import self_check_main
        raise SystemExit(self_check_main(metrics.BENCH))
    from benchkit.cli import run_main
    raise SystemExit(run_main(metrics.BENCH))
