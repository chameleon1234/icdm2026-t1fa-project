from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class ProcessInfo:
    pid: int
    name: str
    ram_gb: float
    path: str


def _run_powershell(command: str) -> str:
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip()


def get_memory_status() -> tuple[float, float]:
    command = (
        "Get-CimInstance Win32_OperatingSystem | "
        "ForEach-Object { '{0:F2},{1:F2}' -f ($_.TotalVisibleMemorySize/1MB), ($_.FreePhysicalMemory/1MB) }"
    )
    output = _run_powershell(command)
    if not output:
        return 0.0, 0.0
    total, free = output.splitlines()[-1].split(",", 1)
    return float(total), float(free)


def get_python_processes() -> list[ProcessInfo]:
    command = (
        "Get-Process python* -ErrorAction SilentlyContinue | "
        "Sort-Object WorkingSet64 -Descending | "
        "ForEach-Object { '{0}|{1}|{2:F3}|{3}' -f $_.Id,$_.ProcessName,($_.WorkingSet64/1GB),$_.Path }"
    )
    rows = []
    for line in _run_powershell(command).splitlines():
        parts = line.split("|", 3)
        if len(parts) != 4:
            continue
        rows.append(ProcessInfo(int(parts[0]), parts[1], float(parts[2]), parts[3]))
    return rows


def get_nvidia_smi() -> str:
    if shutil.which("nvidia-smi") is None:
        return "nvidia-smi not found"
    completed = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=False)
    return completed.stdout.strip() or completed.stderr.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check whether the machine is safe to start a GPU/RAM-heavy training run.")
    parser.add_argument("--min_free_ram_gb", type=float, default=6.0)
    parser.add_argument("--warn_python_ram_gb", type=float, default=2.0)
    args = parser.parse_args()

    total_gb, free_gb = get_memory_status()
    used_gb = max(total_gb - free_gb, 0.0)
    print(f"RAM: total={total_gb:.2f}GB used={used_gb:.2f}GB free={free_gb:.2f}GB")
    if free_gb < args.min_free_ram_gb:
        print(f"WARNING: free RAM is below {args.min_free_ram_gb:.1f}GB. Restart/close apps before full training.")

    processes = get_python_processes()
    if processes:
        print("\nPython processes:")
        for proc in processes:
            flag = "  <-- high RAM" if proc.ram_gb >= args.warn_python_ram_gb else ""
            print(f"  PID={proc.pid:<7} RAM={proc.ram_gb:>6.2f}GB  {proc.name}  {proc.path}{flag}")
    else:
        print("\nPython processes: none")

    print("\nNVIDIA status:")
    print(get_nvidia_smi())

    if free_gb < args.min_free_ram_gb:
        return 2
    if any(proc.ram_gb >= args.warn_python_ram_gb for proc in processes):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
