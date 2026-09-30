"""Best-effort hardware summary for choosing a model tier.

Only *reads* system information: memory via OS APIs, GPU names and dedicated
memory via the Windows registry (the display adapter class key) or
``nvidia-smi`` on Linux. No PowerShell/WMI tools are launched, because some of
them write cache files into the user profile.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


@dataclass
class GpuInfo:
    name: str
    vram_gb: Optional[float] = None


@dataclass
class HardwareInfo:
    cpu_threads: int
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    gpus: List[GpuInfo] = field(default_factory=list)

    @property
    def best_vram_gb(self) -> Optional[float]:
        values = [g.vram_gb for g in self.gpus if g.vram_gb]
        return max(values) if values else None

    def to_dict(self) -> Dict:
        return asdict(self)


def _memory() -> tuple:
    if sys.platform == "win32":
        import ctypes

        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return status.ullTotalPhys / 2**30, status.ullAvailPhys / 2**30
        return None, None
    try:
        info = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, rest = line.partition(":")
                info[key] = int(rest.split()[0]) * 1024
        return info["MemTotal"] / 2**30, info.get("MemAvailable", 0) / 2**30
    except (OSError, KeyError, ValueError):
        pass
    if sys.platform == "darwin":
        try:
            total = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5).stdout)
            return total / 2**30, None
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return None, None


def _gpus_windows() -> List[GpuInfo]:
    import winreg

    base = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
    gpus: List[GpuInfo] = []
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, base)
    except OSError:
        return gpus
    with root:
        for i in range(64):
            try:
                sub = winreg.EnumKey(root, i)
            except OSError:
                break
            if not sub.isdigit():
                continue
            try:
                with winreg.OpenKey(root, sub) as key:
                    name = winreg.QueryValueEx(key, "DriverDesc")[0]
                    vram = None
                    for value_name in ("HardwareInformation.qwMemorySize", "HardwareInformation.MemorySize"):
                        try:
                            raw = winreg.QueryValueEx(key, value_name)[0]
                            if isinstance(raw, bytes):
                                raw = int.from_bytes(raw[:8], "little")
                            vram = int(raw) / 2**30
                            break
                        except OSError:
                            continue
                    if "basic display" not in str(name).lower() and "remote" not in str(name).lower():
                        gpus.append(GpuInfo(str(name), round(vram, 1) if vram else None))
            except OSError:
                continue
    return gpus


def _gpus_nvidia_smi() -> List[GpuInfo]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        out = subprocess.run([exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, _, mem = line.partition(",")
        try:
            gpus.append(GpuInfo(name.strip(), round(float(mem) / 1024, 1)))
        except ValueError:
            gpus.append(GpuInfo(name.strip()))
    return gpus


def detect() -> HardwareInfo:
    total, available = _memory()
    try:
        gpus = _gpus_windows() if sys.platform == "win32" else _gpus_nvidia_smi()
    except Exception:  # noqa: BLE001 - hardware probing must never break the app
        gpus = []
    return HardwareInfo(cpu_threads=os.cpu_count() or 1,
                        ram_total_gb=round(total, 1) if total else None,
                        ram_available_gb=round(available, 1) if available else None, gpus=gpus)


def recommend_tier(info: HardwareInfo) -> str:
    vram = info.best_vram_gb or 0
    ram = info.ram_total_gb or 0
    if vram >= 16:
        return "high"
    if vram >= 8 or ram >= 24:
        return "medium"
    return "low"


def fits(info: HardwareInfo, size_gb: float, vram_gb: float) -> str:
    """'gpu' when the model should fit in VRAM, 'cpu' when only system RAM is enough, 'no' otherwise."""

    if info.best_vram_gb and info.best_vram_gb >= vram_gb:
        return "gpu"
    if info.ram_total_gb and info.ram_total_gb >= size_gb * 1.3 + 2:
        return "cpu"
    if info.ram_total_gb is None:
        return "unknown"
    return "no"
