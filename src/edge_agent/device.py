from __future__ import annotations

import os
from pathlib import Path

from .models import DeviceProfile



def detect_device(name: str = "auto") -> DeviceProfile:
    cpu_cores = os.cpu_count() or 1
    ram_mb = _read_meminfo_mb(default=4096)

    if name != "auto":
        return DeviceProfile(
            name=name,
            cpu_cores=cpu_cores,
            ram_mb=ram_mb,
            has_gpu=False,
            supports_fp16=cpu_cores >= 4,
            supports_int8=True,
        )

    return DeviceProfile(
        name="linux-edge",
        cpu_cores=cpu_cores,
        ram_mb=ram_mb,
        has_gpu=False,
        supports_fp16=cpu_cores >= 4,
        supports_int8=True,
    )



def _read_meminfo_mb(default: int) -> int:
    p = Path("/proc/meminfo")
    if not p.exists():
        return default

    for line in p.read_text().splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2 and parts[1].isdigit():
                return int(parts[1]) // 1024

    return default
