"""Environment capture and resource guards."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from importlib import metadata

from .config import PIN_REQUIREMENTS


def rss_gb() -> float:
    import psutil
    return psutil.Process().memory_info().rss / 1e9


def host_ram_gb() -> dict:
    import psutil
    v = psutil.virtual_memory()
    return {"total_gb": v.total / 1e9, "available_gb": v.available / 1e9}


def free_disk_gb(path) -> float:
    return shutil.disk_usage(path).free / 1e9


def check_pins(strict=True) -> dict:
    got = {}
    bad = []
    for name, want in PIN_REQUIREMENTS.items():
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            have = None
        got[name] = have
        if have != want:
            bad.append(f"{name}: have {have}, need {want}")
    if bad and strict:
        raise RuntimeError("Dependency pins not satisfied (install, then RESTART the runtime): " + "; ".join(bad))
    return got


def gpu_info() -> dict:
    import torch
    info = {"torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
            "torchvision": _ver("torchvision")}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update(gpu_name=p.name, gpu_total_gb=p.total_memory / 1e9, compute_capability=f"{p.major}.{p.minor}",
                    bf16_supported=bool(torch.cuda.is_bf16_supported()), cudnn=torch.backends.cudnn.version())
        try:
            info["driver"] = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                            capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception:
            info["driver"] = None
    return info


def _ver(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def capture_environment() -> dict:
    freeze = sorted(f"{d.metadata['Name']}=={d.version}" for d in metadata.distributions())
    return {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(),
            "host_ram": host_ram_gb(), "gpu": gpu_info(), "pip_freeze": freeze,
            "env_flags": {k: os.environ.get(k) for k in ("COLAB_GPU", "KAGGLE_KERNEL_RUN_TYPE")}}


class Guard:
    """Stop (after checkpointing, which is per item) before host memory is exhausted."""

    def __init__(self, min_avail_gb=1.0, max_vram_frac=0.95):
        self.min_avail_gb, self.max_vram_frac = min_avail_gb, max_vram_frac
        self.peak_rss = 0.0
        self.peak_alloc = 0.0
        self.peak_reserved = 0.0

    def check(self):
        import torch
        self.peak_rss = max(self.peak_rss, rss_gb())
        if host_ram_gb()["available_gb"] < self.min_avail_gb:
            raise MemoryError("host RAM nearly exhausted; stopping safely. Completed items are checkpointed; "
                              "re-run the cell to resume (lower batch size / restart the runtime first).")
        if torch.cuda.is_available():
            self.peak_alloc = max(self.peak_alloc, torch.cuda.max_memory_allocated() / 1e9)
            self.peak_reserved = max(self.peak_reserved, torch.cuda.max_memory_reserved() / 1e9)
            tot = torch.cuda.get_device_properties(0).total_memory / 1e9
            if self.peak_reserved > self.max_vram_frac * tot:
                raise MemoryError("VRAM reservation >95% of device; lower the batch size (do not hide it with "
                                  "a smaller resolution or float16).")

    def summary(self) -> dict:
        return {"peak_rss_gb": self.peak_rss, "peak_vram_allocated_gb": self.peak_alloc,
                "peak_vram_reserved_gb": self.peak_reserved}
