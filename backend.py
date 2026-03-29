import numpy as np
from importlib import import_module

try:
    cp = import_module("cupy")
except ImportError:
    cp = None


def gpu_available() -> bool:
    return cp is not None


def ensure_device_available(device: str):
    if device == "gpu" and not gpu_available():
        raise RuntimeError("cupy is not available, cannot use GPU device")


def xp_for_device(device: str):
    ensure_device_available(device)
    return cp if device == "gpu" else np


def infer_device_from_value(value) -> str:
    module_name = type(value).__module__
    if module_name.startswith("cupy"):
        return "gpu"
    return "cpu"


def asarray_on_device(value, device: str):
    return xp_for_device(device).asarray(value)

class Backend:
    xp = np

    @staticmethod
    def use_gpu():
        ensure_device_available("gpu")
        Backend.xp = cp
    
    @staticmethod
    def use_cpu():
        Backend.xp = np

def xp():
    return Backend.xp