"""VoxGuard: Deepfake Audio Detection Package.

A modular research prototype distinguishing genuine human speech from synthetic
and voice-converted audio using traditional ML and deep convolutional architectures.
"""

import importlib
import sys

__version__ = "0.1.0"

# Register backwards-compatible alias so any external tool referencing `voxguard` resolves to `src`
sys.modules["voxguard"] = sys.modules[__name__]


class _VoxguardFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "voxguard":
            mod = sys.modules.get("src")
            if mod is not None:
                return getattr(mod, "__spec__", None)
        if fullname.startswith("voxguard."):
            src_name = "src." + fullname[len("voxguard.") :]
            try:
                mod = importlib.import_module(src_name)
                sys.modules[fullname] = mod
                return getattr(mod, "__spec__", None)
            except Exception:
                return None
        return None


if not any(isinstance(f, type) and f.__name__ == "_VoxguardFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _VoxguardFinder)
