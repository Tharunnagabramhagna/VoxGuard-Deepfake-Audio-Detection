"""VoxGuard backward-compatibility shim.

This module ensures that any legacy external scripts or imports expecting
`import voxguard` or `from voxguard.x import y` redirect seamlessly to the
canonical `src` package without duplication or attribute divergence.
"""

import importlib
from importlib.machinery import ModuleSpec
import sys

import src

sys.modules["voxguard"] = src


class _AliasLoader:
    def __init__(self, mod):
        self.mod = mod

    def create_module(self, spec):
        return self.mod

    def exec_module(self, module):
        pass


class _VoxguardFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if fullname == "voxguard":
            return ModuleSpec("voxguard", _AliasLoader(src))
        if fullname.startswith("voxguard."):
            src_name = "src." + fullname[len("voxguard.") :]
            try:
                mod = importlib.import_module(src_name)
                return ModuleSpec(fullname, _AliasLoader(mod))
            except Exception:
                return None
        return None


if not any(isinstance(f, type) and f.__name__ == "_VoxguardFinder" for f in sys.meta_path):
    sys.meta_path.insert(0, _VoxguardFinder)

