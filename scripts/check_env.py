"""Environment, dependency, and configuration check for VoxGuard.

Verifies the Python runtime, core scientific packages, PyTorch hardware
acceleration, YAML configuration parsing, and seed reproducibility.
"""

import sys
import platform
from pathlib import Path

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def check_environment() -> int:
    """Run diagnostic checks on environment and dependencies.

    Returns:
        0 if core requirements are met, 1 if critical components are missing.
    """
    print("=" * 70)
    print(" VoxGuard System & Dependency Diagnostics")
    print("=" * 70)

    # 1. System & Python Information
    print(f"\n[1] Runtime Information:")
    print(f"  - Platform: {platform.platform()}")
    print(f"  - Python Executable: {sys.executable}")
    print(f"  - Python Version: {sys.version.split()[0]} ({platform.architecture()[0]})")

    # 2. Package Import Checks
    core_packages = [
        ("numpy", "NumPy (linear algebra)"),
        ("scipy", "SciPy (scientific signal processing)"),
        ("yaml", "PyYAML (configuration loader)"),
        ("sklearn", "Scikit-Learn (classical ML & metrics)"),
        ("torch", "PyTorch (deep learning core)"),
    ]

    optional_audio_packages = [
        ("torchaudio", "TorchAudio (spectrograms & I/O)"),
        ("soundfile", "SoundFile (audio file reader/writer)"),
        ("librosa", "Librosa (acoustic analysis)"),
        ("pandas", "Pandas (protocol dataframes)"),
        ("streamlit", "Streamlit (web demo interface)"),
        ("matplotlib", "Matplotlib (plotting & EDA)"),
    ]

    print("\n[2] Core Dependency Status:")
    core_missing = []
    for module_name, desc in core_packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK]   {module_name.ljust(12)} {ver.ljust(12)} - {desc}")
        except ImportError as e:
            print(f"  [FAIL] {module_name.ljust(12)} {'MISSING'.ljust(12)} - {desc} ({e})")
            core_missing.append(module_name)

    print("\n[3] Extended / Audio Packages:")
    for module_name, desc in optional_audio_packages:
        try:
            mod = __import__(module_name)
            ver = getattr(mod, "__version__", "installed")
            print(f"  [OK]   {module_name.ljust(12)} {ver.ljust(12)} - {desc}")
        except ImportError as e:
            print(f"  [WARN] {module_name.ljust(12)} {'NOT FOUND'.ljust(12)} - {desc}")

    # 3. Hardware & PyTorch Acceleration
    print("\n[4] Hardware Acceleration Status:")
    try:
        import torch
        cuda_avail = torch.cuda.is_available()
        print(f"  - PyTorch CUDA Available: {cuda_avail}")
        if cuda_avail:
            print(f"  - CUDA Device Count: {torch.cuda.device_count()}")
            print(f"  - Active Device Name: {torch.cuda.get_device_name(0)}")
        else:
            print("  - Execution Device: CPU (Targeted for local development & smoke tests)")
    except ImportError:
        print("  - PyTorch not imported; cannot query acceleration hardware.")

    # 4. Configuration File Validation
    print("\n[5] Configuration File Validation:")
    config_path = Path("configs/default.yaml")
    if not config_path.exists():
        print(f"  [FAIL] configs/default.yaml NOT found at {config_path.resolve()}")
        return 1

    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        seed = cfg.get("seed", None)
        sr = cfg.get("audio", {}).get("sample_rate", None)
        models = list(cfg.get("models", {}).keys())
        print(f"  [OK]   Loaded {config_path} successfully.")
        print(f"  - Configured Seed: {seed}")
        print(f"  - Preprocessing Sample Rate: {sr} Hz")
        print(f"  - Registered Model Configurations: {', '.join(models)}")
    except Exception as e:
        print(f"  [FAIL] Error parsing configs/default.yaml: {e}")
        return 1

    # 5. Reproducibility Seed Check
    print("\n[6] Reproducibility Test:")
    try:
        from src.utils.seed import set_seed
        set_seed(42)
        import random
        v1 = random.random()
        set_seed(42)
        v2 = random.random()
        if abs(v1 - v2) < 1e-12:
            print(f"  [OK]   Seed determinism verified (v1={v1:.6f}, v2={v2:.6f}).")
        else:
            print(f"  [WARN] Seed determinism mismatch: v1={v1}, v2={v2}")
    except Exception as e:
        print(f"  [FAIL] Error running seed test: {e}")

    print("\n" + "=" * 70)
    if core_missing:
        print(f"DIAGNOSTIC RESULT: Missing core dependencies: {core_missing}")
        print("Please run: pip install -r requirements.txt")
        return 1
    else:
        print("DIAGNOSTIC RESULT: Environment is ready for Phase 2 data pipeline.")
        return 0


if __name__ == "__main__":
    sys.exit(check_environment())
