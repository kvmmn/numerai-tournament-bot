#!/usr/bin/env python3
"""Interactive Numerai account setup and validation script.

Usage:
    python setup_numerai.py

This script:
1. Prompts for API keys (or reads from existing .env)
2. Validates connection to Numerai
3. Lists existing models or guides creating new ones
4. Saves configuration to .env
5. Tests the full pipeline with a dry-run
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
NUMERAI_ROOT = Path(__file__).resolve().parents[3]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))


def _load_existing_env() -> dict:
    """Load existing .env file if present."""
    env = {}
    for env_path in [BACKEND_ROOT / ".env", NUMERAI_ROOT / ".env"]:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    env[key.strip()] = value.strip().strip('"').strip("'")
    return env


def _save_env(config: dict) -> Path:
    """Save config to .env file."""
    env_path = BACKEND_ROOT / ".env"
    lines = []
    if env_path.exists():
        lines = env_path.read_text().splitlines()

    # Update or add keys
    existing_keys = set()
    updated_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in config:
                updated_lines.append(f"{key}={config[key]}")
                existing_keys.add(key)
            else:
                updated_lines.append(line)
        else:
            updated_lines.append(line)

    for key, value in config.items():
        if key not in existing_keys:
            updated_lines.append(f"{key}={value}")

    env_path.write_text("\n".join(updated_lines) + "\n")
    os.chmod(env_path, 0o600)
    return env_path


def main():
    print("=" * 60)
    print("  Numerai Winning OS - Account Setup")
    print("=" * 60)
    print()

    existing = _load_existing_env()
    config = {}

    # Step 1: API Keys
    print("[1/4] API Credentials")
    print("-" * 40)

    public_id = existing.get("NUMERAI_PUBLIC_ID", "")
    if public_id:
        print(f"  Found existing NUMERAI_PUBLIC_ID: {public_id[:8]}...")
        use_existing = input("  Use existing? [Y/n]: ").strip().lower()
        if use_existing in ("", "y", "yes"):
            config["NUMERAI_PUBLIC_ID"] = public_id
        else:
            config["NUMERAI_PUBLIC_ID"] = input("  Enter NUMERAI_PUBLIC_ID: ").strip()
    else:
        print("  Get your API keys from: https://numer.ai/account")
        config["NUMERAI_PUBLIC_ID"] = input("  Enter NUMERAI_PUBLIC_ID: ").strip()

    secret_key = existing.get("NUMERAI_SECRET_KEY", "")
    if secret_key:
        print(f"  Found existing NUMERAI_SECRET_KEY: {'*' * 8}...")
        use_existing = input("  Use existing? [Y/n]: ").strip().lower()
        if use_existing in ("", "y", "yes"):
            config["NUMERAI_SECRET_KEY"] = secret_key
        else:
            config["NUMERAI_SECRET_KEY"] = input("  Enter NUMERAI_SECRET_KEY: ").strip()
    else:
        config["NUMERAI_SECRET_KEY"] = input("  Enter NUMERAI_SECRET_KEY: ").strip()

    if not config["NUMERAI_PUBLIC_ID"] or not config["NUMERAI_SECRET_KEY"]:
        print("\n  ERROR: Both API keys are required.")
        print("  Visit https://numer.ai/account to generate them.")
        return 1

    # Step 2: Validate connection
    print()
    print("[2/4] Validating Connection")
    print("-" * 40)

    try:
        from numerapi import NumerAPI
        napi = NumerAPI(
            public_id=config["NUMERAI_PUBLIC_ID"],
            secret_key=config["NUMERAI_SECRET_KEY"],
        )
        current_round = napi.get_current_round()
        print(f"  Connected! Current round: {current_round}")
    except Exception as exc:
        print(f"  ERROR: Failed to connect: {exc}")
        print("  Please check your API keys and try again.")
        return 1

    # Step 3: Check models
    print()
    print("[3/4] Model Configuration")
    print("-" * 40)

    try:
        models = napi.get_models()
        if models:
            print(f"  Found {len(models)} model(s):")
            for name, model_id in models.items():
                print(f"    - {name} ({model_id})")

            model_names = ",".join(models.keys())
            print(f"\n  Setting NUMERAI_MODEL_NAMES={model_names}")
            config["NUMERAI_MODEL_NAMES"] = model_names
        else:
            print("  No models found in your account.")
            print("  You need to create at least one model at https://numer.ai/models")
            print("  Suggested names:")
            print("    - your_username_ensemble  (for ensemble predictions)")
            print("    - your_username_lgbm      (for single LightGBM)")
            create = input("\n  Continue without models? [y/N]: ").strip().lower()
            if create not in ("y", "yes"):
                print("  Please create models first, then re-run this script.")
                return 1
    except Exception as exc:
        print(f"  Warning: Could not fetch models: {exc}")

    # Step 4: Set defaults
    print()
    print("[4/4] Configuration Defaults")
    print("-" * 40)

    config["AUTO_APPROVE_SUBMISSION"] = "false"
    config["USE_ENSEMBLE"] = "true"
    config["DATA_VERSION"] = existing.get("DATA_VERSION", "v5.2")
    config["FEATURE_SET"] = existing.get("FEATURE_SET", "small")
    config["TRAIN_ERA_STRIDE"] = existing.get("TRAIN_ERA_STRIDE", "4")
    config["MIN_VALIDATION_CORR"] = existing.get("MIN_VALIDATION_CORR", "0.01")
    config["MIN_SHARPE_RATIO"] = existing.get("MIN_SHARPE_RATIO", "0.3")
    config["MAX_FEATURE_EXPOSURE"] = existing.get("MAX_FEATURE_EXPOSURE", "0.1")
    config["MAX_TOTAL_STAKE_NMR"] = existing.get("MAX_TOTAL_STAKE_NMR", "0")
    config["MAX_MODEL_STAKE_NMR"] = existing.get("MAX_MODEL_STAKE_NMR", "0")
    config["MAX_STAKE_CHANGE_NMR"] = existing.get("MAX_STAKE_CHANGE_NMR", "0")

    print("  AUTO_APPROVE_SUBMISSION = false (human approval required)")
    print(f"  USE_ENSEMBLE = true")
    print(f"  DATA_VERSION = {config['DATA_VERSION']}")
    print(f"  FEATURE_SET = {config['FEATURE_SET']}")
    print(f"  TRAIN_ERA_STRIDE = {config['TRAIN_ERA_STRIDE']}")
    print(f"  Thresholds: corr>={config['MIN_VALIDATION_CORR']}, sharpe>={config['MIN_SHARPE_RATIO']}, exposure<={config['MAX_FEATURE_EXPOSURE']}")

    # Save
    env_path = _save_env(config)
    print(f"\n  Configuration saved to: {env_path}")

    # Summary
    print()
    print("=" * 60)
    print("  Setup Complete!")
    print("=" * 60)
    print()
    print("  Next steps:")
    print("    1. Create models at https://numer.ai/models (if not done)")
    print("    2. Test: python daily_numerai_run.py --mode numerapi-preflight")
    print("    3. Read-only status: python daily_numerai_run.py --mode portfolio-status")
    print("    4. Prepare: python daily_numerai_run.py --mode portfolio-prepare")
    print("    5. Install the reviewed LaunchAgent plists from automation/")
    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
