#!/usr/bin/env python3
"""Deploy, audit, back up, verify, and recover the Numerai runtime."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.core.runtime_manager import (  # noqa: E402
    RuntimeManager,
    RuntimeManagerError,
)


DEFAULT_RUNTIME = Path(
    os.environ.get(
        "NUMERAI_RUNTIME_BACKEND",
        "~/Library/Application Support/Numerai/runtime/backend",
    )
).expanduser()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Governed Numerai runtime deployment and recovery."
    )
    parser.add_argument(
        "--source",
        default=str(BACKEND_ROOT),
        help="Reviewed source backend directory.",
    )
    parser.add_argument(
        "--runtime",
        default=str(DEFAULT_RUNTIME),
        help="Canonical unattended runtime backend directory.",
    )
    parser.add_argument("--backup-dir", help="Override backup directory.")
    parser.add_argument("--strict", action="store_true")
    subcommands = parser.add_subparsers(dest="command", required=True)

    subcommands.add_parser("audit", help="Read-only source/runtime drift audit.")

    deploy = subcommands.add_parser("deploy", help="Test and deploy managed code.")
    deploy.add_argument("--skip-tests", action="store_true")
    deploy.add_argument("--python", help="Python executable used for source tests.")

    backup = subcommands.add_parser(
        "backup-state",
        help="Create and immediately verify a credential-free state backup.",
    )
    backup.add_argument(
        "--keep",
        type=int,
        default=7,
        help="Retain this many newest state backups.",
    )

    verify = subcommands.add_parser(
        "verify-backup",
        help="Verify archive, sidecar, manifest, paths, and member checksums.",
    )
    verify.add_argument("--backup", required=True)

    restore = subcommands.add_parser(
        "restore-state",
        help="Restore into staging or, with exact confirmation, the live runtime.",
    )
    restore.add_argument("--backup", required=True)
    restore.add_argument("--destination", required=True)
    restore.add_argument("--confirmation")

    rollback = subcommands.add_parser(
        "rollback-code",
        help="Restore a prior managed-code backup with exact confirmation.",
    )
    rollback.add_argument("--backup", required=True)
    rollback.add_argument("--confirmation", required=True)
    return parser


def _run(args: argparse.Namespace) -> dict[str, Any]:
    manager = RuntimeManager(
        source_root=args.source,
        runtime_root=args.runtime,
        backup_dir=args.backup_dir,
    )
    if args.command == "audit":
        return manager.audit()
    if args.command == "deploy":
        return manager.deploy(
            run_tests=not args.skip_tests,
            python_path=args.python,
        )
    if args.command == "backup-state":
        return manager.create_state_backup(keep=args.keep)
    if args.command == "verify-backup":
        return manager.verify_state_backup(args.backup)
    if args.command == "restore-state":
        return manager.restore_state_backup(
            args.backup,
            destination=args.destination,
            confirmation=args.confirmation,
        )
    if args.command == "rollback-code":
        return manager.rollback_code(
            args.backup,
            confirmation=args.confirmation,
        )
    raise RuntimeManagerError(f"Unknown command: {args.command}")


def main() -> int:
    args = _parser().parse_args()
    try:
        result = _run(args)
    except Exception as exc:
        result = {
            "ok": False,
            "status": "RUNTIME_OPERATION_FAILED",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        }
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    if args.strict and (
        not result.get("ok")
        or result.get("status") == "RUNTIME_DRIFT"
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
