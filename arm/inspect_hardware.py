"""Capture read-only encoder/health evidence for A6; never authorizes or performs motion."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from arm.driver import inspect_hardware


def main(argv: list[str] | None = None) -> int:
    from dotenv import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--follower-port", default=os.environ.get("FOLLOWER_PORT"))
    parser.add_argument("--leader-port", default=os.environ.get("LEADER_PORT"))
    parser.add_argument("--out", type=Path, required=True, help="new JSON report path (never overwritten)")
    args = parser.parse_args(argv)
    ports = {"follower": args.follower_port, "leader": args.leader_port}
    if not any(ports.values()):
        parser.error("set --follower-port and/or --leader-port (or their environment variables)")
    if len({p.casefold() for p in ports.values() if p}) != sum(bool(p) for p in ports.values()):
        parser.error("leader and follower must use different ports")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the evidence path before touching any hardware. Keep errors in the report too.
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        reports = {}
        failed = False
        for role, port in ports.items():
            if not port:
                continue
            try:
                report = inspect_hardware(port=port, role=role)
                reports[role] = report
                matched = report["calibration"]["matches_hardware"]
                failed |= not matched or bool(report["issues"])
                print(f"{role} {port}: six motors read; encoder calibration match={matched}")
                for issue in report["issues"]:
                    print(f"  {issue}")
            except Exception as exc:
                failed = True
                reports[role] = {"port": port, "error": f"{type(exc).__name__}: {exc}"}
                print(f"{role} {port}: read failed ({type(exc).__name__})")
        json.dump({"read_only": True, "arms": reports}, output, indent=2, allow_nan=False)
        output.write("\n")
    print(f"Saved {args.out}. A6 physical joint-map measurements remain required before Cartesian replay.")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
