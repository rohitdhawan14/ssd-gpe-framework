#!/usr/bin/env python3
import argparse, subprocess
from pathlib import Path
from itertools import combinations

def parse_args():
    p = argparse.ArgumentParser(
        description="Run two-region simulations for every pair of regions."
    )
    p.add_argument("--sim-path", required=True,
                   help="Path to simulate_case_studyN_regions.py (or the 2-region variant).")
    p.add_argument("--pushes", required=True,
                   help="CSV of hourly pushes/runs (hour_utc,workflow_runs).")
    p.add_argument("--region", nargs=2, action="append", metavar=("NAME","CI_CSV"),
                   help="Repeat per region, e.g., --region GB entsoe_GB_2023_hourly_ci.csv",
                   required=True)
    p.add_argument("--outdir", required=True, help="Output root directory.")
    # policy knobs (defaults match what you’ve been using)
    p.add_argument("--window-hours", type=int, default=6)
    p.add_argument("--phi", type=float, default=0.5)
    p.add_argument("--cap-mult", type=float, default=2.0)
    p.add_argument("--geo-tau", type=float, default=0.15)
    p.add_argument("--opportunistic-phi", type=float, default=0.25)
    p.add_argument("--energy-per-push-kwh", type=float, default=0.1)
    return p.parse_args()

def main():
    args = parse_args()
    outroot = Path(args.outdir)
    outroot.mkdir(parents=True, exist_ok=True)

    # [(name, path), ...]
    regions = [(name, Path(path)) for name, path in args.region]
    if len(regions) < 2:
        raise SystemExit("Provide at least two --region NAME FILE.csv entries.")

    # Run for every unordered pair (A,B)
    for (nameA, pathA), (nameB, pathB) in combinations(regions, 2):
        pair_dir = outroot / f"{nameA}_{nameB}"
        pair_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            "python3", args.sim_path,
            "--pushes", args.pushes,
            "--region", nameA, str(pathA),
            "--region", nameB, str(pathB),
            "--outdir", str(pair_dir),
            "--window-hours", str(args.window_hours),
            "--phi", str(args.phi),
            "--cap-mult", str(args.cap_mult),
            "--geo-tau", str(args.geo_tau),
            "--opportunistic-phi", str(args.opportunistic_phi),
            "--energy-per-push-kwh", str(args.energy_per_push_kwh),
        ]
        print(f"[+] Running pair: {nameA} vs {nameB} → {pair_dir}")
        subprocess.run(cmd, check=True)

    print("[✓] Done pairwise portfolios.")

if __name__ == "__main__":
    main()
