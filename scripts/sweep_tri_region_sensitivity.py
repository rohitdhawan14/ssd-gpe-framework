#!/usr/bin/env python3
"""
Sensitivity parameter sweep script for tri-region analysis.

This script runs the simulation with different parameter values to analyze
sensitivity of GPE framework to its four guardrail parameters (φ, D, κ, τ).

The simulator creates summary_caseN.csv and raw_totals_caseN.csv in each
output directory. The plot script will automatically find these files.

Usage:
    python3 sweep_tri_region_sensitivity.py \
        --pushes outputs/prep_out/gha_runs_2023_hourly.csv \
        --region GB outputs/prep_out/entsoe_GB_2023_hourly_ci.csv \
        --region DE outputs/prep_out/entsoe_DE_2023_hourly_ci.csv \
        --region FR outputs/prep_out/entsoe_FR_2023_hourly_ci.csv \
        --outdir outputs/out_sens_tri \
        --sim-path scripts/simulate_case_studyN_regions.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run parameter sensitivity sweeps for tri-region analysis"
    )
    
    parser.add_argument(
        "--pushes",
        required=True,
        help="Path to hourly workflow runs CSV (hour_utc, pushes)"
    )
    
    parser.add_argument(
        "--region",
        nargs=2,
        action="append",
        metavar=("NAME", "CI_CSV"),
        required=True,
        help="Region name and CI CSV path (repeat for each region)"
    )
    
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for sweep results (will create subdirs inside)"
    )
    
    parser.add_argument(
        "--sim-path",
        required=True,
        help="Path to simulate_case_studyN_regions.py"
    )
    
    # Baseline parameter values (from paper)
    parser.add_argument(
        "--baseline-phi",
        type=float,
        default=0.5,
        help="Baseline deferrable fraction (default: 0.5)"
    )
    
    parser.add_argument(
        "--baseline-d",
        type=int,
        default=6,
        help="Baseline deferral window in hours (default: 6)"
    )
    
    parser.add_argument(
        "--baseline-kappa",
        type=float,
        default=2.0,
        help="Baseline capacity cap multiplier (default: 2.0)"
    )
    
    parser.add_argument(
        "--baseline-tau",
        type=float,
        default=0.15,
        help="Baseline geographic threshold (default: 0.15)"
    )
    
    parser.add_argument(
        "--baseline-opp-phi",
        type=float,
        default=0.25,
        help="Baseline opportunistic phi (default: 0.25)"
    )
    
    parser.add_argument(
        "--energy-per-push",
        type=float,
        default=0.1,
        help="Energy per workflow run in kWh (default: 0.1)"
    )
    
    # Sweep ranges
    parser.add_argument(
        "--sweep-d",
        nargs="+",
        type=int,
        default=[2, 4, 6, 8, 10, 12],
        help="Values for D sweep (default: 2 4 6 8 10 12)"
    )
    
    parser.add_argument(
        "--sweep-phi",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6],
        help="Values for phi sweep (default: 0.3 0.4 0.5 0.6)"
    )
    
    parser.add_argument(
        "--sweep-kappa",
        nargs="+",
        type=float,
        default=[1.5, 2.0, 2.5],
        help="Values for kappa sweep (default: 1.5 2.0 2.5)"
    )
    
    parser.add_argument(
        "--sweep-tau",
        nargs="+",
        type=float,
        default=[0.10, 0.15, 0.20],
        help="Values for tau sweep (default: 0.10 0.15 0.20)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed output from simulation runs"
    )
    
    return parser.parse_args()


def run_simulation(sim_path, pushes, regions, output_subdir,
                   phi, d, kappa, tau, opp_phi, energy_per_push,
                   verbose=False):
    """
    Run a single simulation with specified parameters.
    
    The simulator creates summary_caseN.csv and raw_totals_caseN.csv
    in the output_subdir directory.
    
    Args:
        sim_path: Path to simulation script
        pushes: Path to pushes CSV
        regions: List of (name, ci_csv_path) tuples
        output_subdir: Output directory (e.g., out_sens_tri/d_6/)
        phi: Deferrable fraction
        d: Deferral window (hours)
        kappa: Capacity cap multiplier
        tau: Geographic threshold
        opp_phi: Opportunistic phi
        energy_per_push: Energy per workflow (kWh)
        verbose: Show command output
    """
    cmd = [
        "python3", sim_path,
        "--pushes", pushes,
        "--window-hours", str(d),
        "--phi", str(phi),
        "--cap-mult", str(kappa),
        "--geo-tau", str(tau),
        "--opportunistic-phi", str(opp_phi),
        "--energy-per-push-kwh", str(energy_per_push),
        "--outdir", str(output_subdir),  # ← Key: Each run gets its own directory
    ]
    
    # Add regions
    for name, ci_csv in regions:
        cmd.extend(["--region", name, ci_csv])
    
    # Run simulation
    try:
        if verbose:
            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)
        else:
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
    except subprocess.CalledProcessError as e:
        print(f"\n  ERROR: Simulation failed!", file=sys.stderr)
        print(f"  Parameters: phi={phi}, D={d}, kappa={kappa}, tau={tau}", file=sys.stderr)
        if e.stderr:
            print(f"  Error: {e.stderr.decode()}", file=sys.stderr)
        raise


def run_parameter_sweep(args):
    """Run all parameter sweeps."""
    
    # Create output directory
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Verify simulator exists
    sim_path = Path(args.sim_path)
    if not sim_path.exists():
        print(f"ERROR: Simulator not found at: {sim_path}", file=sys.stderr)
        print(f"Current directory: {Path.cwd()}", file=sys.stderr)
        sys.exit(1)
    
    # Verify input files exist
    if not Path(args.pushes).exists():
        print(f"ERROR: Pushes file not found: {args.pushes}", file=sys.stderr)
        sys.exit(1)
    
    for name, ci_csv in args.region:
        if not Path(ci_csv).exists():
            print(f"ERROR: CI file not found for {name}: {ci_csv}", file=sys.stderr)
            sys.exit(1)
    
    print("=" * 60)
    print("Parameter Sensitivity Sweep")
    print("=" * 60)
    print(f"Simulator: {sim_path}")
    print(f"Pushes: {args.pushes}")
    print(f"Regions: {', '.join(name for name, _ in args.region)}")
    print(f"Output: {outdir}/")
    print()
    print("Baseline parameters:")
    print(f"  φ = {args.baseline_phi}")
    print(f"  D = {args.baseline_d}h")
    print(f"  κ = {args.baseline_kappa}")
    print(f"  τ = {args.baseline_tau}")
    print()
    
    total_runs = (
        len(args.sweep_d) +
        len(args.sweep_phi) +
        len(args.sweep_kappa) +
        len(args.sweep_tau)
    )
    print(f"Total simulation runs: {total_runs}")
    print()
    print("Note: Each run creates a subdirectory with summary_caseN.csv")
    print("      and raw_totals_caseN.csv inside it.")
    print("=" * 60)
    print()
    
    run_count = 0
    
    # Sweep D (deferral window)
    print(f"[1/4] Sweeping D (deferral window): {args.sweep_d}")
    for d_val in args.sweep_d:
        run_count += 1
        print(f"  [{run_count}/{total_runs}] D = {d_val}h...", end=" ", flush=True)
        
        # Create subdirectory for this parameter value
        param_dir = outdir / f"d_{d_val}"
        param_dir.mkdir(parents=True, exist_ok=True)
        
        run_simulation(
            sim_path=str(sim_path),
            pushes=args.pushes,
            regions=args.region,
            output_subdir=param_dir,
            phi=args.baseline_phi,
            d=d_val,
            kappa=args.baseline_kappa,
            tau=args.baseline_tau,
            opp_phi=args.baseline_opp_phi,
            energy_per_push=args.energy_per_push,
            verbose=args.verbose
        )
        print("✓")
    
    # Sweep phi (deferrable fraction)
    print(f"[2/4] Sweeping φ (deferrable fraction): {args.sweep_phi}")
    for phi_val in args.sweep_phi:
        run_count += 1
        print(f"  [{run_count}/{total_runs}] φ = {phi_val}...", end=" ", flush=True)
        
        param_dir = outdir / f"phi_{phi_val}"
        param_dir.mkdir(parents=True, exist_ok=True)
        
        run_simulation(
            sim_path=str(sim_path),
            pushes=args.pushes,
            regions=args.region,
            output_subdir=param_dir,
            phi=phi_val,
            d=args.baseline_d,
            kappa=args.baseline_kappa,
            tau=args.baseline_tau,
            opp_phi=args.baseline_opp_phi,
            energy_per_push=args.energy_per_push,
            verbose=args.verbose
        )
        print("✓")
    
    # Sweep kappa (capacity cap)
    print(f"[3/4] Sweeping κ (capacity cap): {args.sweep_kappa}")
    for kappa_val in args.sweep_kappa:
        run_count += 1
        print(f"  [{run_count}/{total_runs}] κ = {kappa_val}...", end=" ", flush=True)
        
        param_dir = outdir / f"cap_{kappa_val}"
        param_dir.mkdir(parents=True, exist_ok=True)
        
        run_simulation(
            sim_path=str(sim_path),
            pushes=args.pushes,
            regions=args.region,
            output_subdir=param_dir,
            phi=args.baseline_phi,
            d=args.baseline_d,
            kappa=kappa_val,
            tau=args.baseline_tau,
            opp_phi=args.baseline_opp_phi,
            energy_per_push=args.energy_per_push,
            verbose=args.verbose
        )
        print("✓")
    
    # Sweep tau (geographic threshold)
    print(f"[4/4] Sweeping τ (geographic threshold): {args.sweep_tau}")
    for tau_val in args.sweep_tau:
        run_count += 1
        print(f"  [{run_count}/{total_runs}] τ = {tau_val}...", end=" ", flush=True)
        
        param_dir = outdir / f"tau_{tau_val}"
        param_dir.mkdir(parents=True, exist_ok=True)
        
        run_simulation(
            sim_path=str(sim_path),
            pushes=args.pushes,
            regions=args.region,
            output_subdir=param_dir,
            phi=args.baseline_phi,
            d=args.baseline_d,
            kappa=args.baseline_kappa,
            tau=tau_val,
            opp_phi=args.baseline_opp_phi,
            energy_per_push=args.energy_per_push,
            verbose=args.verbose
        )
        print("✓")
    
    print()
    print("=" * 60)
    print(f"✓ Sweep complete! Results in: {outdir}/")
    print()
    print("Directory structure:")
    print(f"  {outdir}/")
    for subdir in sorted(outdir.iterdir()):
        if subdir.is_dir():
            csv_files = list(subdir.glob("*.csv"))
            print(f"  ├── {subdir.name}/")
            for csv in csv_files:
                print(f"  │   └── {csv.name}")
    print()
    print("Next step: Generate plots with:")
    print(f"  python3 plot_sensitivity.py {outdir} --scenario pol")
    print("=" * 60)


def main():
    """Main entry point."""
    args = parse_args()
    
    try:
        run_parameter_sweep(args)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()