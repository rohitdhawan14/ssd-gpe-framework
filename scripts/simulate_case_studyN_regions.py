
#!/usr/bin/env python3
"""
simulate_case_studyN_regions.py

Generalized Case Study 4 simulator for N regions.

Inputs:
- Pushes CSV: hourly timeline with columns: hour_utc, pushes (or workflow_runs -> will be aliased)
- One or more region CI CSVs: each with hour_utc, ci_kg_per_kwh

Scenarios produced:
- Unmanaged (per region, and a "Portfolio" = sum of all regions assuming identical workload copy per region?)
- Opportunistic (per region only): shift phi_op of pushes into below-median hours (per day) within each region
- GPE (per region only): governance-constrained deferral window D, deferrable fraction phi, capacity cap C*median
- Geo-GPE (portfolio): N-region routing with policy constraints, choosing the cleanest eligible hour+region within window

Design notes:
- This N-region script treats the *same hourly push timeline* as the common workload demand vector.
  For per-region "Unmanaged" and "GPE" we compute the emissions as if the entire workload executed in that region.
  For "Geo-GPE" we *jointly* allocate flexible work across all regions with capacity caps in destination regions;
  non-deferrable work stays in its source hour (but can be placed in any region; we assume it stays local for fairness).
  In practice, we follow the paper's method: per-region what-if + a joint portfolio optimization.

Outputs:
- outdir/summary_caseN.csv   (columns: setting,region,index_normalized and absolute emissions)
- outdir/raw_totals_caseN.csv

CLI example:
python simulate_case_studyN_regions.py \
  --pushes prep_out/gha_runs_2023_hourly.csv \
  --region GB prep_out/entsoe_GB_2023_hourly_ci.csv \
  --region DE prep_out/entsoe_DE_2023_hourly_ci.csv \
  --region FR prep_out/entsoe_FR_2023_hourly_ci.csv \
  --outdir out_caseN \
  --window-hours 6 --phi 0.5 --cap-mult 2.0 --geo-tau 0.15 --opportunistic-phi 0.25

Assumptions:
- All inputs are hourly and aligned on hour_utc. We'll inner-join on common hours.
- E_w (kWh per push) is a scalar; normalized indices are unaffected by its absolute value, but we keep it for completeness.
"""

import argparse
import csv
import math
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple
import sys

import pandas as pd
import numpy as np


def read_pushes(pth: Path) -> pd.DataFrame:
    df = pd.read_csv(pth)
    # normalize column names
    cols = {c.lower().strip(): c for c in df.columns}
    # find hour and pushes/workflow_runs
    def pick(colnames, candidates):
        for cand in candidates:
            for c in colnames:
                if c == cand:
                    return colnames[c]
        return None

    hour_col = pick(cols, ["hour_utc","timestamp","hour"])
    pushes_col = pick(cols, ["pushes","workflow_runs","runs_total","runs__total"])
    if hour_col is None:
        raise ValueError("Pushes CSV must have an hour column (hour_utc/timestamp/hour).")
    if pushes_col is None:
        # try sum of runs__* if exists
        run_cols = [c for c in df.columns if c.lower().startswith("runs__")]
        if run_cols:
            df["pushes"] = df[run_cols].sum(axis=1)
            pushes_col = "pushes"
        else:
            raise ValueError("Pushes CSV must have pushes/workflow_runs or runs__* columns.")
    out = df[[hour_col, pushes_col]].copy()
    out.columns = ["hour_utc","pushes"]
    # parse hour_utc to datetime (UTC)
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True).dt.floor("H")
    out = out.groupby("hour_utc", as_index=False)["pushes"].sum().sort_values("hour_utc")
    return out


def read_ci(pth: Path) -> pd.DataFrame:
    df = pd.read_csv(pth)
    cols = {c.lower().strip(): c for c in df.columns}
    hour_col = None
    for cand in ["hour_utc","timestamp","time","hour"]:
        if cand in cols:
            hour_col = cols[cand]; break
    ci_col = None
    for cand in ["ci_kg_per_kwh","ci","carbon_intensity","kgco2_per_kwh"]:
        if cand in cols:
            ci_col = cols[cand]; break
    if hour_col is None or ci_col is None:
        raise ValueError(f"CI CSV {pth} must have hour_utc and ci_kg_per_kwh (or equivalent).")
    out = df[[hour_col, ci_col]].copy()
    out.columns = ["hour_utc","ci_kg_per_kwh"]
    out["hour_utc"] = pd.to_datetime(out["hour_utc"], utc=True).dt.floor("H")
    out = out.groupby("hour_utc", as_index=False)["ci_kg_per_kwh"].mean().sort_values("hour_utc")
    return out


def align_common_hours(pushes: pd.DataFrame, cis: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    hours = set(pushes["hour_utc"].astype(np.int64))
    for _, d in cis.items():
        hours = hours & set(d["hour_utc"].astype(np.int64))
    # inner-join hours
    hours = sorted(list(hours))
    if not hours:
        raise ValueError("No overlapping hours between pushes and CI files.")
    hours = pd.to_datetime(pd.Series(hours).astype("int64")).dt.tz_localize("UTC")
    pushes2 = pushes[pushes["hour_utc"].isin(hours)].copy().reset_index(drop=True)
    cis2 = {name: d[d["hour_utc"].isin(hours)].copy().reset_index(drop=True)
            for name, d in cis.items()}
    # ensure identical order
    pushes2 = pushes2.sort_values("hour_utc").reset_index(drop=True)
    for name in cis2:
        cis2[name] = cis2[name].sort_values("hour_utc").reset_index(drop=True)
    return pushes2, cis2


def unmanaged_emissions(pushes: pd.Series, ci: pd.Series, E_w: float) -> float:
    return float(np.sum(pushes.values * E_w * ci.values))


def opportunistic_within_region(pushes: pd.DataFrame, ci: pd.DataFrame, phi_op: float) -> float:
    """
    Best-effort: for each day, shift phi_op of work from dirty to clean (below daily median hours).
    Simple proportional reallocation — match earlier script semantics.
    Returns emissions (k gCO2).
    """
    df = pushes.merge(ci, on="hour_utc", how="left").copy()
    df["date"] = df["hour_utc"].dt.date
    out_push = df["pushes"].astype(float).values.copy()

    # group by day
    for date, idxs in df.groupby("date").groups.items():
        idxs = list(idxs)
        day_ci = df.loc[idxs, "ci_kg_per_kwh"].values
        day_push = df.loc[idxs, "pushes"].values.astype(float)
        md = np.median(day_ci)
        clean_mask = day_ci < md
        dirty_mask = ~clean_mask
        if not clean_mask.any() or not dirty_mask.any():
            continue

        movable = np.minimum(phi_op * day_push, day_push) * dirty_mask
        total_movable = movable.sum()
        if total_movable <= 0:
            continue
        # allocate proportionally to (md - ci)^+ over clean hours
        weights = (md - day_ci) * clean_mask
        weights = np.maximum(weights, 0)
        if weights.sum() == 0:
            continue
        alloc = total_movable * (weights / weights.sum())
        # apply
        out_push[idxs] = day_push - movable + alloc

    # emissions (use E_w=1 for normalized; we'll multiply by E_w externally)
    emis = np.sum(out_push * df["ci_kg_per_kwh"].values)
    return float(emis)


def gpe_within_region(pushes: pd.DataFrame, ci: pd.DataFrame, D: int, phi: float, cap_mult: float) -> float:
    """
    Governance-constrained within-region reallocation:
    - deferral window <= D hours
    - at most phi fraction deferrable
    - per-hour capacity cap <= cap_mult * median hourly throughput (over entire study window)
    Greedy: for each source hour, send to cleanest target hours within [i, i+D] while respecting cap.
    """
    df = pushes.merge(ci, on="hour_utc", how="left").copy()
    H = len(df)
    push = df["pushes"].astype(float).values
    ci_vals = df["ci_kg_per_kwh"].astype(float).values

    median_thr = np.median(push[push > 0]) if np.any(push > 0) else 0.0
    cap = cap_mult * median_thr if median_thr > 0 else np.inf
    out = push.copy()

    for i in range(H):
        base = push[i]
        movable = min(phi * base, base)
        if movable <= 0:
            continue
        # candidate targets j in [i, i+D]
        j_lo = i
        j_hi = min(H-1, i + D)
        cand = list(range(j_lo, j_hi+1))
        # sort by ascending ci
        cand.sort(key=lambda j: ci_vals[j])
        x = movable
        for j in cand:
            room = max(0.0, cap - out[j])
            r = min(x, room)
            if r > 0:
                out[i] -= r
                out[j] += r
                x -= r
                if x <= 1e-12:
                    break

    emis = float(np.sum(out * ci_vals))
    return emis


def geo_gpe_portfolio(pushes: pd.DataFrame, cis: Dict[str, pd.DataFrame],
                      D: int, phi: float, cap_mult: float, tau: float) -> float:
    """
    N-region Geo-GPE: Joint allocation across regions.
    Mechanics:
    - For each source hour i, take pushes[i]. The non-deferrable (1-phi) stays at hour i in its "local"
      region if we had regions per-source; since we have no source region data, we treat non-deferrable as
      executing *in-place* and compute its emissions at the *weighted average* CI across regions (neutral).
      Then, from the deferrable phi*pushes[i], choose (hour,region) pairs within [i, i+D] that are at least
      tau cleaner than the source ci of the same hour in a reference region (we use the *mean across regions*
      as the source benchmark to avoid bias), and respect per-region per-hour cap = cap_mult * med_throughput.
    - This is a conservative, neutral interpretation that avoids assuming a particular origin region.
    - Greedy algorithm: per hour i, create candidate (j,region) with ci reduction >= tau threshold w.r.t mean-ci[i],
      sort by ascending ci, allocate until movable is placed or caps saturate.

    Returns total portfolio emissions.
    """
    # Build aligned matrix: hours x regions
    hours = pushes["hour_utc"].values
    R = list(cis.keys())
    H = len(hours)
    ci_mat = np.zeros((H, len(R)), dtype=float)
    for r_idx, r in enumerate(R):
        ci_mat[:, r_idx] = cis[r]["ci_kg_per_kwh"].astype(float).values
    push = pushes["pushes"].astype(float).values

    # Caps per region per hour using global median throughput (same cap for all hours for simplicity)
    median_thr = np.median(push[push > 0]) if np.any(push > 0) else 0.0
    cap = cap_mult * median_thr if median_thr > 0 else np.inf
    # Initialize allocation (per hour, per region)
    alloc = np.zeros((H, len(R)), dtype=float)

    total_emis = 0.0

    # Non-deferrable share: we model its emissions using the mean CI across regions at hour i (neutral baseline)
    mean_ci = ci_mat.mean(axis=1)
    nondef = (1.0 - phi) * push
    total_emis += float(np.sum(nondef * mean_ci))

    # Deferrable allocation across hours/regions
    for i in range(H):
        movable = phi * push[i]
        if movable <= 0:
            continue
        # threshold for being "cleaner enough" than current hour (mean benchmark)
        thresh = (1.0 - tau) * mean_ci[i]
        # candidates j in [i, i+D] x regions where ci[j,r] <= thresh
        j_lo = i
        j_hi = min(H-1, i + D)
        candidates = []
        for j in range(j_lo, j_hi+1):
            for r_idx in range(len(R)):
                cij = ci_mat[j, r_idx]
                if cij <= thresh:
                    candidates.append((cij, j, r_idx))
        # sort by ascending CI
        candidates.sort(key=lambda t: t[0])
        x = movable
        for cij, j, r_idx in candidates:
            room = max(0.0, cap - alloc[j, r_idx])
            r = min(x, room)
            if r > 0:
                alloc[j, r_idx] += r
                x -= r
                if x <= 1e-12:
                    break
        # any leftover that couldn't be allocated to cleaner slots runs at the original hour mean CI
        if x > 1e-12:
            total_emis += float(x * mean_ci[i])

    # add emissions from allocated deferrable work
    # sum over (j, r) of alloc[j,r] * ci[j,r]
    total_emis += float(np.sum(alloc * ci_mat))

    return total_emis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pushes", type=Path, required=True, help="CSV with hour_utc and pushes/workflow_runs")
    ap.add_argument("--region", nargs=2, action="append", metavar=("NAME","CI_CSV"),
                    help="Repeat for each region, e.g., --region GB file.csv --region DE file.csv")
    ap.add_argument("--outdir", type=Path, required=True)

    ap.add_argument("--window-hours", type=int, default=6)
    ap.add_argument("--phi", type=float, default=0.5)
    ap.add_argument("--cap-mult", type=float, default=2.0)
    ap.add_argument("--geo-tau", type=float, default=0.15)
    ap.add_argument("--opportunistic-phi", type=float, default=0.25)

    ap.add_argument("--energy-per-push-kwh", type=float, default=0.1)

    args = ap.parse_args()
    if not args.region or len(args.region) < 2:
        raise SystemExit("Provide at least two --region NAME FILE.csv entries.")

    args.outdir.mkdir(parents=True, exist_ok=True)

    pushes = read_pushes(args.pushes)

    # Read all region CIs
    cis = {}
    for name, file in args.region:
        cis[name] = read_ci(Path(file))

    # Align on common hours
    pushes, cis = align_common_hours(pushes, cis)

    # Unmanaged & within-region scenarios for each region
    rows = []
    unmanaged_totals = {}
    for name, ci_df in cis.items():
        emis_unmg = unmanaged_emissions(pushes["pushes"], ci_df["ci_kg_per_kwh"], args.energy_per_push_kwh)
        unmanaged_totals[name] = emis_unmg
        rows.append({"setting":"Unmanaged","region":name,"emissions_kg":emis_unmg})

        emis_opp = opportunistic_within_region(pushes, ci_df, phi_op=args.opportunistic_phi) * args.energy_per_push_kwh
        rows.append({"setting":"Opportunistic","region":name,"emissions_kg":emis_opp})

        emis_gpe = gpe_within_region(pushes, ci_df, D=args.window_hours, phi=args.phi, cap_mult=args.cap_mult) * args.energy_per_push_kwh
        rows.append({"setting":"GPE","region":name,"emissions_kg":emis_gpe})

    # Portfolio via Geo-GPE across all regions jointly
    emis_geo_portfolio = geo_gpe_portfolio(pushes, cis, D=args.window_hours, phi=args.phi,
                                           cap_mult=args.cap_mult, tau=args.geo_tau) * args.energy_per_push_kwh
    rows.append({"setting":"Geo-GPE","region":"Portfolio","emissions_kg":emis_geo_portfolio})

    # Portfolio unmanaged index for normalization: mean of per-region Unmanaged? In paper we index per region to 100.
    # Here, for Portfolio, we define unmanaged_portfolio as mean unmanaged across regions (neutral), for index.
    unmanaged_portfolio = float(np.mean(list(unmanaged_totals.values())))

    # Build summary with normalized index = (emissions / unmanaged_region) * 100
    summary = []
    for r in rows:
        setting = r["setting"]; region = r["region"]; emis = r["emissions_kg"]
        if region == "Portfolio":
            base = unmanaged_portfolio
        else:
            base = unmanaged_totals.get(region, np.nan)
        idx = (emis / base * 100.0) if base and not math.isnan(base) and base > 0 else np.nan
        summary.append({
            "setting": setting,
            "region": region,
            "emissions_kg": f"{emis:.6f}",
            "index_normalized": f"{idx:.2f}"
        })

    # Write CSVs
    with open(args.outdir / "raw_totals_caseN.csv","w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting","region","emissions_kg"])
        w.writeheader()
        for r in rows:
            out = dict(r)
            out["emissions_kg"] = f"{out['emissions_kg']:.6f}"
            w.writerow(out)

    with open(args.outdir / "summary_caseN.csv","w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["setting","region","emissions_kg","index_normalized"])
        w.writeheader()
        for r in summary:
            w.writerow(r)

    print("Wrote:", args.outdir / "summary_caseN.csv")
    print("Wrote:", args.outdir / "raw_totals_caseN.csv")


if __name__ == "__main__":
    main()
