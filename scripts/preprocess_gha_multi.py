#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified preprocessing for:
  - GitHub Actions "workflow runs" JSON (.json or .json.gz), first-attempt filter, per-event counts
  - ENTSO-E (or similar) hourly carbon intensity CSVs per region
      * Either (A) already "hour_utc, ci_*" style
      * Or (B) generation-mix exports with MTU + technology MW columns (auto-compute kgCO2/kWh)

Outputs:
  - gha_runs_<year>_hourly.csv
  - gha_runs_by_event_<year>_hourly.csv
  - entsoe_<REG>_<year>_hourly_ci.csv

Usage:
  python3 preprocess_gha_multi.py \
    --runs-json runs.json.gz \
    --region GB ../raw/entsoe_gb_raw.csv \
    --region DE ../raw/entsoe_de_raw.csv \
    --region FR ../raw/entsoe_fr_raw.csv \
    --year 2023 \
    --first-attempt-only \
    --outdir prep_out
"""

import argparse
import csv
import gzip
import io
import json
import os
import re
from collections import defaultdict, Counter
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple, List
from datetime import datetime
import pytz
import numpy as np
import pandas as pd
from typing import Optional


try:
    import pytz
except ImportError:
    raise SystemExit("Please: pip install pytz pandas numpy")

# ---------------------------- Common helpers ----------------------------
"""
Emission factors (kg CO₂e per kWh) based on lifecycle assessment (LCA).

Sources:
- Fossil fuels: IPCC 2014 Guidelines for National GHG Inventories
- Nuclear/Renewables: IPCC AR5 WG3 Annex III (Schlömer et al., 2014)
  "Technology-specific Cost and Performance Parameters"
- Biomass: [YOUR SPECIFIC SOURCE - e.g., Menten et al. 2013 or DEFRA 2023]
- Waste: EU Reference Document on Best Available Techniques

Key assumptions:
1. Biomass uses a partial carbon-neutrality approach accounting for 
   supply chain emissions (230 g CO₂e/kWh) based on [CITATION].
2. Hydro pumped storage generation = 0 (only generation side counted;
   pumping consumption handled separately in ENTSO-E data).
3. Values represent lifecycle emissions including construction, 
   operation, and decommissioning.

References:
- IPCC (2014). Climate Change 2014: Mitigation of Climate Change.
- Schlömer et al. (2014). Annex III: Technology-specific cost and 
  performance parameters. In: Climate Change 2014: Mitigation.
"""
EF_KG_PER_KWH = {
    # Fossils (high emission factors)
    "Fossil Brown coal/Lignite": 1.05,    # [IPCC 2019, Table 2.3]
    "Fossil Hard coal": 0.95,              # [IPCC 2019, Table 2.3]
    "Fossil Coal-derived gas": 0.90,       # [IPCC estimate]
    "Fossil Gas": 0.49,                    # [IPCC AR5 + 3% methane leakage]
    "Fossil Oil": 0.78,                    # [IPCC 2019, Table 2.3]
    "Fossil Oil shale": 1.10,              # [Conservative estimate]
    "Fossil Peat": 1.10,                   # [IPCC 2019]

    # Low-carbon / renewables (lifecycle LCA factors)
    "Nuclear": 0.012,                      # [IPCC AR5 Annex III, median]
    "Hydro Pumped Storage": 0.0,           # Generation only (see note above)
    "Hydro Run-of-river and poundage": 0.024,  # [IPCC AR5 Annex III]
    "Hydro Water Reservoir": 0.024,        # [IPCC AR5 Annex III]
    "Biomass": 0.23,                       # [YOUR CITATION - CRITICAL!]
    "Geothermal": 0.038,                   # [IPCC AR5 Annex III]
    "Wind Onshore": 0.012,                 # [IPCC AR5 Annex III, median]
    "Wind Offshore": 0.012,                # [IPCC AR5 Annex III, median]
    "Solar": 0.045,                        # [IPCC AR5 Annex III, median PV]
    "Marine": 0.02,                        # [Conservative estimate]
    "Other renewable": 0.05,               # [Conservative estimate]
    "Other": 0.40,                         # [Fallback: mixed technologies]
    "Waste": 0.35,                         # [EU BREF, mixed municipal waste]
    "Energy storage": 0.0,                 # Not counted as generation
}

def open_maybe_gzip(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def parse_iso_trunc_hour(s: str) -> Optional[str]:
    """Parse many ISO-ish timestamps and return UTC hour 'YYYY-MM-DDTHH:00:00Z'."""
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z","+00:00"))
        dt = dt.astimezone(timezone.utc)
        return dt.strftime("%Y-%m-%dT%H:00:00Z")
    except Exception:
        pass
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%d %H:%M:%S%z",
    ]
    for f in fmts:
        try:
            dt = datetime.strptime(s, f)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.strftime("%Y-%m-%dT%H:00:00Z")
        except Exception:
            continue
    return None

def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def write_csv(path: str, header: list, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

# ------------------------ GHA runs (unchanged behavior) ------------------------

def extract_run_started_at(rec: dict) -> Optional[str]:
    md = rec.get("metadata") or rec.get("run", {}).get("metadata")
    if isinstance(md, dict):
        for k in ["run_started_at","created_at","updated_at","started_at"]:
            ts = md.get(k)
            if ts:
                hour = parse_iso_trunc_hour(ts)
                if hour: return hour
    for k in ["run_started_at","created_at","updated_at","timestamp","time"]:
        if k in rec:
            hour = parse_iso_trunc_hour(rec[k])
            if hour: return hour
    return None

def extract_event(rec: dict) -> str:
    md = rec.get("metadata")
    if isinstance(md, dict):
        ev = md.get("event")
        if ev: return ev
    return rec.get("event") or "other"

def extract_attempt(rec: dict) -> int:
    md = rec.get("metadata")
    if isinstance(md, dict) and "run_attempt" in md:
        try: return int(md["run_attempt"])
        except: pass
    if "run_attempt" in rec:
        try: return int(rec["run_attempt"])
        except: pass
    return 1

def _consume_run_record(rec, hourly_counts, event_counts, year, first_attempt_only):
    hour = extract_run_started_at(rec)
    if hour is None:
        return
    if year is not None and not hour.startswith(str(year)):
        return
    if first_attempt_only and extract_attempt(rec) != 1:
        return
    ev = extract_event(rec)
    hourly_counts[hour] += 1
    event_counts[hour][ev] += 1

def process_runs_json(runs_json_path: str, outdir: str, year: Optional[int], first_attempt_only: bool):
    hourly_counts = Counter()
    event_counts = defaultdict(Counter)

    with open_maybe_gzip(runs_json_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                _consume_run_record(rec, hourly_counts, event_counts, year, first_attempt_only)
            except Exception:
                # try array once
                try:
                    data = json.loads(line)
                    if isinstance(data, list):
                        for rec in data:
                            _consume_run_record(rec, hourly_counts, event_counts, year, first_attempt_only)
                except Exception:
                    continue

    rows = sorted((h, c) for h, c in hourly_counts.items())
    write_csv(os.path.join(outdir, f"gha_runs_{year or 'all'}_hourly.csv"),
              ["hour_utc", "workflow_runs"], rows)

    all_events = sorted({ev for _h, cdict in event_counts.items() for ev in cdict.keys()})
    header = ["hour_utc"] + [f"runs__{ev}" for ev in all_events] + ["runs__other", "runs__total"]
    out_rows = []
    for hour in sorted(event_counts.keys()):
        row = [hour]
        total = 0
        evdict = event_counts[hour]
        for ev in all_events:
            val = evdict.get(ev, 0)
            total += val
            row.append(val)
        other = 0
        row.append(other)
        row.append(total + other)
        out_rows.append(row)
    write_csv(os.path.join(outdir, f"gha_runs_by_event_{year or 'all'}_hourly.csv"),
              header, out_rows)

# ------------------------ ENTSO-E CI processing ------------------------

# Lifecycle EF (kg CO2 per MWh). Adjust if you have a canonical table.
# EF = {
#     "hard_coal": 820, "lignite": 1050, "fossil_coal_gas": 820,
#     "gas": 490, "oil": 650, "oil_shale": 900, "peat": 1060,
#     "nuclear": 12, "wind_on": 11, "wind_off": 12,
#     "solar": 45, "hydro_ror": 24, "hydro_res": 24,
#     "biomass": 230, "waste": 500, "geothermal": 45,
#     "marine": 10, "other_ren": 100, "other": 300,
#     "hydro_ps_gen": 24, "storage_gen": 400,  # conservative placeholders
# }

GEN_COLS_CANON = [
    "Biomass - Actual Aggregated [MW]",
    "Energy storage - Actual Aggregated [MW]",
    "Fossil Brown coal/Lignite - Actual Aggregated [MW]",
    "Fossil Coal-derived gas - Actual Aggregated [MW]",
    "Fossil Gas - Actual Aggregated [MW]",
    "Fossil Hard coal - Actual Aggregated [MW]",
    "Fossil Oil - Actual Aggregated [MW]",
    "Fossil Oil shale - Actual Aggregated [MW]",
    "Fossil Peat - Actual Aggregated [MW]",
    "Geothermal - Actual Aggregated [MW]",
    "Hydro Pumped Storage - Actual Aggregated [MW]",
    "Hydro Pumped Storage - Actual Consumption [MW]",
    "Hydro Run-of-river and poundage - Actual Aggregated [MW]",
    "Hydro Water Reservoir - Actual Aggregated [MW]",
    "Marine - Actual Aggregated [MW]",
    "Nuclear - Actual Aggregated [MW]",
    "Other - Actual Aggregated [MW]",
    "Other renewable - Actual Aggregated [MW]",
    "Solar - Actual Aggregated [MW]",
    "Waste - Actual Aggregated [MW]",
    "Wind Offshore - Actual Aggregated [MW]",
    "Wind Onshore - Actual Aggregated [MW]",
]

def _tech_label(col: str) -> str:
    s = col.lower()
    if "biomass" in s: return "biomass"
    if "coal-derived" in s: return "fossil_coal_gas"
    if "brown coal" in s or "lignite" in s: return "lignite"
    if "hard coal" in s: return "hard_coal"
    if "fossil gas" in s: return "gas"
    if "fossil oil shale" in s: return "oil_shale"
    if "fossil peat" in s: return "peat"
    if "fossil oil" in s and "shale" not in s: return "oil"
    if "geothermal" in s: return "geothermal"
    if "nuclear" in s: return "nuclear"
    if "solar" in s: return "solar"
    if "wind offshore" in s: return "wind_off"
    if "wind onshore" in s: return "wind_on"
    if "hydro pumped storage - actual aggregated" in s: return "hydro_ps_gen"
    if "hydro pumped storage - actual consumption" in s: return "hydro_ps_cons"
    if "hydro run-of-river" in s: return "hydro_ror"
    if "hydro water reservoir" in s: return "hydro_res"
    if "marine" in s: return "marine"
    if "other renewable" in s: return "other_ren"
    if "other - actual aggregated" in s: return "other"
    if "energy storage - actual aggregated" in s: return "storage_gen"
    return "unknown"

def _looks_like_mix(header: List[str]) -> bool:
    return ("MTU" in header) and any(h.endswith("[MW]") for h in header)

# ---- helper: map ENTSO-E column names to technology names above ----
def _technology_from_column(col: str) -> Optional[str]:
    # Columns look like: "Wind Onshore - Actual Aggregated [MW]"
    if not col.endswith("[MW]"):
        return None
    name = col.split("- Actual")[0].strip()
    return name  # matches keys in EF_KG_PER_KWH if present, else falls back later


def _mix_to_ci_hourly(path: str, year: Optional[int]) -> pd.DataFrame:
    # Some ENTSO-E exports are semicolon-separated; detect quickly.
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        sniff = f.read(1024)
    sep = ";" if sniff.count(";") > sniff.count(",") else ","

    df = pd.read_csv(path, sep=sep, engine="python")
    if "MTU" not in df.columns:
        raise ValueError("Expected 'MTU' column in ENTSO-E mix export.")

    # Parse MTU start time like "01.01.2023 00:00 - 01.01.2023 00:15 (CET/CEST)"
    tz_brussels = pytz.timezone("Europe/Brussels")

    def parse_mtu_start(s):
        if pd.isna(s):
            return pd.NaT
        s = str(s)
        # keep the left bound, drop right bound and the (CET/CEST) suffix if present
        s = s.split("-")[0].split("(")[0].strip()
        dt = None
        for fmt in ("%d.%m.%Y %H:%M", "%Y-%m-%d %H:%M"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except Exception:
                continue
        if dt is None:
            # final fallback with dayfirst parsing
            dt = pd.to_datetime(s, dayfirst=True, errors="coerce")
            if pd.isna(dt):
                return pd.NaT
            # pandas returns tz-naive here; localize below
            dt = dt.to_pydatetime()
        # local time in CET/CEST → to UTC
        local = tz_brussels.localize(dt)
        return local.astimezone(pytz.UTC)

    df["hour_utc"] = df["MTU"].map(parse_mtu_start)
    df = df.dropna(subset=["hour_utc"])
    if year is not None:
        df = df[df["hour_utc"].dt.year == int(year)]

    # Normalize to **hourly UTC** by flooring
    df["hour_utc"] = df["hour_utc"].dt.floor("h")

    # Convert generation columns to numeric MW (n/e → NaN)
    gen_cols = [c for c in df.columns if c.endswith("[MW]")]
    for c in gen_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Separate out consumption fields to exclude them from generation sums
    consumption_cols = [c for c in gen_cols if "Consumption" in c]
    gen_only_cols = [c for c in gen_cols if c not in consumption_cols]

    # Build per-row carbon intensity numerator and denominator
    # numerator = sum(gen_MW * EF); denom = sum(gen_MW)
    def row_intensity_kg_per_kwh(row) -> float:
        num = 0.0
        den = 0.0
        for c in gen_only_cols:
            tech = _technology_from_column(c)
            mw = row[c]
            if pd.isna(mw) or mw <= 0:
                continue
            ef = EF_KG_PER_KWH.get(tech, EF_KG_PER_KWH.get("Other", 0.40))
            num += mw * ef
            den += mw
        if den <= 0:
            return np.nan
        # MW-weighted average factor is already kg/kWh because EF is in kg/kWh
        return num / den

    df["ci_kg_per_kwh"] = df.apply(row_intensity_kg_per_kwh, axis=1)

    # Aggregate sub-hourly → hourly mean (already floored)
    out = (
        df[["hour_utc", "ci_kg_per_kwh"]]
        .groupby("hour_utc", as_index=False)["ci_kg_per_kwh"]
        .mean()
        .sort_values("hour_utc")
    )

    # Drop hours with all-NaN intensity (no generation)
    out = out.dropna(subset=["ci_kg_per_kwh"])

    return out[["hour_utc", "ci_kg_per_kwh"]]

def _intensity_csv_to_hourly(path: str, year: Optional[int]) -> pd.DataFrame:
    # generic two-column (ts, ci) file (units auto-detected)
    with open(path, "r", encoding="utf-8") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        if not header:
            raise ValueError("Empty CSV header.")
        # detect ts, ci columns simply by position (0,1) or name patterns
        lower = [h.lower() for h in header]
        def pick_ts():
            for k in ["hour_utc","datetime_utc","utc_datetime","datetime","time_utc","timestamp","ts"]:
                if k in lower:
                    return header[lower.index(k)]
            return header[0]
        def pick_ci():
            for k in ["ci_kg_per_kwh","carbon_intensity_kg_per_kwh","ci_g_per_kwh","gco2_per_kwh","carbon_intensity_gco2_per_kwh","intensity_g_per_kwh","ci","carbon_intensity"]:
                if k in lower:
                    return header[lower.index(k)]
            return header[1] if len(header) > 1 else None

        ts_col, ci_col = pick_ts(), pick_ci()
        if ci_col is None:
            raise ValueError("Cannot detect CI column.")
        ts_idx, ci_idx = header.index(ts_col), header.index(ci_col)

        rows = []
        for row in rdr:
            if not row or len(row) <= max(ts_idx, ci_idx):
                continue
            hour = parse_iso_trunc_hour(row[ts_idx].strip())
            if hour is None:
                continue
            if year is not None and not hour.startswith(str(year)):
                continue
            raw = row[ci_idx].strip()
            # normalize grams → kg if needed, by column name:
            lname = ci_col.lower()
            try:
                val = float(re.sub(r"[^0-9.\-eE]", "", raw))
            except Exception:
                continue
            if "g_per_kwh" in lname or "gco2" in lname or "intensity_g" in lname:
                val = val / 1000.0
            rows.append((hour, val))

    df = pd.DataFrame(rows, columns=["hour_utc","ci_kg_per_kwh"])
    df["hour_utc"] = pd.to_datetime(df["hour_utc"], utc=True)
    df = df.dropna().sort_values("hour_utc")
    return df

def process_region_ci(region_name: str, csv_path: str, outdir: str, year: Optional[int]):
    # Decide which path: mix or already intensity
    # Heuristic: if file contains "MTU" and many "[MW]" columns → compute from mix.
    hdr = None
    with open(csv_path, "r", encoding="utf-8") as f:
        snif = csv.Sniffer()
        sample = f.read(4096)
        f.seek(0)
        rdr = csv.reader(f)
        hdr = next(rdr)
    looks_mix = ("MTU" in hdr) and any(h.endswith("[MW]") for h in hdr)

    if looks_mix:
        out = _mix_to_ci_hourly(csv_path, year)
    else:
        out = _intensity_csv_to_hourly(csv_path, year)

    out_path = os.path.join(outdir, f"entsoe_{region_name.upper()}_{year or 'all'}_hourly_ci.csv")
    out.to_csv(out_path, index=False)
    print(f"[+] Wrote {out_path} rows={len(out)}  range: {out['hour_utc'].min()} → {out['hour_utc'].max()}")

# ------------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-json", required=True, help="Path to GHA runs JSON (.json or .json.gz)")
    ap.add_argument("--region", action="append", nargs=2, metavar=("NAME","CSV"),
                    help="Repeatable: region name and path to raw CSV (either CI or ENTSO-E mix).",
                    default=[])
    ap.add_argument("--year", type=int, default=None, help="Year filter (e.g., 2023). If omitted, keep all.")
    ap.add_argument("--first-attempt-only", action="store_true", help="Only include run_attempt == 1")
    ap.add_argument("--outdir", default="prep_out", help="Output directory")
    args = ap.parse_args()

    ensure_outdir(args.outdir)

    print(f"[+] Processing runs: {args.runs_json}")
    process_runs_json(args.runs_json, args.outdir, args.year, args.first_attempt_only)
    print("[+] Wrote hourly runs CSVs.")

    for name, path in args.region:
        print(f"[+] Processing region {name}: {path}")
        process_region_ci(name, path, args.outdir, args.year)

    print("[✓] Done.")

if __name__ == "__main__":
    main()
