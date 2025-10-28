#!/usr/bin/env python3
import argparse, pathlib, re, sys
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# Ensure text renders cleanly in PDFs (no clipping/odd glyphs)
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

PREF_PATTERNS = ["*summary*.csv", "*totals*.csv", "*.csv"]

def find_csv_recursive(pathlike, debug=False):
    p = pathlib.Path(pathlike)
    if p.is_file() and p.suffix.lower()==".csv":
        return p
    if p.is_dir():
        for pat in PREF_PATTERNS:
            hits = sorted(p.rglob(pat))
            if hits:
                if debug:
                    print(f"[DEBUG] Using CSV {hits[0]} for {p.name}")
                return hits[0]
    return None

def to_num(s):
    return pd.to_numeric(s, errors='coerce')

def autodetect_totals(df, debug=False):
    if df is None or df.empty:
        return dict(unmanaged=None, opt=None, pol=None, geo=None)
    cols_lc = {c.lower(): c for c in df.columns}

    scen_col = None
    for cand in ["scenario","setting","stage","label","name","mode","model"]:
        if cand in cols_lc:
            scen_col = cols_lc[cand]; break

    em_col = None
    for cand in ["emissions","total_emissions","emissions_kg","kgco2e","kgco2","value","total"]:
        if cand in cols_lc:
            em_col = cols_lc[cand]; break

    reg_col = None
    for cand in ["region","zone","country","market"]:
        if cand in cols_lc:
            reg_col = cols_lc[cand]; break

    norm_col = None
    for cand in ["normalized","norm","index","normalized_emissions","norm_emissions"]:
        if cand in cols_lc:
            norm_col = cols_lc[cand]; break

    # LONG normalized
    if norm_col and scen_col:
        s = df.copy()
        s[scen_col] = s[scen_col].astype(str).str.lower()
        g = s.groupby(s[scen_col])[norm_col].apply(lambda x: to_num(x).dropna().mean())
        def pick(*names):
            for n in names:
                if n in g.index: return float(g[n])
            return None
        unmanaged = pick("unmanaged","as_is","status_quo","baseline")
        opt       = pick("opt","opportunistic","optimal","counterfactual","best_effort")
        pol       = pick("pol","policy","gpe","policy_constrained")
        geo       = pick("geo","geo-gpe","portfolio","multi_region")
        if unmanaged is None: unmanaged = 100.0
        if debug:
            print(f"[DEBUG] LONG normalized scenarios -> {list(g.index)}; unmanaged={unmanaged}, pol={pol}, opt={opt}, geo={geo}")
        return dict(unmanaged=unmanaged, opt=opt, pol=pol, geo=geo)

    # LONG totals
    if scen_col and em_col:
        s = df.copy()
        s[scen_col] = s[scen_col].astype(str).str.lower()
        s[em_col] = to_num(s[em_col]).fillna(0)
        g = s.groupby(s[scen_col])[em_col].sum()
        def pick(*names):
            for n in names:
                if n in g.index: return float(g[n])
            return None
        unmanaged = pick("unmanaged","as_is","status_quo","baseline")
        opt       = pick("opt","opportunistic","optimal","counterfactual","best_effort")
        pol       = pick("pol","policy","gpe","policy_constrained")
        geo       = pick("geo","geo-gpe","portfolio","multi_region")
        if debug:
            print(f"[DEBUG] LONG totals scenarios -> {list(g.index)}; unmanaged={unmanaged}, pol={pol}, opt={opt}, geo={geo}")
        return dict(unmanaged=unmanaged, opt=opt, pol=pol, geo=geo)

    # WIDE fallback
    def sum_cols(tokens):
        chosen = [cols_lc[c] for c in cols_lc if any(t in c for t in tokens)]
        if not chosen: return None
        vals = df[chosen].apply(to_num, errors='coerce').fillna(0)
        return float(vals.to_numpy().sum())

    unmanaged = sum_cols(["unmanaged","as_is","status_quo","baseline"])
    opt       = sum_cols(["opt","opportunistic","optimal","counterfactual","best_effort"])
    pol       = sum_cols(["pol","policy","gpe","policy_constrained"])
    geo       = sum_cols(["geo","portfolio","multi_region","geo_gpe","geo-gpe"])

    if debug:
        print(f"[DEBUG] WIDE probe -> unmanaged={unmanaged} opt={opt} pol={pol} geo={geo}")

    if any(v is not None for v in [unmanaged,opt,pol,geo]):
        return dict(unmanaged=unmanaged, opt=opt, pol=pol, geo=geo)

    return dict(unmanaged=None, opt=None, pol=None, geo=None)

def extract_x(name, prefix):
    stem = pathlib.Path(name).name
    if stem.endswith(".csv"): stem = stem[:-4]
    if prefix in ("tau","geo_tau"):
        m = re.search(r"(?:^tau_|^geo[_-]tau_)([0-9.]+)", stem, re.I)
    else:
        m = re.search(r"^" + re.escape(prefix) + r"_([0-9.]+)", stem, re.I)
    if not m: return None
    try: return float(m.group(1))
    except: return None

def collect(indir, prefix, scen_key, debug=False):
    indir = pathlib.Path(indir)
    items = sorted(list(indir.glob(f"{prefix}_*")))
    xs, ys = [], []
    for item in items:
        xval = extract_x(item.name, prefix)
        if xval is None: continue
        csvp = find_csv_recursive(item, debug=debug)
        if not csvp:
            print(f"[WARN] No CSV in {item}", file=sys.stderr)
            continue
        try:
            df = pd.read_csv(csvp)
        except Exception as e:
            print(f"[WARN] Read failed {csvp}: {e}", file=sys.stderr)
            continue
        totals = autodetect_totals(df, debug=debug)
        un = totals.get("unmanaged"); sc = totals.get(scen_key)
        if un is None and sc is not None:
            un = 100.0  # normalized case
        if un is None or sc is None or un <= 0:
            print(f"[WARN] Skipping {item}: missing totals (unmanaged={un}, {scen_key}={sc})")
            continue
        y = 100.0 * sc / un if un != 100.0 else float(sc)
        xs.append(xval); ys.append(y)
    if not xs:
        return None, None
    xs, ys = zip(*sorted(zip(xs, ys)))
    return np.array(xs), np.array(ys)

def lineplot(x, y, xlabel, outfile):
    # One-column IEEE figure size
    fig, ax = plt.subplots(figsize=(3.25, 2.6), constrained_layout=False)

    ax.plot(x, y, marker='o', linewidth=1.5)
    ax.set_xlabel(xlabel, fontsize=9)

    # Two-line ylabel to avoid left clipping in narrow columns
    ax.set_ylabel("Normalized emissions\n(Unmanaged = 100)", fontsize=9, labelpad=6)

    ax.tick_params(axis='both', labelsize=8)
    ax.grid(True, linewidth=0.4, alpha=0.5)

    # Ensure nothing is clipped even with long labels
    fig.tight_layout(pad=1.0)
    fig.subplots_adjust(left=0.34, right=0.98, top=0.96, bottom=0.22)

    fig.savefig(outfile, bbox_inches='tight', dpi=300)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("indir")
    ap.add_argument("--scenario", default="pol", choices=["pol","geo","opt"])
    ap.add_argument("--tau-prefix", default="tau", choices=["tau","geo_tau"])
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    scen_key = {"pol":"pol", "geo":"geo", "opt":"opt"}[args.scenario]

    x,y = collect(args.indir, "d", scen_key, debug=args.debug)
    if x is not None: lineplot(x,y, "Deferral window $D$ (hours)", pathlib.Path(args.indir)/"sens_window_hours.pdf")
    else: print("[WARN] No D sweep found.")

    x,y = collect(args.indir, "phi", scen_key, debug=args.debug)
    if x is not None: lineplot(x,y, r"Deferrable share $\phi$", pathlib.Path(args.indir)/"sens_phi.pdf")
    else: print("[WARN] No phi sweep found.")

    x,y = collect(args.indir, "cap", scen_key, debug=args.debug)
    if x is not None: lineplot(x,y, r"Capacity cap $\kappa$", pathlib.Path(args.indir)/"sens_cap_mult.pdf")
    else: print("[WARN] No cap/kappa sweep found.")

    # Robust tau sweep handling
    tried = []
    for pref in [args.tau_prefix, ("geo_tau" if args.tau_prefix=="tau" else "tau")]:
        tried.append(pref)
        x,y = collect(args.indir, pref, scen_key, debug=args.debug)
        if x is not None:
            lineplot(x,y, r"Geo threshold $\tau$", pathlib.Path(args.indir)/"sens_geo_tau.pdf")
            break
    else:
        print(f"[WARN] No tau sweep found (tried {tried}).")

if __name__ == "__main__":
    main()
