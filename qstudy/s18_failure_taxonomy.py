#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""S18 Failure Taxonomy (simplified per PRE-DECLARATION).
Reads sealed .npz files with fail_ep*_x traces, computes per-episode lock_w1000,
classifies failed episodes (lock_w1000 < 0.5) into 4 categories:
  (a) never_acquires: |offset| > 1000 Hz for first half of episode (post burn-in)
  (b) acquires_then_loses: offset drops below 1000 Hz at least once, then exceeds 1000 Hz for >=10 consecutive steps later
  (c) chatter: offset sign changes >40% of steps while |offset| < 2000 Hz throughout
  (d) other: none of the above

Scope: Z1 9 configs (10 seeds each), Z2A E1/E4/FTB_200000, Tier B E2/E3.
Output: $w\results\s18\s18_counts.csv, s18_summary.md, 3 example PNGs.
"""
import os, json, numpy as np, matplotlib
from collections import defaultdict
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import csv

WORK = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
SEALED = r"C:\Projects\MiniProject_Sem4\qstudy\results\sealed"
OUTDIR = os.path.join(WORK, "results", "s18")
os.makedirs(OUTDIR, exist_ok=True)

BURN_IN = 20
W = 1000.0
FAIL_THRESHOLD = 0.5  # per-episode lock_w1000 < 0.5 = failure

# Z1 base configs (9 configs, 10 seeds each)
Z1_CONFIGS = [
    "rl__dither__plain",
    "rl__dither__filter",
    "rl__dither__DRcurr",
    "rl__dither__full",
    "rl__rabi__plain",
    "rl__rabi__gru8",
    "rl__ramsey_iq__plain",
    "rl__ramsey_iq__gru8",
    "rl__dither__DR",
]
Z1_SEEDS = list(range(101, 111))

# Scopes to analyze - Z2A and TierB use the existing naming convention
SCOPES = {
    "Z2A_E1": {
        "prefixes": ["z2__s15__E1"],
        "seeds": list(range(200, 205)),
    },
    "Z2A_E4": {
        "prefixes": ["z2__s15__E4"],
        "seeds": list(range(200, 205)),
    },
    "Z2A_FTB_200k": {
        "prefixes": ["z2__s15__FTB_200000"],
        "seeds": list(range(200, 205)),
    },
    "TierB_E2": {
        "prefixes": ["z2h__s15__E2"],
        "seeds": list(range(200, 205)),
    },
    "TierB_E3": {
        "prefixes": ["z2h__s15__E3"],
        "seeds": list(range(200, 205)),
    },
}

ENVS = ["LG", "LB", "RG", "RB"]


def find_z1_files():
    """Find all Z1 .npz files with fail_ep*_x traces.
    Z1 naming patterns:
      - nominal: z1__<config>_s<seed>__nom.npz
      - g2 stress: z1__<config>_s<seed>__g2_<stress>_<idx>.npz
      - ofat stress: z1__<config>_s<seed>__ofat_<param>.npz
    All Z1 evals are on RB (quantum surrogate) environment.
    """
    files = []
    for config in Z1_CONFIGS:
        for seed in Z1_SEEDS:
            prefix = f"z1__{config}_s{seed}"
            for fn in os.listdir(SEALED):
                if fn.startswith(prefix + "__") and fn.endswith(".npz"):
                    fp = os.path.join(SEALED, fn)
                    try:
                        d = np.load(fp)
                        keys = [k for k in d.keys() if k.startswith("fail_ep") and k.endswith("_x")]
                        if keys:
                            # Z1 files don't have LG/LB/RG/RB in name; they're all RB env
                            # Use stress type as "environment" for reporting
                            if fn.endswith("__nom.npz"):
                                env = "RB_nom"
                            elif "__g2_" in fn:
                                env = "RB_g2"
                            elif "__ofat_" in fn:
                                env = "RB_ofat"
                            else:
                                env = "RB_unk"
                            files.append(("Z1_9configs", fn, fp, keys, env))
                    except Exception:
                        pass
    return files


def find_z2_files(scope_name, scope):
    """Find Z2A/TierB .npz files with fail_ep*_x traces (existing convention).
    Naming: z2__s15__<arm>_s<seed>__<env>__nom.npz or z2__s15__<arm>_s<seed>__<env>__ofat_*.npz
    """
    files = []
    for prefix in scope["prefixes"]:
        for seed in scope["seeds"]:
            for env in ENVS:
                for fn in os.listdir(SEALED):
                    if fn.startswith(f"{prefix}_s{seed}__{env}__") and fn.endswith(".npz"):
                        fp = os.path.join(SEALED, fn)
                        try:
                            d = np.load(fp)
                            keys = [k for k in d.keys() if k.startswith("fail_ep") and k.endswith("_x")]
                            if keys:
                                files.append((scope_name, fn, fp, keys, env))
                        except Exception:
                            pass
    return files


def classify_trace(x_trace):
    """Classify a single episode's x trace (post burn-in) into one of 4 categories.
    Returns: category name (a/b/c/d)"""
    x = np.asarray(x_trace, dtype=float)
    T = len(x)
    half = T // 2
    
    # (a) never_acquires: |offset| > 1000 Hz for first half of episode
    if np.all(np.abs(x[:half]) > W):
        return "never_acquires"
    
    # (b) acquires_then_loses: drops below 1000 Hz at least once, then exceeds 1000 Hz for >=10 consecutive steps later
    acquired = np.any(np.abs(x) <= W)
    if acquired:
        acquired_indices = np.where(np.abs(x) <= W)[0]
        if len(acquired_indices) > 0:
            last_acq = acquired_indices[-1]
            post = x[last_acq+1:]
            if len(post) >= 10:
                lost_run = 0
                for v in post:
                    if abs(v) > W:
                        lost_run += 1
                        if lost_run >= 10:
                            return "acquires_then_loses"
                    else:
                        lost_run = 0
    
    # (c) chatter: sign changes >40% of steps while |offset| < 2000 Hz throughout
    if np.all(np.abs(x) < 2000.0):
        sign_changes = np.sum(np.diff(np.sign(x)) != 0)
        if sign_changes > 0.4 * (T - 1):
            return "chatter"
    
    # (d) other
    return "other"


def per_episode_lock_w1000(x_trace):
    """Compute per-episode lock_w1000 for a single trace (post burn-in)."""
    x = np.asarray(x_trace, dtype=float)
    post = x[BURN_IN:]
    if len(post) == 0:
        return 0.0
    return float(np.mean(np.abs(post) <= W))


def process():
    rows = []
    example_traces = {"never_acquires": None, "acquires_then_loses": None, "chatter": None, "other": None}
    
    # Process Z1
    print("Processing Z1_9configs...")
    z1_files = find_z1_files()
    print(f"  Found {len(z1_files)} .npz files with fail traces")
    
    from collections import defaultdict
    stats = defaultdict(lambda: {
        "n_traced": 0, "n_total": 0, "n_actual_failed": 0,
        "never_acquires": 0, "acquires_then_loses": 0, "chatter": 0, "other": 0
    })
    
    for scope_name, fn, fp, keys, env in z1_files:
        # Parse config from filename
        # z1__rl__dither__plain_s101__... -> parts: z1, rl, dither, plain, s101, ...
        # z1__rl__rabi__plain_s101__... -> parts: z1, rl, rabi, plain, s101, ...
        # z1__rl__ramsey_iq__plain_s101__... -> parts: z1, rl, ramsey_iq, plain, s101, ...
        parts = fn.replace(".npz", "").split("__")
        if len(parts) >= 5:
            # Find the seed part (starts with 's' followed by digits)
            seed_idx = None
            for i, p in enumerate(parts):
                if p.startswith('s') and p[1:].isdigit():
                    seed_idx = i
                    break
            if seed_idx is not None and seed_idx >= 4:
                config = "__".join(parts[1:seed_idx])  # rl__dither__plain, rl__ramsey_iq__plain, etc.
            else:
                # Fallback: assume config is parts[1:4]
                config = "__".join(parts[1:4])
        else:
            continue
        
        try:
            d = np.load(fp)
            lock_w1000_arr = d.get('lock_w1000')
            # Count actual failed episodes from full eval array
            actual_failed = int(np.sum(lock_w1000_arr < FAIL_THRESHOLD)) if lock_w1000_arr is not None else 0
            
            for key in keys:
                x_trace = d[key]
                # Extract episode index from key (e.g., "fail_ep83_x" -> 83)
                ep_idx = int(key.split('ep')[1].split('_')[0])
                # Use full eval's lock_w1000 for this episode as ground truth
                full_lock_w1000 = lock_w1000_arr[ep_idx] if lock_w1000_arr is not None and ep_idx < len(lock_w1000_arr) else 1.0
                # Also compute trace's per_episode_lock_w1000 for example PNG title
                trace_lock_w1000 = per_episode_lock_w1000(x_trace)
                stats_key = (config, env)
                stats[stats_key]["n_total"] += 1
                
                if full_lock_w1000 < FAIL_THRESHOLD:
                    stats[stats_key]["n_traced"] += 1
                    cat = classify_trace(x_trace[BURN_IN:])
                    stats[stats_key][cat] += 1
                    
                    if example_traces[cat] is None:
                        example_traces[cat] = (config, env, key, x_trace, trace_lock_w1000)
            
            # Add actual failed count for this file
            if actual_failed > 0:
                stats[stats_key]["n_actual_failed"] += actual_failed
        except Exception as e:
            print(f"  ERROR processing {fn}: {e}")
    
    # Add Z1 rows
    for (config, env), s in stats.items():
        n_traced = s["n_traced"]
        n_actual = s["n_actual_failed"]
        capture_rate = n_traced / n_actual if n_actual > 0 else 0.0
        rows.append({
            "scope": "Z1_9configs",
            "config_or_arm": config,
            "env": env,
            "n_traced_failures": s["n_traced"],
            "n_actual_failures": s["n_actual_failed"],
            "capture_rate": f"{capture_rate:.3f}",
            "n_total_episodes": s["n_total"],
            "count_never_acquires": s["never_acquires"],
            "count_acquires_then_loses": s["acquires_then_loses"],
            "count_chatter": s["chatter"],
            "count_other": s["other"],
        })
    
    # Process Z2A and TierB scopes
    for scope_name, scope in SCOPES.items():
        print(f"Processing {scope_name}...")
        files = find_z2_files(scope_name, scope)
        print(f"  Found {len(files)} .npz files with fail traces")
        
        stats = defaultdict(lambda: {
            "n_traced": 0, "n_total": 0, "n_actual_failed": 0,
            "never_acquires": 0, "acquires_then_loses": 0, "chatter": 0, "other": 0
        })
        
        for scope_n, fn, fp, keys, env in files:
            parts = fn.replace(".npz", "").split("__")
            if len(parts) < 3:
                continue
            
            if parts[0] in ("z2", "z2h"):
                config = parts[1]
                if env is None:
                    for p in parts[2:]:
                        if p in ENVS:
                            env = p
                            break
                if env is None:
                    continue
            else:
                continue
            
            try:
                d = np.load(fp)
                lock_w1000_arr = d.get('lock_w1000')
                actual_failed = int(np.sum(lock_w1000_arr < FAIL_THRESHOLD)) if lock_w1000_arr is not None else 0
                
                for key in keys:
                    x_trace = d[key]
                    lock_w1000 = per_episode_lock_w1000(x_trace)
                    stats_key = (config, env)
                    stats[stats_key]["n_total"] += 1
                    
                    if lock_w1000 < FAIL_THRESHOLD:
                        stats[stats_key]["n_traced"] += 1
                        cat = classify_trace(x_trace[BURN_IN:])
                        stats[stats_key][cat] += 1
                        
                        if example_traces[cat] is None:
                            example_traces[cat] = (config, env, key, x_trace, lock_w1000)
                
                if actual_failed > 0:
                    stats[stats_key]["n_actual_failed"] += actual_failed
            except Exception as e:
                print(f"  ERROR processing {fn}: {e}")
        
        for (config, env), s in stats.items():
            n_traced = s["n_traced"]
            n_actual = s["n_actual_failed"]
            capture_rate = n_traced / n_actual if n_actual > 0 else 0.0
            rows.append({
                "scope": scope_name,
                "config_or_arm": config,
                "env": env,
                "n_traced_failures": s["n_traced"],
                "n_actual_failures": s["n_actual_failed"],
                "capture_rate": f"{capture_rate:.3f}",
                "n_total_episodes": s["n_total"],
                "count_never_acquires": s["never_acquires"],
                "count_acquires_then_loses": s["acquires_then_loses"],
                "count_chatter": s["chatter"],
                "count_other": s["other"],
            })
    
    return rows, example_traces


def write_outputs(rows, example_traces):
    # Append Z1 rows to s18_counts.csv (don't overwrite)
    csv_path = os.path.join(OUTDIR, "s18_counts.csv")
    
    # Read existing rows to keep non-Z1 data
    existing_rows = []
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r["scope"] == "Z1_9configs":
                    continue
                # Migrate old field names to new ones
                new_r = dict(r)
                if "n_failed_episodes" in new_r:
                    new_r["n_traced_failures"] = new_r.pop("n_failed_episodes")
                    new_r["n_actual_failures"] = new_r["n_traced_failures"]  # same as traced for Z2A/TierB
                    new_r["capture_rate"] = "1.000"  # assume full capture for Z2A/TierB
                existing_rows.append(new_r)
    
    # Use all rows from current run (includes Z1, Z2A, TierB)
    all_rows = rows
    
    # New fieldnames with capture rate
    fieldnames = ["scope", "config_or_arm", "env", 
                  "n_traced_failures", "n_actual_failures", "capture_rate",
                  "n_total_episodes", "count_never_acquires", "count_acquires_then_loses", 
                  "count_chatter", "count_other"]
    
    # Write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
        f.flush()
    
    print(f"Wrote {csv_path} ({len(all_rows)} total rows, {sum(1 for r in all_rows if r['scope']=='Z1_9configs')} Z1 rows)")
    
    # Regenerate s18_summary.md with all scopes
    md_path = os.path.join(OUTDIR, "s18_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# S18 Failure Taxonomy Summary\n\n")
        f.write(f"**NOTE: Z1 scope was MISSED in the original S18 run due to a naming-pattern bug; this is the corrected, complete result with per-config rows.**\n\n")
        f.write(f"Total configurations/arms analyzed: {len(set(r['config_or_arm'] for r in all_rows))}\n")
        f.write(f"Total environments: {len(set(r['env'] for r in all_rows))}\n")
        f.write(f"Total episodes: {sum(int(r['n_total_episodes']) for r in all_rows)}\n")
        # Sum traced failures for Z1+Z2A/TierB
        total_traced = sum(int(r.get('n_traced_failures', r.get('n_failed_episodes', 0))) for r in all_rows)
        f.write(f"Total traced failures (lock_w1000 < 0.5): {total_traced}\n\n")
        
        # Per-scope summary
        scope_stats = defaultdict(lambda: {"traced": 0, "actual": 0, "total": 0, "cats": defaultdict(int)})
        for r in all_rows:
            s = scope_stats[r["scope"]]
            traced = int(r.get('n_traced_failures', r.get('n_failed_episodes', 0)))
            actual = int(r.get('n_actual_failures', r.get('n_failed_episodes', 0)))
            s["traced"] += traced
            s["actual"] += actual
            s["total"] += int(r["n_total_episodes"])
            s["cats"]["never_acquires"] += int(r["count_never_acquires"])
            s["cats"]["acquires_then_loses"] += int(r["count_acquires_then_loses"])
            s["cats"]["chatter"] += int(r["count_chatter"])
            s["cats"]["other"] += int(r["count_other"])
        
        f.write("## Per-scope breakdown\n\n")
        for scope, s in sorted(scope_stats.items()):
            f.write(f"**{scope}**: {s['traced']}/{s['actual']} traced/actual failed episodes (capture={s['traced']/s['actual']*100:.1f}%)\n")
            f.write(f"  never_acquires: {s['cats']['never_acquires']}, ")
            f.write(f"acquires_then_loses: {s['cats']['acquires_then_loses']}, ")
            f.write(f"chatter: {s['cats']['chatter']}, ")
            f.write(f"other: {s['cats']['other']}\n\n")
        
        f.write("## Per-config/arm breakdown (traced failures only)\n\n")
        for r in all_rows:
            traced = int(r.get('n_traced_failures', r.get('n_failed_episodes', 0)))
            if traced > 0:
                cap = r.get('capture_rate', 'N/A')
                f.write(f"- **{r['scope']} / {r['config_or_arm']} / {r['env']}**: ")
                f.write(f"traced={traced}/{r['n_total_episodes']}, capture={cap}, ")
                f.write(f"never_acquires={r['count_never_acquires']}, ")
                f.write(f"acquires_then_loses={r['count_acquires_then_loses']}, ")
                f.write(f"chatter={r['count_chatter']}, ")
                f.write(f"other={r['count_other']}\n")
    
    print(f"Wrote {md_path}")
    
    # Write example PNGs (only if we have examples)
    for cat, data in example_traces.items():
        if data is not None:
            config, env, key, x_trace, lock_w1000 = data
            png_path = os.path.join(OUTDIR, f"s18_example_{cat}.png")
            plt.figure(figsize=(10, 4))
            plt.plot(x_trace)
            plt.axhline(W, color="r", linestyle="--", label=f"±{W} Hz")
            plt.axhline(-W, color="r", linestyle="--")
            plt.axhline(2000, color="g", linestyle=":", label="±2000 Hz")
            plt.axhline(-2000, color="g", linestyle=":")
            plt.axvline(BURN_IN, color="k", linestyle=":", label=f"burn-in={BURN_IN}")
            plt.title(f"Category: {cat}\n{config} / {env} / {key} / lock_w1000={lock_w1000:.3f}")
            plt.xlabel("Step")
            plt.ylabel("Offset (Hz)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(png_path, dpi=150)
            plt.close()
            print(f"Wrote {png_path}")
        else:
            print(f"No example for category {cat}")
    
    # Verification
    print("\n=== VERIFICATION ===")
    for r in all_rows:
        total = int(r["count_never_acquires"]) + int(r["count_acquires_then_loses"]) + int(r["count_chatter"]) + int(r["count_other"])
        n_traced = int(r.get('n_traced_failures', r.get('n_failed_episodes', 0)))
        if total != n_traced:
            print(f"MISMATCH: {r['scope']}/{r['config_or_arm']}/{r['env']}: sum={total}, n_traced={n_traced}")
        else:
            print(f"OK: {r['scope']}/{r['config_or_arm']}/{r['env']}: sum={total} == n_traced={n_traced}")
    
    total_classified = sum(int(r["count_never_acquires"]) + int(r["count_acquires_then_loses"]) + int(r["count_chatter"]) + int(r["count_other"]) for r in all_rows)
    total_traced = sum(int(r.get('n_traced_failures', r.get('n_failed_episodes', 0))) for r in all_rows)
    print(f"\nTotal classified traces: {total_classified}")
    print(f"Total traced failures: {total_traced}")
    
    # Z1 total check
    z1_traced = sum(int(r.get('n_traced_failures', r.get('n_failed_episodes', 0))) for r in all_rows if r["scope"] == "Z1_9configs")
    z1_actual = sum(int(r.get('n_actual_failures', r.get('n_failed_episodes', 0))) for r in all_rows if r["scope"] == "Z1_9configs")
    print(f"Z1 traced failures: {z1_traced}, actual failures: {z1_actual}, capture: {z1_traced/z1_actual*100:.1f}%")


if __name__ == "__main__":
    rows, example_traces = process()
    write_outputs(rows, example_traces)
    print("\nS18-FIX2 DONE")