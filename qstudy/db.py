"""qstudy db — SQLite database of the study. Schema created now (Stage 3.6); filled by later stages.
Usage: python db.py create | ingest <results_dir> | export_csv | describe | integrity_check
"""
import datetime, glob, json, os, sqlite3, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "db", "qstudy.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS evals (
  eval_id TEXT PRIMARY KEY, method TEXT, model_dir TEXT, cond_id TEXT, split TEXT,
  episodes INTEGER, seed_base INTEGER, lock_w500_mean REAL, lock_w500_sd REAL,
  lock_w1000_mean REAL, lock_w1000_sd REAL, lock_w2000_mean REAL, lock_w2000_sd REAL,
  lock_w3333_mean REAL, lock_w3333_sd REAL, lock_w5000_mean REAL, lock_w5000_sd REAL,
  rms_mean REAL, rms_sd REAL, med_abs_mean REAL, med_abs_sd REAL,
  p95_abs_mean REAL, p95_abs_sd REAL, max_abs_mean REAL, max_abs_sd REAL,
  lock_lost_rate REAL, mean_reward_mean REAL, mean_reward_sd REAL,
  git_head TEXT, versions TEXT, started TEXT, finished TEXT, cmd TEXT,
  npz_path TEXT, npz_sha256 TEXT, ingested_at TEXT
);
CREATE TABLE IF NOT EXISTS validation (
  id INTEGER PRIMARY KEY AUTOINCREMENT, test TEXT, kind TEXT, cell TEXT,
  engine TEXT, metric TEXT, value REAL, expected TEXT, observed TEXT,
  pass INTEGER, evidence_file TEXT, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS runs (
  run_id TEXT PRIMARY KEY, config_name TEXT, seed INTEGER, split TEXT,
  config_hash TEXT, steps INTEGER, wall_seconds REAL, best_val REAL,
  val_history TEXT, versions TEXT, cmd TEXT, finished TEXT
);
CREATE TABLE IF NOT EXISTS contrasts (
  id INTEGER PRIMARY KEY AUTOINCREMENT, contrast_id TEXT, endpoint TEXT,
  method_a TEXT, method_b TEXT, n_a INTEGER, n_b INTEGER,
  statistic REAL, p_value REAL, effect_size REAL, ci_low REAL, ci_high REAL,
  holm_p REAL, exploratory INTEGER DEFAULT 0, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS anomalies (
  id INTEGER PRIMARY KEY AUTOINCREMENT, what TEXT, detail TEXT, evidence TEXT, recorded_at TEXT
);
CREATE INDEX IF NOT EXISTS ix_evals_method ON evals(method);
CREATE INDEX IF NOT EXISTS ix_evals_cond ON evals(cond_id);
"""

# Part-2 APPENDIX G: idempotent CREATE TABLE IF NOT EXISTS additions (S15-S19).
APPENDIX_G_SCHEMA = """
CREATE TABLE IF NOT EXISTS prereg_hashes (
  file TEXT PRIMARY KEY, sha256 TEXT, recorded_at TEXT
);
CREATE TABLE IF NOT EXISTS platform_profiles (
  profile TEXT, field TEXT, value TEXT, tag TEXT, citation_status TEXT DEFAULT 'CITATION NEEDED'
);
CREATE TABLE IF NOT EXISTS env_variants (
  env_id TEXT PRIMARY KEY, lineshape TEXT, noise TEXT, engine TEXT, notes TEXT
);
CREATE TABLE IF NOT EXISTS transfer (
  row_id TEXT PRIMARY KEY, train_env TEXT, eval_env TEXT, algo TEXT, arm TEXT,
  seed INT, ft_steps INT, lock_w1000 REAL, p2_macro REAL, eval_id TEXT
);
CREATE TABLE IF NOT EXISTS correlations (
  corr_id TEXT PRIMARY KEY, level TEXT, x_name TEXT, y_name TEXT, n INT,
  spearman REAL, kendall REAL, ci_lo REAL, ci_hi REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS lock_states (
  eval_id TEXT, method TEXT, env TEXT, profile TEXT, cond_id TEXT, p_locked REAL,
  p_transit REAL, p_lost REAL, t_first_lock_mean REAL, relock_mean REAL,
  losses_per_episode REAL, trans_json TEXT
);
CREATE TABLE IF NOT EXISTS failures (
  eval_id TEXT, episode INT, onset INT, label TEXT, evidence_json TEXT
);
CREATE TABLE IF NOT EXISTS platform_results (
  eval_id TEXT, profile TEXT, method TEXT, shots INT, drift_mult REAL,
  lock_w1000 REAL, rms REAL, cvar95 REAL, m_star REAL
);
CREATE TABLE IF NOT EXISTS adaptive_results (
  eval_id TEXT, policy TEXT, lambda REAL, fixed_mode TEXT, fixed_shots INT,
  mean_shots REAL, lock_w1000 REAL, pareto_area REAL
);
CREATE TABLE IF NOT EXISTS changepoint (
  eval_id TEXT, method TEXT, detector INT, detect_rate REAL, delay_mean REAL,
  false_alarm_per100 REAL, recovery_mean REAL, max_err_after_jump REAL
);
CREATE TABLE IF NOT EXISTS calibration (
  eval_id TEXT, probe TEXT, level REAL, coverage REAL, mean_width REAL, log_score REAL
);
CREATE TABLE IF NOT EXISTS complexity (
  item TEXT, param TEXT, value REAL, unit TEXT, p50 REAL, p95 REAL, p99 REAL,
  peak_rss_mb REAL, notes TEXT
);
CREATE TABLE IF NOT EXISTS realism_audit (
  item TEXT, tag TEXT, sensitivity_rank_change REAL, notes TEXT
);
"""

# Part-2 O5: nullable risk-metric columns added to evals (existing rows stay NULL).
O5_COLUMNS = [
    ("cvar95_rms", "REAL"), ("cvar95_lockmiss", "REAL"), ("p95_rms", "REAL"),
    ("lock_loss_prob", "REAL"), ("first_lock_time_mean", "REAL"), ("first_lock_time_never_frac", "REAL"),
]


def connect():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.executescript(APPENDIX_G_SCHEMA)
    for name, typ in O5_COLUMNS:
        try:
            con.execute("ALTER TABLE evals ADD COLUMN %s %s" % (name, typ))
        except sqlite3.OperationalError:
            pass  # column already exists
    con.commit()
    return con


def eval_id_from(j):
    return "%s|%s|%s|%s|%s" % (j.get("method"), j.get("model_dir") or "", j.get("cond_id"),
                               j.get("split"), j.get("seed_base"))


def ingest(results_dir):
    con = connect()
    n = 0
    for p in sorted(glob.glob(os.path.join(results_dir, "**", "*.json"), recursive=True)):
        if os.path.basename(p).startswith("train_summary"):
            continue
        if "_invalid" in p.lower():
            continue  # M2 F3: exclude every *_INVALID* directory / INVALID-tagged result
        try:
            j = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue
        if "method" not in j or "lock_w1000_mean" not in j:
            continue
        eid = eval_id_from(j)
        cur = con.execute("SELECT 1 FROM evals WHERE eval_id=?", (eid,))
        if cur.fetchone():
            continue  # idempotent: skip rows whose eval_id exists
        cols = {c[1] for c in con.execute("PRAGMA table_info(evals)")}
        row = {k: j.get(k) for k in cols if k in j}
        row["eval_id"] = eid
        row["versions"] = json.dumps(j.get("versions", {}))
        row["ingested_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        keys = ",".join(row)
        ph = ",".join("?" for _ in row)
        con.execute("INSERT OR IGNORE INTO evals (%s) VALUES (%s)" % (keys, ph), tuple(row.values()))
        n += 1
    con.commit()
    con.close()
    print("INGESTED %d new eval rows from %s" % (n, results_dir))
    return n


def ingest_validation():
    """Ingest results/validation/*.json rows into the validation table (idempotent by (test,cell))."""
    import glob
    con = connect()
    vdir = os.path.join(HERE, "results", "validation")
    n = 0
    for p in sorted(glob.glob(os.path.join(vdir, "*.json"))):
        j = json.load(open(p, encoding="utf-8"))
        test = os.path.basename(p).split(".")[0]
        when = datetime.datetime.now().isoformat(timespec="seconds")
        if "rows" in j:
            for r in j["rows"]:
                cell = "%s|%s" % (r.get("kind", ""), r.get("delta", r.get("shots", "")))
                cur = con.execute("SELECT 1 FROM validation WHERE test=? AND cell=?", (test, cell))
                if cur.fetchone():
                    continue
                con.execute("INSERT INTO validation (test, kind, cell, engine, metric, value, expected, observed, pass, evidence_file, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (test, str(r.get("kind", "")) + "|" + str(r.get("noise", r.get("drift", ""))), cell, "surrogate-vs-aer",
                             "p_aer" if "p_aer" in r else ("ks_p" if "p_value" in r else "value"),
                             float(r.get("p_aer", r.get("p_value", 0)) ), "", "", 1 if r.get("pass_") else 0, p, when))
                n += 1
        else:
            key = "summary|%s" % test
            cur = con.execute("SELECT 1 FROM validation WHERE test=? AND cell=?", (test, key))
            if not cur.fetchone():
                con.execute("INSERT INTO validation (test, kind, cell, engine, metric, value, expected, observed, pass, evidence_file, recorded_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (test, "summary", key, "surrogate-vs-aer", "pass", 1.0 if j.get("pass") else 0.0, "", json.dumps(j)[:500], 1 if j.get("pass") else 0, p, when))
                n += 1
    con.commit()
    con.close()
    print("INGESTED %d validation rows" % n)
    return n


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "create"
    if cmd == "create":
        connect().close()
        print("DB ready:", DB)
    elif cmd == "ingest":
        ingest(sys.argv[2])
    elif cmd == "ingest_validation":
        ingest_validation()
    elif cmd == "integrity_check":
        con = connect()
        print(con.execute("PRAGMA integrity_check").fetchone()[0])
        con.close()
    else:
        print("unknown command (Stage 3 stub):", cmd)
        print("full export_csv/describe implemented in Stage 13")


if __name__ == "__main__":
    main()
