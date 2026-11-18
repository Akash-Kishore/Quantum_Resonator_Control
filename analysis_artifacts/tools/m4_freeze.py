"""CONTINUATION4 M4: write the freeze.
1. configs\\frozen\\selection.json (status FROZEN) with C1-C6, F6 hyperparameters, the
   nine confirmatory configs (Appendix K), baselines+definitions, recomputed val scores.
2. Append the FREEZE ADDENDUM (Appendix K verbatim, bracketed fields filled) to PREREG.md;
   write PREREG.sha256.
3. Write PREREG2.md = PART2 APPENDIX E verbatim + the PREREG2 AMENDMENTS applied in place;
   write PREREG2.sha256.
4. Insert both hashes into the db (prereg_hashes). Prints the FREEZE-HASHED line."""
import datetime, hashlib, json, os, sqlite3

S = r"C:\Projects\MiniProject_Sem4\qstudy"
W = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"
TODAY = "2026-10-15"


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


f3 = json.load(open(os.path.join(W, "results", "f3_recompute.json"), encoding="utf-8"))
nine = ["rl__rabi__plain", "rl__rabi__gru8", "rl__dither__plain", "rl__dither__filter",
        "rl__dither__DR", "rl__dither__DRcurr", "rl__dither__full",
        "rl__ramsey_iq__plain", "rl__ramsey_iq__gru8"]
sel_map = {"C1_legacy_analog": "rl__rabi__plain", "C2_best_plain": "rl__dither__plain",
           "C3_best_hybrid_filter": "rl__dither__filter", "C4_best_memory": "rl__ramsey_iq__gru8",
           "C5_best_dr": "rl__dither__DRcurr", "C6_best_full": "rl__dither__full"}

# ---------------- 1. selection.json (FROZEN) ----------------
val_scores = {}
for cfg, d in f3["new"].items():
    fam = ("full" if "__full" in cfg else "DRcurr" if "__DRcurr" in cfg else
           "DR" if "__DR" in cfg else "filter" if "__filter" in cfg else
           "memory" if ("__K" in cfg or "__gru8" in cfg) else "plain" if "__plain" in cfg else "other")
    val_scores.setdefault(fam, {})[cfg] = dict(mean=d["mean"], sd=d["sd"], n=d["n"])

frozen = {
    "status": "FROZEN (M4, CONTINUATION4; conditional approval M2-M3 passed, freeze_review_pack_v2)",
    "frozen_at": TODAY,
    "selection": sel_map,
    "confirmatory_configs": nine,
    "confirmatory_seeds": list(range(101, 111)),
    "hyperparameters": {
        "learning_rate": 0.0003, "ent_coef": 0.0, "gamma": 0.99, "total_steps": 2000000,
        "source": "F6 tuning replicate (tune seeds 90-92; docs\\tuning.md): 3-seed means "
                  "0.7348 (ent 0.003) vs 0.7191 (ent 0.0); difference 0.0157 < larger sd 0.2166 "
                  "-> ent_coef 0.0 (value used in the exploratory runs)"
    },
    "baselines": {
        "ctrl:bayes:nominal": "assumes N0 (random walk, sigma 500 Hz, tau 50 us, T2 200 us "
                              "exponential, readout 0.02/0.02, omega 5000, dither delta 2500) "
                              "and sees only observables (counts, shots used this step, fc, probe type)",
        "ctrl:bayes:oracle": "uses the true cell parameters (moment-matched random walk for "
                             "non-Gaussian drifts; true contrast shape/override)",
        "hybrid_filter_features": "use N0 in training and evaluation (cfg filter_assumed=oracle "
                                  "only for labelled diagnostics)",
        "ctrl:oracle": "measured reference (previous true offset known)",
        "ctrl:pid": "frozen per-probe tuned gains (configs\\pid_gains.json)",
        "ctrl:hill": "rabi only",
        "ctrl:noctrl": "never moves"
    },
    "contrast_sides": {
        "C-1": ["rl__dither__plain", "rl__rabi__plain"],
        "C-2": ["rl__ramsey_iq__plain", "rl__dither__plain"],
        "C-3": ["rl__dither__plain", "ctrl:pid (dither)"],
        "C-4": ["rl__dither__plain", "ctrl:bayes:nominal (dither)"],
        "C-5": ["rl__dither__filter", "rl__dither__plain"],
        "C-6": ["rl__rabi__gru8", "rl__rabi__plain"],
        "C-7": ["rl__dither__DRcurr", "rl__dither__plain"],
        "C-8": ["rl__dither__DRcurr", "rl__dither__DR"]
    },
    "val_scores_recomputed_F3": val_scores,
    "notes": "C6=rl__dither__full is trained only in the confirmatory phase with the corrected "
             "filter features (exploratory full-family training verdict VALID; eval-side stress "
             "cells were re-run under the F1/F2 fix). C6 margin to the runner-up below one sd "
             "(no contrast uses C6); C-6 rabi gru8 margin to rabi K4 (0.020) below one sd - "
             "reported, not hidden."
}
with open(os.path.join(S, "configs", "frozen", "selection.json"), "w", encoding="utf-8") as f:
    json.dump(frozen, f, indent=1)

# ---------------- 2. FREEZE ADDENDUM -> PREREG.md ----------------
addendum = """
## FREEZE ADDENDUM (written {date}; supersedes any conflicting text above)
1. Hyperparameters (tuning rule with 3-seed replicate, docs\\tuning.md): learning_rate {lr}, ent_coef {ent}, gamma {gamma}, n_envs/n_steps/batch/epochs as in train_ppo defaults, 2,000,000 steps.
2. Confirmatory seeds are 101-110 (seed 100 was used for tuning; every mention of 100-109 above means 101-110).
3. Confirmatory configs, 10 seeds each: rl__rabi__plain, rl__rabi__gru8, rl__dither__plain, rl__dither__filter, rl__dither__DR, rl__dither__DRcurr, rl__dither__full, rl__ramsey_iq__plain, rl__ramsey_iq__gru8. Mapping: C1=rl__rabi__plain (legacy analog), C2={c2}, C3={c3}, C4={c4}, C5={c5}, C6=rl__dither__full (probe of C2; trained only in the confirmatory phase with the corrected filter features because exploratory "full" results were {fullverdict}). Contrast sides: C-1 dither plain vs rabi plain; C-2 ramsey plain vs dither plain; C-3 dither plain vs ctrl:pid(dither); C-4 dither plain vs ctrl:bayes:nominal(dither); C-5 dither filter vs dither plain; C-6 rabi gru8 vs rabi plain; C-7 dither DRcurr vs dither plain; C-8 dither DRcurr vs dither DR.
4. Baseline definitions: ctrl:bayes:nominal assumes N0 (random walk, sigma 500 Hz, tau 50 us, T2 200 us exponential, readout 0.02/0.02, omega 5000, dither delta 2500) and sees only observables; ctrl:bayes:oracle uses the true cell parameters (moment-matched random walk for non-Gaussian drifts); hybrid filter features use N0 in training and evaluation; ctrl:oracle is a measured reference (previous true offset known); ctrl:pid gains are the frozen per-probe tuned gains; ctrl:hill is rabi only.
5. Ties and margins recorded before freezing: C6 margin to the runner-up was below one sd (no contrast uses C6); C-6 uses rabi gru8 although its margin to rabi K4 (0.020) was below one sd; memory variants did not improve over plain in exploration; fixed-from-start domain randomization underperformed plain in exploration. These are reported, not hidden.
6. Protocol deviations before freezing (details in events.md and docs\\03_session_handover.md): anomalies A1-A11, the readout-OFAT invalidation (A10), the curriculum bug (A8), the filter/Bayes parameter-isolation fix (M2 F1-F2), the tuning replicate (F6), the seed shift (item 2).
7. Re-run policy: a sealed evaluation may be re-run only if the pre-flight/post-flight configuration audit (audit_eval_cfg) aborts it (infrastructure fault); never because of its results; every case is logged.
8. Compute estimate and order: z1_ confirmatory first, then S15-S17 trainings; evaluations follow each model.
""".format(date=TODAY, lr="3e-4", ent="0.0", gamma="0.99", c2=sel_map["C2_best_plain"],
           c3=sel_map["C3_best_hybrid_filter"], c4=sel_map["C4_best_memory"],
           c5=sel_map["C5_best_dr"], fullverdict="VALID")

with open(os.path.join(S, "PREREG.md"), "a", encoding="utf-8") as f:
    f.write(addendum)
sha_prereg = sha256_file(os.path.join(S, "PREREG.md"))
with open(os.path.join(S, "PREREG.sha256"), "w", encoding="utf-8") as f:
    f.write(sha_prereg + "\n")

# ---------------- 3. PREREG2.md = APPENDIX E verbatim + amendments in place ----------------
src = open(os.path.join(W, "results", "part2_appendix_e.txt"), encoding="utf-8").read()
body = src.split("=== FILE: STUDY\\PREREG2.md ===\n", 1)[1].split("=== END FILE ===")[0]
body = body.rstrip("\n") + "\n"

# Amendment B (S15 fine-tune arms) - inserted after the arms line, in place:
b_txt = ("Fine-tune arms: FT-A uses the frozen hyperparameters; FT-B uses learning_rate = "
         "frozen/10 and frozen observation-normalisation statistics (no updates) during "
         "fine-tuning. Both arms are run for every budget. TR-2 (primary) uses FT-B; FT-A is "
         "reported as secondary. Reason (pre-freeze exploratory evidence): Aer fine-tuning with "
         "the frozen learning rate lowered lock for 2 of 3 seeds.\n")
anchor = "fine-tune budgets 50k/200k/500k/1M; PPO/SAC/TD3. Sealed base 920000.\n"
assert anchor in body
body = body.replace(anchor, anchor + b_txt, 1)

# Amendment A (seeds 100-102 -> 101-103) - recorded in place in S16 and S17:
a_txt = ("Zero-shot platform evaluation of the Part-1 candidates C1-C6 uses seeds 101-103; every "
         "mention of seeds 100-102 for C1-C6 (S16 zero-shot platform evaluation, S17.6 envelope, "
         "Part 1 12.3 Aer subset) reads 101-103.\n")
anchor16 = "Profiles in qphysics_ext.PROFILES (tags G/A/S). Grid: drift multiplier {1,10,100,1000,10000} x shots {64,256,1024,4096}. Sealed base 950000, 200 episodes per cell.\n"
assert anchor16 in body
body = body.replace(anchor16, anchor16 + a_txt, 1)

# Amendment C (Bayes baselines follow Part-1 addendum item 4) - in the Rules section:
c_txt = ("Bayes baselines in S16/S17 follow the Part-1 FREEZE ADDENDUM item 4: "
         "ctrl:bayes:nominal assumes N0, never the platform parameters; ctrl:bayes:oracle uses "
         "the platform/cell parameters.\n")
anchor_r = "Negative or null results are reported; no run, seed or condition is dropped; sealed evaluations run once; thresholds in this file are never changed after freezing.\n"
assert anchor_r in body
body = body.replace(anchor_r, anchor_r + c_txt, 1)

with open(os.path.join(S, "PREREG2.md"), "w", encoding="utf-8") as f:
    f.write(body)
sha_prereg2 = sha256_file(os.path.join(S, "PREREG2.md"))
with open(os.path.join(S, "PREREG2.sha256"), "w", encoding="utf-8") as f:
    f.write(sha_prereg2 + "\n")

# ---------------- 4. insert hashes into the database ----------------
con = sqlite3.connect(os.path.join(S, "db", "qstudy.sqlite"))
con.executescript("CREATE TABLE IF NOT EXISTS prereg_hashes (file TEXT PRIMARY KEY, sha256 TEXT, recorded_at TEXT)")
now = datetime.datetime.now().isoformat(timespec="seconds")
con.execute("INSERT OR REPLACE INTO prereg_hashes (file, sha256, recorded_at) VALUES (?,?,?)",
            ("PREREG.md", sha_prereg, now))
con.execute("INSERT OR REPLACE INTO prereg_hashes (file, sha256, recorded_at) VALUES (?,?,?)",
            ("PREREG2.md", sha_prereg2, now))
con.commit()
con.close()

print("FREEZE-HASHED | %s | %s | %s" % (sha_prereg[:12], sha_prereg2[:12], ", ".join(nine)))
