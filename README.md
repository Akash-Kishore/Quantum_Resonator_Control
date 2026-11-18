# Quantum Resonator Control — ARCS/QRC Pipeline

This repository contains a curated snapshot of the **PPO-based RL resonator calibration pipeline** developed for the **IEEE Access case study** (Mini Project Sem 4).

## Project Overview
- **Target**: Quantum resonator frequency locking via PPO RL control
- **Environment**: Custom BatchEnv + QuantumCalibGym with surrogate/AER engines
- **Methods**: PPO (dither, rabi, ramsey_iq probes), PID, HillClimb, Bayes (nominal/oracle)
- **Key Results**: C-1..C-8 confirmatory contrasts (seeds 101-110), TR-1/TR-2/TR-4 sealed, Tier B gap decomposition

## Repository Contents

### Core Pipeline Code (`qstudy/`)
- `train_ppo.py`, `evaluate.py`, `eval_set.py`, `envs.py`, `engines.py`, `controllers.py` — frozen pipeline (hash-verified)
- `contrasts.py`, `stats.py`, `db.py`, `lockstates.py`, `s18_failure_taxonomy.py`, etc. — analysis tools
- `configs/frozen/` — 9 frozen confirmatory configs + selection.json (PREREG sealed)
- `configs_tierb/` — Tier B configs (E2=env LB, E3=env RG)
- `configs_sens2r/` — SENS2R configs

### Curated Results (`analysis_artifacts/`)
- `tables_paper/` — 8 CSV tables (T1–T7, CAPTIONS.md)
- `figs_paper/` — 28 figures (PDF + PNG, 3.5"/7.16" widths)
- `traj_v3/` — 9 NPZ trajectory files + 18 PNG plots + traj_summary_v3.csv
- `s18/` — Failure taxonomy: s18_counts.csv (290 rows), s18_summary.md
- `tierb3_tr4_gap.json` — TR-4 result + 4×4 transfer matrix + gap decomposition
- `sens2r_p2t.json` — SENS2R P2T result (FTB raw vs E4 200k)
- `events.md` — Full project event log
- `RESUME_CARD.md` — Current project state
- `VERIFY_FINAL_20261118.md` — Pre-paper integrity gate report
- `tools/` — 91 analysis scripts (.py)

### Sample Trained Policies (`model_checkpoints/`)
5 checkpoint pairs (best_model.zip + vec_normalize.pkl, ~0.42 MB each):
- `E1/` — LG zero-shot (seed 200)
- `E4/` — RB from-scratch 2M (seed 200)
- `FTB_200000/` — Fine-tune B 200k steps (seed 200)
- `E2/` — LB zero-shot (seed 200)
- `E3/` — RG zero-shot (seed 200)

### Prompts (`prompts/`)
All QSTUDY_PROMPT*.md files documenting the project specification and task history.

## What's NOT Included (Size Exclusions)
- **Full sealed dataset**: 30,106 files / 2.5 GB (available on request)
- **Training runs**: qstudy/runs/ (1,294 files, 166 MB) + qstudy/runs_snap/ (168 files, 18 MB)
- **RL training checkpoints**: rl_training/trained_models/ (1,415 .zip files, 1.7 GB)
- **Raw simulator output**: Schematics_Simulations/Envelope Detector.raw (2.4 GB)
- **Database**: qstudy/db/qstudy.sqlite (17.5 MB)

These are excluded via `.gitignore`. The curated snapshot above represents the complete paper-worthy deliverable (~50 MB total).

## Verification
- Code freeze: 6 pipeline files match AS_RUN_v1.zip hashes
- Prereg integrity: PREREG.md / PREREG2.md hashes match
- Sealed results: 276/300 random sample matches 26-Oct manifest
- Database: 21 contrast rows correct (C-1..C-8 FINAL, TR-1/2/4 FINAL, TR-3/5 PENDING)
- All checkpoints < 1 MB each

## Target Journal
IEEE Access — Case Study: "PPO RL for Quantum Resonator Calibration"