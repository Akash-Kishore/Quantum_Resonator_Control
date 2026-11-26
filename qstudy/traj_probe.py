"""TRAJ1c: validation-split trajectory probe (fixed).
Regenerates trajectories on VALIDATION split (seed base 500000) for specified models.
Records per-episode: MAE (Hz), lock_w1000. No label column.
Produces plots and traj_summary_v3.csv in WORK\results\traj_v3\.
"""
import argparse
import datetime
import json
import os
import pickle
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
WORK = r"C:\Projects\MiniProject_Sem4_QSTUDY_WORK"

from envs import BatchEnv, DEFAULT_CFG
from stable_baselines3 import PPO

VAL_SEED_BASE = 500000
N_EPISODES = 5
HORIZON = 200

MODELS = [
    {"name": "E1_zero_shot_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__E1\seed_200", "cond": "RB", "model_seed": 200, "env": "hybrid_rabi"},
    {"name": "E1_zero_shot_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__E1\seed_202", "cond": "RB", "model_seed": 202, "env": "hybrid_rabi"},
    {"name": "E4_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__E4\seed_200", "cond": "RB", "model_seed": 200, "env": "surrogate_rabi"},
    {"name": "E4_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__E4\seed_202", "cond": "RB", "model_seed": 202, "env": "surrogate_rabi"},
    {"name": "FTB_200000_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__FTB_200000\seed_200", "cond": "RB", "model_seed": 200, "env": "surrogate_rabi"},
    {"name": "FTB_200000_RB", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\s15__FTB_200000\seed_202", "cond": "RB", "model_seed": 202, "env": "surrogate_rabi"},
    {"name": "Z1_dither_plain", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\rl__dither__plain\seed_101", "cond": "nominal", "model_seed": 101, "env": "dither"},
    {"name": "Z1_dither_plain", "model_dir": r"C:\Projects\MiniProject_Sem4\qstudy\runs\rl__dither__plain\seed_102", "cond": "nominal", "model_seed": 102, "env": "dither"},
]


def load_rl_frozen(model_dir):
    """Load PPO model with VecNormalize frozen (training=False, norm_reward=False)."""
    model = PPO.load(os.path.join(model_dir, "best_model.zip"), device="cpu")
    with open(os.path.join(model_dir, "vec_normalize.pkl"), "rb") as f:
        vn = pickle.load(f)
    vn.training = False
    vn.norm_reward = False
    obs_rms = vn.obs_rms
    clip = getattr(vn, "clip_obs", 10.0)
    eps = getattr(vn, "epsilon", 1e-8)

    def norm(obs):
        return np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + eps), -clip, clip)

    return model, norm


def make_env(cfg, B, seed_base):
    return BatchEnv(cfg, B, seed_base)


def run_episodes_rl(model, norm, cfg, seed_base, n_episodes):
    env = make_env(cfg, n_episodes, seed_base)
    obs = env.reset()
    T = env.T
    e_tr = np.zeros((T, n_episodes))
    fc_tr = np.zeros((T, n_episodes))
    x_tr = np.zeros((T, n_episodes))
    a_tr = np.zeros((T, n_episodes))
    for t in range(T):
        obs_norm = norm(obs)
        a, _ = model.predict(obs_norm, deterministic=True)
        a = np.asarray(a, dtype=float).reshape(-1, 1)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
        fc_tr[t] = info["fc"]
        x_tr[t] = info["x"]
        a_tr[t] = a.reshape(-1)
    return e_tr, fc_tr, x_tr, a_tr, env


def run_episodes_hillclimb(cfg, seed_base, n_episodes):
    from controllers import HillClimb
    env = make_env(cfg, n_episodes, seed_base)
    ctrl = HillClimb(cfg, env)
    obs = env.reset()
    T = env.T
    e_tr = np.zeros((T, n_episodes))
    fc_tr = np.zeros((T, n_episodes))
    x_tr = np.zeros((T, n_episodes))
    a_tr = np.zeros((T, n_episodes))
    for t in range(T):
        a = ctrl.act(None).reshape(-1, 1)
        obs, rew, done, info = env.step(a)
        e_tr[t] = info["e"]
        fc_tr[t] = info["fc"]
        x_tr[t] = info["x"]
        a_tr[t] = a.reshape(-1)
    return e_tr, fc_tr, x_tr, a_tr, env


def compute_metrics(e_tr, env):
    """Compute per-episode metrics: MAE, lock_w1000."""
    T, B = e_tr.shape
    bi = int(env.cfg.get("burn_in", 20))
    post = e_tr[bi:, :]
    mae = np.mean(np.abs(post), axis=0)
    lock_w1000 = (np.abs(post) <= 1000.0).mean(axis=0)
    return mae, lock_w1000


def get_model_config(model_dir):
    """Extract the frozen config name from model_dir and load it."""
    base = os.path.basename(os.path.normpath(model_dir))
    parent = os.path.basename(os.path.dirname(model_dir))
    config_name = parent.replace("s15__", "s15__").replace("rl__", "rl__")
    frozen_path = os.path.join(HERE, "configs", "frozen", config_name + ".json")
    if os.path.exists(frozen_path):
        with open(frozen_path, "r") as f:
            return json.load(f)
    return {}


def build_eval_cfg(model_info):
    """Build evaluation cfg dict for the model's native environment on the target condition."""
    frozen_cfg = get_model_config(model_info["model_dir"])
    env_cfg = frozen_cfg.get("env", {})
    cfg = dict(DEFAULT_CFG)
    cfg.update(env_cfg)
    cond = model_info["cond"]
    if cond == "RB":
        cfg.update({"drift": "gaussian", "drift_sigma": 500.0, "probe": "rabi"})
    elif cond == "nominal":
        cfg.update({"drift": "gaussian", "drift_sigma": 500.0, "probe": "dither"})
    return cfg


def run_model(model_info, out_dir):
    """Run validation episodes for one RL model."""
    print(f"\n=== Running {model_info['name']} (seed {model_info['model_seed']}) from {model_info['model_dir']} ===")
    cfg = build_eval_cfg(model_info)
    model, norm = load_rl_frozen(model_info["model_dir"])
    e_tr, fc_tr, x_tr, a_tr, env = run_episodes_rl(model, norm, cfg, VAL_SEED_BASE, N_EPISODES)
    mae, lock_w1000 = compute_metrics(e_tr, env)

    results = []
    for ep in range(N_EPISODES):
        results.append({
            "model": model_info["name"],
            "env": model_info["env"],
            "model_seed": model_info["model_seed"],
            "episode": ep,
            "mae_hz": float(mae[ep]),
            "lock_w1000": float(lock_w1000[ep]),
        })
        print(f"  Ep {ep}: MAE={mae[ep]:.2f} Hz, lock_w1000={lock_w1000[ep]:.4f}")

    # Save trajectories for plotting (include model_seed in filename)
    seed_suffix = f"_s{model_info['model_seed']}"
    np.savez_compressed(os.path.join(out_dir, f"{model_info['name']}{seed_suffix}_trajectories.npz"),
                        e=e_tr, fc=fc_tr, x=x_tr, actions=a_tr)

    return results, e_tr, fc_tr, x_tr, a_tr, env


def run_hillclimb_baseline(out_dir):
    """Run HillClimb baseline on RB nominal env with same VAL seeds (PID doesn't support rabi probe)."""
    print(f"\n=== Running HillClimb baseline on RB (PID unsupported for rabi probe) ===")
    cfg = dict(DEFAULT_CFG)
    cfg.update({"drift": "gaussian", "drift_sigma": 500.0, "probe": "rabi"})
    e_tr, fc_tr, x_tr, a_tr, env = run_episodes_hillclimb(cfg, VAL_SEED_BASE, N_EPISODES)
    mae, lock_w1000 = compute_metrics(e_tr, env)

    results = []
    for ep in range(N_EPISODES):
        results.append({
            "model": "HillClimb_baseline_RB",
            "env": "surrogate_rabi",
            "model_seed": "baseline",
            "episode": ep,
            "mae_hz": float(mae[ep]),
            "lock_w1000": float(lock_w1000[ep]),
        })
        print(f"  Ep {ep}: MAE={mae[ep]:.2f} Hz, lock_w1000={lock_w1000[ep]:.4f}")

    np.savez_compressed(os.path.join(out_dir, "HillClimb_baseline_RB_trajectories.npz"),
                        e=e_tr, fc=fc_tr, x=x_tr, actions=a_tr)

    return results, e_tr, fc_tr, x_tr, a_tr, env


def plot_trajectories(model_name, model_seed, e_tr, fc_tr, x_tr, a_tr, env, out_dir, rep_ep=0):
    """Plot agent offset/frequency vs reference over time for 1 rep episode + overlay all 5."""
    T = e_tr.shape[0]
    t_axis = np.arange(T)
    seed_suffix = f"_s{model_seed}" if model_seed != "baseline" else ""

    # Figure 1: Representative episode
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ep = rep_ep
    axes[0].plot(t_axis, e_tr[:, ep], label='Error (x - fc)', color='tab:red')
    axes[0].axhline(1000, color='k', ls='--', alpha=0.3)
    axes[0].axhline(-1000, color='k', ls='--', alpha=0.3)
    axes[0].set_ylabel('Offset (Hz)')
    axes[0].set_title(f'{model_name}{seed_suffix} - Episode {ep} (VAL seed {VAL_SEED_BASE+ep})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t_axis, fc_tr[:, ep], label='Control freq (fc)', color='tab:blue')
    axes[1].axhline(0, color='k', ls='--', alpha=0.3)
    axes[1].set_ylabel('Control Offset (Hz)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(t_axis, a_tr[:, ep], label='Action', color='tab:green')
    axes[2].axhline(0, color='k', ls='--', alpha=0.3)
    axes[2].set_ylabel('Action (norm)')
    axes[2].set_xlabel('Step')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}{seed_suffix}_ep{ep}_trajectory.png"), dpi=150)
    plt.close()

    # Figure 2: Overlay all 5 episodes (error only)
    fig, ax = plt.subplots(figsize=(10, 5))
    for ep in range(N_EPISODES):
        ax.plot(t_axis, e_tr[:, ep], alpha=0.7, label=f'Ep {ep}')
    ax.axhline(1000, color='k', ls='--', alpha=0.3)
    ax.axhline(-1000, color='k', ls='--', alpha=0.3)
    ax.set_ylabel('Error (Hz)')
    ax.set_xlabel('Step')
    ax.set_title(f'{model_name}{seed_suffix} - All 5 Episodes Overlay')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{model_name}{seed_suffix}_overlay.png"), dpi=150)
    plt.close()


def main():
    out_dir = os.path.join(WORK, "results", "traj_v3")
    os.makedirs(out_dir, exist_ok=True)

    all_results = []

    # Run RL models
    for model_info in MODELS:
        results, e_tr, fc_tr, x_tr, a_tr, env = run_model(model_info, out_dir)
        all_results.extend(results)
        seed_suffix = f"_s{model_info['model_seed']}"
        plot_trajectories(model_info["name"], model_info["model_seed"], e_tr, fc_tr, x_tr, a_tr, env, out_dir, rep_ep=0)

    # Run HillClimb baseline
    hc_results, e_tr, fc_tr, x_tr, a_tr, env = run_hillclimb_baseline(out_dir)
    all_results.extend(hc_results)
    plot_trajectories("HillClimb_baseline_RB", "baseline", e_tr, fc_tr, x_tr, a_tr, env, out_dir, rep_ep=0)

    # Add per-(model, seed) mean rows
    models_to_average = ["E1_zero_shot_RB", "E4_RB", "FTB_200000_RB", "Z1_dither_plain", "HillClimb_baseline_RB"]
    for model in models_to_average:
        model_rows = [r for r in all_results if r["model"] == model]
        if not model_rows:
            continue
        # Group by model_seed
        for seed in set(r["model_seed"] for r in model_rows):
            seed_rows = [r for r in model_rows if r["model_seed"] == seed]
            env = seed_rows[0]["env"]
            mean_mae = np.mean([float(r["mae_hz"]) for r in seed_rows])
            mean_lock = np.mean([float(r["lock_w1000"]) for r in seed_rows])
            mean_row = {
                "model": model,
                "env": env,
                "model_seed": seed,
                "episode": "mean",
                "mae_hz": round(mean_mae, 6),
                "lock_w1000": round(mean_lock, 6),
            }
            all_results.append(mean_row)

    # Save summary CSV
    csv_path = os.path.join(out_dir, "traj_summary_v3.csv")
    import csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "env", "model_seed", "episode", "mae_hz", "lock_w1000"])
        writer.writeheader()
        for row in all_results:
            writer.writerow(row)
    print(f"\nSaved summary CSV: {csv_path}")

    # Print summary table
    print("\n=== TRAJ_SUMMARY_V3 ===")
    for row in all_results:
        ep_str = str(row['episode'])
        seed_str = str(row['model_seed'])
        print(f"{row['model']:25s} {row['env']:15s} seed{seed_str:6s} ep{ep_str:4s}  MAE={row['mae_hz']:7.2f}  lock_w1000={row['lock_w1000']:.4f}")

    # Verify seed-202 rows equal the second block of the old CSV
    print("\n=== VERIFICATION: seed-202 rows vs old CSV second block ===")
    old_csv = os.path.join(WORK, "results", "traj", "traj_summary.csv")
    if os.path.exists(old_csv):
        import csv as csvmod
        with open(old_csv, "r") as f:
            reader = csvmod.DictReader(f)
            old_rows = list(reader)
        # Second block for each model (rows 6-10, 17-21, 27-31, 37-41)
        for model in ["E1_zero_shot_RB", "E4_RB", "FTB_200000_RB", "Z1_dither_plain"]:
            new_rows = [r for r in all_results if r["model"] == model and r["model_seed"] in (202, 102) and r["episode"] != "mean"]
            if model == "Z1_dither_plain":
                old_block = old_rows[31:36]  # rows 37-41 (0-indexed 31-35)
            elif model == "E1_zero_shot_RB":
                old_block = old_rows[6:11]   # rows 7-11
            elif model == "E4_RB":
                old_block = old_rows[16:21]  # rows 17-21
            elif model == "FTB_200000_RB":
                old_block = old_rows[26:31]  # rows 27-31
            for i, (new_r, old_r) in enumerate(zip(new_rows, old_block)):
                mae_match = abs(new_r["mae_hz"] - float(old_r["mae_hz"])) < 0.01
                lock_match = abs(new_r["lock_w1000"] - float(old_r["lock_w1000"])) < 0.0001
                print(f"  {model} seed{new_r['model_seed']} ep{i}: MAE {'MATCH' if mae_match else 'DIFFER'}, lock {'MATCH' if lock_match else 'DIFFER'}")

    # Check files created
    print("\n=== FILES CREATED ===")
    for f in sorted(os.listdir(out_dir)):
        fp = os.path.join(out_dir, f)
        stat = os.stat(fp)
        print(f"  {f}  {stat.st_size} bytes  mtime={datetime.datetime.fromtimestamp(stat.st_mtime).isoformat()}")

    print("\nTRAJ1c DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())