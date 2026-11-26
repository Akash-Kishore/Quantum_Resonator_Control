"""qstudy lockstates — Part-2 S18.1 lock-state machine + S18.2 failure taxonomy.

Lock-state machine (per episode, applied to e over all steps 0..T-1; W = 1000 Hz or 0.05
cycles in platform units): state LOCKED if |e|<=W for at least 3 consecutive steps; state
LOST if |e|>10000 Hz for at least 3 consecutive steps; otherwise TRANSIT.
Per-episode outputs: first-lock time, LOCKED fraction, number of lock-loss events (entries
into LOST after having been LOCKED), re-lock time (steps from entering LOST to next LOCKED),
sojourn-time lists, 3x3 transition counts (states sampled per step, transitions counted).

Failure taxonomy (rule-based; first matching rule wins; onset t0 = first step of a >=20-step
run with |e|>10000 Hz after burn-in; inspect the 10 steps before t0):
F1 INFORMATION DEFICIT, F2 SUDDEN JUMP, F3 DELAY/TRACKING-RATE, F4 ESTIMATE ALIASING,
F5 CONTROL INSTABILITY, F6 QUANTISATION, F7 MODEL MISMATCH, F8 UNEXPLAINED.
"""
import numpy as np

LOST_T = 10000.0


def lock_state_series(e, W=1000.0):
    """Per-step state series: 0=TRANSIT, 1=LOCKED, 2=LOST.
    Rule (S18.1, applied literally): step t is LOCKED if the run of consecutive steps with
    |e|<=W ending at t has length >= 3; step t is LOST if the run of consecutive steps with
    |e|>10000 Hz ending at t has length >= 3 (LOST wins if both could fire)."""
    e = np.asarray(e, dtype=float)
    T = e.shape[0]
    inside = np.abs(e) <= W
    far = np.abs(e) > LOST_T
    st = np.zeros(T, dtype=int)
    c_in = c_far = 0
    for t in range(T):
        c_in = c_in + 1 if inside[t] else 0
        c_far = c_far + 1 if far[t] else 0
        if c_in >= 3:
            st[t] = 1
        if c_far >= 3:
            st[t] = 2
    return st


def episode_lock_states(e, W=1000.0):
    """Per-episode outputs (S18.1). e: (T,) trace. Returns a dict."""
    e = np.asarray(e, dtype=float)
    st = lock_state_series(e, W)
    T = len(e)
    first_lock = -1
    for t in range(T):
        if st[t] == 1:
            first_lock = t
            break
    locked_frac = float((st == 1).mean())
    # lock-loss events: entries into LOST after having been LOCKED
    was_locked = False
    losses = 0
    relock_times = []
    in_lost = False
    lost_start = -1
    for t in range(T):
        if st[t] == 2 and not in_lost:
            in_lost = True
            lost_start = t
            if was_locked:
                losses += 1
        elif st[t] != 2 and in_lost:
            in_lost = False
            if was_locked:
                # re-lock time: steps from entering LOST to next LOCKED
                if st[t] == 1:
                    relock_times.append(t - lost_start)
                else:
                    for tt in range(t, T):
                        if st[tt] == 1:
                            relock_times.append(tt - lost_start)
                            break
        if st[t] == 1:
            was_locked = True
    # sojourn times: consecutive-run lengths per state
    sojourns = {0: [], 1: [], 2: []}
    t = 0
    while t < T:
        t2 = t
        while t2 + 1 < T and st[t2 + 1] == st[t]:
            t2 += 1
        sojourns[int(st[t])].append(t2 - t + 1)
        t = t2 + 1
    # 3x3 transition counts (step t -> t+1)
    trans = np.zeros((3, 3), dtype=int)
    for t in range(T - 1):
        trans[int(st[t]), int(st[t + 1])] += 1
    return dict(first_lock=first_lock, locked_frac=locked_frac, losses=losses,
                relock_times=relock_times, sojourns=sojourns, transitions=trans, states=st)


def failure_label(e, W=1000.0, burn_in=20, fisher_floor=None, unambiguous=None,
                  prev_action=None, dac_lsb=None, assumed_sigma=None, true_sigma=None,
                  jump_sizes=None, delay=None, x_trace=None):
    """S18.2 rule-based failure taxonomy. Returns (label, evidence dict).
    fisher_floor: callable(delta)->static std from Fisher info at the true detuning (None -> F1 first rule skipped).
    unambiguous: callable(delta)->bool True inside the probe's unambiguous region (None -> F1 second rule skipped).
    prev_action: (T,) actions (None -> F5 skipped). dac_lsb: float (None -> F6 skipped).
    assumed_sigma/true_sigma: floats (None -> F7 skipped). jump_sizes: (T,) |jump| per step.
    x_trace: (T,) TRUE-frequency trace; required for F3 (|x(t0) - x(t0-delay-1)| > W)."""
    e = np.asarray(e, dtype=float)
    T = len(e)
    ev = {}
    # onset t0: first step of a >=20-step run with |e|>10000 Hz AFTER burn-in
    t0 = None
    run = 0
    for t in range(burn_in, T):
        run = run + 1 if abs(e[t]) > LOST_T else 0
        if run >= 20:
            t0 = t - 19
            break
    ev["onset"] = t0
    if t0 is None:
        return "NO_FAILURE", ev
    lo = max(0, t0 - 10)
    ev["window"] = (lo, t0)
    # F1 INFORMATION DEFICIT
    if fisher_floor is not None and callable(fisher_floor):
        sd = fisher_floor(float(e[t0]))
        ev["f1_static_std"] = sd
        if sd > 5000.0:
            return "IF", ev
    if unambiguous is not None and callable(unambiguous):
        out = not unambiguous(float(e[t0]))
        ev["f1_outside_unambiguous"] = out
        if out:
            return "IF", ev
    # F2 SUDDEN JUMP: a drift jump/flip with size > 3000 Hz within 5 steps before t0
    if jump_sizes is not None:
        js = np.asarray(jump_sizes, dtype=float)
        pre = js[max(0, t0 - 5):t0]
        big = float(pre.max()) if pre.size else 0.0
        ev["f2_max_jump_before"] = big
        if big > 3000.0:
            return "JUMP", ev
    # F3 DELAY/TRACKING-RATE: |x(t0) - x(t0 - delay - 1)| > W while no jump (needs x_trace)
    if delay is not None and x_trace is not None:
        xt = np.asarray(x_trace, dtype=float)
        i2 = t0 - delay - 1
        if 0 <= i2 < len(xt):
            d = abs(float(xt[t0]) - float(xt[i2]))
            ev["f3_delay_delta"] = d
            if d > W:
                return "DELAY", ev
    # F4 ESTIMATE ALIASING: probe estimate error > half the unambiguous range with adequate info
    # (caller supplies alias_error > half_range via unambiguous callback context; skipped here
    # when no alias probe model is available)
    # F5 CONTROL INSTABILITY: action saturation |a|>0.95 in >8 of the 10 steps before t0,
    # or >=5 sign changes in those 10 steps
    if prev_action is not None:
        a = np.asarray(prev_action, dtype=float)[lo:t0]
        if a.size:
            sat = int((np.abs(a) > 0.95).sum())
            signs = np.sign(a[np.abs(a) > 0.05])
            flips = int((np.diff(signs) != 0).sum()) if signs.size > 1 else 0
            ev["f5_sat_steps"] = sat
            ev["f5_sign_changes"] = flips
            if sat > 8 or flips >= 5:
                return "CONTROL", ev
    # F6 QUANTISATION: DAC LSB > W/2
    if dac_lsb is not None:
        ev["f6_dac_lsb"] = float(dac_lsb)
        if dac_lsb > W / 2.0:
            return "DAC", ev
    # F7 MODEL MISMATCH: (filter-based) assumed drift sigma differs from truth by > factor 2
    if assumed_sigma is not None and true_sigma is not None:
        ratio = float(max(assumed_sigma / max(true_sigma, 1e-9), true_sigma / max(assumed_sigma, 1e-9)))
        ev["f7_sigma_ratio"] = ratio
        if ratio > 2.0:
            return "MISMATCH", ev
    # F8
    return "UNEXPLAINED", ev


def aggregate_lock_states(e_tr, W=1000.0):
    """Aggregate per (method, env variant, condition) column: e_tr (T, B) -> occupancy
    fractions, mean/median sojourn times, transition matrix (summed), first-lock-time list,
    re-lock-time list, losses per episode."""
    T, B = e_tr.shape
    occ = np.zeros(3)
    trans = np.zeros((3, 3), dtype=int)
    sojourns = {0: [], 1: [], 2: []}
    first_locks, relocks, losses = [], [], []
    locked_fracs = []
    for b in range(B):
        r = episode_lock_states(e_tr[:, b], W)
        occ += np.bincount(r["states"], minlength=3) / T
        trans += r["transitions"]
        for k in sojourns:
            sojourns[k].extend(r["sojourns"][k])
        first_locks.append(r["first_lock"])
        relocks.extend(r["relock_times"])
        losses.append(r["losses"])
        locked_fracs.append(r["locked_frac"])
    out = dict(occupancy=(occ / B).tolist(),
               transition_matrix=trans.tolist(),
               first_lock_times=first_locks,
               relock_times=relocks,
               losses_per_episode=losses,
               locked_frac_mean=float(np.mean(locked_fracs)))
    for k, name in ((0, "transit"), (1, "locked"), (2, "lost")):
        s = sojourns[k]
        out["sojourn_mean_%s" % name] = float(np.mean(s)) if s else None
        out["sojourn_median_%s" % name] = float(np.median(s)) if s else None
        out["sojourn_%s" % name] = s
    return out
