"""D8 (P2): frozen configs vs exploratory configs - only the four frozen keys may differ."""
import json, os

S = r"C:\Projects\MiniProject_Sem4\qstudy"
NINE = ["rl__rabi__plain", "rl__rabi__gru8", "rl__dither__plain", "rl__dither__filter",
        "rl__dither__DR", "rl__dither__DRcurr", "rl__dither__full",
        "rl__ramsey_iq__plain", "rl__ramsey_iq__gru8"]
ALLOWED = {("ppo", "learning_rate"): 0.0003, ("ppo", "ent_coef"): 0.0,
           ("ppo", "gamma"): 0.99, ("total_steps",): 2000000}


def walk(a, b, prefix=()):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            out += walk(a.get(k), b.get(k), prefix + (k,))
    elif a != b:
        out.append((prefix, a, b))
    return out


bad = 0
for name in NINE:
    ex = json.load(open(os.path.join(S, "configs", name + ".json"), encoding="utf-8"))
    fz = json.load(open(os.path.join(S, "configs", "frozen", name + ".json"), encoding="utf-8"))
    diffs = walk(ex, fz)
    illegal = [d for d in diffs if d[0] not in ALLOWED or ALLOWED[d[0]] != d[2]]
    blocks = [json.dumps(ex.get(bl), indent=1) == json.dumps(fz.get(bl), indent=1)
              for bl in ("env", "memory", "algo")]
    if illegal:
        bad += 1
        print("%-22s FAIL illegal diffs: %s" % (name, illegal[:3]))
    else:
        print("%-22s OK diffs=%s env/mem/algo identical=%s" % (
            name, [list(d[0]) for d in diffs], all(blocks)))
print("SUMMARY:", "ALL NINE PASS" if bad == 0 else "%d CONFIGS FAIL" % bad)
