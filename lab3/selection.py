import numpy as np

TIE_ORDER = ["frozen", "unfreeze", "scratch"]


def regime_means(f1):
    out = {}
    for regime in ("scratch", "frozen", "unfreeze"):
        vals = [v for (r, _), v in f1.items() if r == regime]
        if vals:
            out[regime] = float(np.mean(vals))
    return out


def choose(f1):
    means = regime_means(f1)
    best_regime = None
    for r in TIE_ORDER:
        if r in means and (best_regime is None or means[r] > means[best_regime]):
            best_regime = r
    seeds = sorted(s for (r, s) in f1 if r == best_regime)
    best_seed = max(seeds, key=lambda s: (f1[(best_regime, s)], -s))
    return best_regime, best_seed, means


def effects(f1):
    scratch = f1[("scratch", 1)]
    fz = [f1[("frozen", s)] for s in (1, 2, 3)]
    uf = [f1[("unfreeze", s)] for s in (1, 2, 3)]
    fbar_fz, fbar_uf = float(np.mean(fz)), float(np.mean(uf))
    r_fz, r_uf = max(fz) - min(fz), max(uf) - min(uf)
    R = max(r_fz, r_uf)
    d_pre = fbar_fz - scratch
    d_unf = fbar_uf - fbar_fz
    return {
        "F_scratch": scratch, "Fbar_frozen": fbar_fz, "Fbar_unfreeze": fbar_uf,
        "R_frozen": r_fz, "R_unfreeze": r_uf, "R": R,
        "delta_pre": d_pre, "delta_unf": d_unf, "delta_total": d_pre + d_unf,
        "verdict_pre": verdict(d_pre, R), "verdict_unf": verdict(d_unf, R),
    }


def verdict(delta, R):
    if abs(delta) <= R:
        return "не відрізняється від розкиду: результату немає"
    return "ефект вартий перевірки, але не доведений"
