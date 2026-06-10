"""
spade_stats.py — pure statistical core for SPADE (no Streamlit dependency).

These functions are the testable, reusable versions of the algorithms used in
wheat_dashboard.py. They take plain arrays / dicts instead of reading
st.session_state, so they can be imported by the validation harness
(validate_spade.py) and, ideally, by the dashboard itself to remove duplication.

References
----------
Dixon's Q critical values : Rorabacher, D.B. (1991). Anal. Chem. 63(2), 139–146.
Grubbs' test               : Grubbs, F.E. (1969). Technometrics 11(1), 1–21.
Compact letter display     : Piepho, H.P. (2004). J. Comput. Graph. Stat. 13(2), 456–466.
Partial omega-squared      : Olejnik, S. & Algina, J. (2003). Psychol. Methods 8(4), 434–447.
"""
from itertools import combinations
import numpy as np
from scipy import stats


# ── ONE-WAY ANOVA ─────────────────────────────────────────────────────────────
def one_way_anova(groups):
    """groups: list of 1-D arrays (one per treatment). Balanced or unbalanced."""
    valid = [np.asarray(g, float) for g in groups if len(g) >= 2]
    if len(valid) < 2:
        return None
    all_v = np.concatenate(valid)
    gm = all_v.mean()
    N, k = len(all_v), len(valid)
    ss_t = sum(len(g) * (g.mean() - gm) ** 2 for g in valid)
    ss_e = sum(((g - g.mean()) ** 2).sum() for g in valid)
    df_t, df_e = k - 1, N - k
    ms_t = ss_t / df_t
    ms_e = ss_e / df_e if df_e > 0 else np.nan
    f = ms_t / ms_e if ms_e and ms_e > 0 else np.nan
    p = 1 - stats.f.cdf(f, df_t, df_e) if not np.isnan(f) else np.nan
    sizes = [len(g) for g in valid]
    n_harm = len(sizes) / sum(1.0 / s for s in sizes)
    return dict(ss_t=ss_t, ss_e=ss_e, ss_total=ss_t + ss_e,
                df_t=df_t, df_e=df_e, ms_t=ms_t, ms_e=ms_e,
                f=f, p=p, N=N, k=k, n_harm=n_harm)


def effect_sizes(an):
    if not an or an["ss_total"] == 0:
        return None
    eta2 = an["ss_t"] / an["ss_total"]
    omega2 = max(0.0, (an["ss_t"] - an["df_t"] * an["ms_e"]) /
                 (an["ss_total"] + an["ms_e"])) if an["ms_e"] > 0 else np.nan
    return {"eta2": eta2, "omega2": omega2}


def lsd_05(an, r=None):
    if not an or np.isnan(an["ms_e"]):
        return np.nan
    if r is None:
        r = an.get("n_harm", 3)
    return stats.t.ppf(0.975, an["df_e"]) * np.sqrt(2 * an["ms_e"] / r)


# ── OUTLIER TESTS ─────────────────────────────────────────────────────────────
DIXON_Q_CRIT = {3: 0.970, 4: 0.829, 5: 0.710, 6: 0.625,
                7: 0.568, 8: 0.526, 9: 0.493, 10: 0.466}


def dixon_q_test(vals):
    vals = [v for v in vals if not np.isnan(v)]
    n = len(vals)
    if n < 3:
        return []
    s = sorted(vals)
    rng = s[-1] - s[0]
    if rng == 0:
        return []
    q_crit = DIXON_Q_CRIT.get(n, 0.466)
    out = []
    q_low = (s[1] - s[0]) / rng
    q_high = (s[-1] - s[-2]) / rng
    if q_low > q_crit:
        out.append({"value": s[0], "Q": q_low, "Q_crit": q_crit, "end": "low"})
    if q_high > q_crit:
        out.append({"value": s[-1], "Q": q_high, "Q_crit": q_crit, "end": "high"})
    return out


def grubbs_test(vals, alpha=0.05):
    vals = [v for v in vals if not np.isnan(v)]
    n = len(vals)
    if n < 3:
        return None
    arr = np.asarray(vals, float)
    sd = arr.std(ddof=1)
    if sd == 0:
        return None
    G = np.max(np.abs(arr - arr.mean())) / sd
    t_crit = stats.t.ppf(1 - alpha / (2 * n), df=n - 2)
    G_crit = ((n - 1) / np.sqrt(n)) * np.sqrt(t_crit ** 2 / (n - 2 + t_crit ** 2))
    return {"G": G, "G_crit": G_crit, "outlier": bool(G > G_crit)}


# ── COMPACT LETTER DISPLAY ────────────────────────────────────────────────────
def compact_letter_display(ns_pairs, groups, means):
    """
    ns_pairs : set of frozenset({a,b}) for treatment pairs that are NOT
               significantly different.
    groups   : ordered list of treatment labels.
    means    : dict label -> mean (controls letter ordering).
    Returns dict label -> letter string. Maximal-clique enumeration (Piepho 2004).
    """
    gu = list(groups)
    ns = set()
    for a, b in ns_pairs:
        ns.add((a, b)); ns.add((b, a))

    def is_ns(a, b):
        return a == b or (a, b) in ns

    def is_clique(nodes):
        return all(is_ns(nodes[i], nodes[j])
                   for i in range(len(nodes)) for j in range(i + 1, len(nodes)))

    maximal = []
    for size in range(len(gu), 0, -1):
        for combo in combinations(gu, size):
            cs = set(combo)
            if is_clique(list(combo)) and not any(cs < set(mc) for mc in maximal):
                maximal.append(list(combo))
    maximal.sort(key=lambda c: max(means.get(g, 0) for g in c), reverse=True)
    res = {g: [] for g in gu}
    for li, clique in enumerate(maximal[:26]):
        for g in clique:
            res[g].append(chr(ord("a") + li))
    return {g: "".join(sorted(res[g])) if res[g] else "?" for g in gu}


# ── PLANNED CONTRAST (pooled error) ───────────────────────────────────────────
def contrast_t(mean_i, mean_j, n_i, n_j, mse, df_e):
    """Two-group a-priori contrast using pooled MS_error. Returns (est, se, t, p)."""
    est = mean_i - mean_j
    se = np.sqrt(mse * (1.0 / n_i + 1.0 / n_j))
    t = est / se
    p = 2 * (1 - stats.t.cdf(abs(t), df_e))
    return est, se, t, p
