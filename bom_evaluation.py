"""Auswertung: ein Lauf mit gewähltem Modell gegen den Standard und den Zufall, Kennzahlen, vorgerechnete Studien über alle Stellschrauben."""

import json
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import bom_algorithm as A
import bom_constants as C
import bom_scenario as S

_HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    noise: float = C.DEFAULT_NOISE
    seed: int = C.DEFAULT_SEED


@dataclass(frozen=True)
class Choice:
    """Die gewählten Stellschrauben des Ersatzmodells (Beschriftungen wie in C.*_CHOICES)."""
    kernel: str = "matern52"
    lengthscale: str = "gelernt"
    noise: str = "gelernt"
    transform: str = "none"
    acquisition: str = "ei"
    beta: float = 4.0

    def model(self):
        return A.Model(kernel=self.kernel, lengthscale=C.LS_VALUE[self.lengthscale], noise=C.NOISE_VALUE[self.noise], transform=self.transform,
                       acquisition=self.acquisition, beta=self.beta)


DEFAULT_MODEL = A.Model()


def variant_model(key):
    """Das Modell einer Studienvariante (eine Stellschraube gegen den Standard verändert)."""
    if key == "default":
        return A.Model()
    if key.startswith("ls_"):
        return A.Model(lengthscale=float(key[3:]))
    if key.startswith("kernel_"):
        return A.Model(kernel=key[7:])
    if key == "noise_tiny":
        return A.Model(noise=1e-6)
    if key == "noise_big":
        return A.Model(noise=0.3)
    if key == "log":
        return A.Model(transform="log")
    if key == "acq_pi":
        return A.Model(acquisition="pi")
    if key.startswith("acq_ucb_"):
        return A.Model(acquisition="ucb", beta=float(key[8:]))
    if key == "acq_ts":
        return A.Model(acquisition="ts")
    raise KeyError(key)


@dataclass
class Run:
    label: str
    idx: list
    y: list
    truth_trace: list    # wahrer Wert der Empfehlung (beste Beobachtung) nach t Auswertungen


@dataclass
class Analysis:
    kind: str
    settings: Settings
    runs: dict           # "random", "default", "choice" -> Run
    truth: np.ndarray    # nur 2D-Landschaften
    optimum: tuple       # (Zeile, Spalte, Wert)
    x_values: list
    y_values: list


def landscape(kind, noise=C.DEFAULT_NOISE, trucks=C.DEFAULT_TRUCKS, seed=0):
    """(Kandidaten im Einheitswürfel, Wahrheit je Kandidat, observe(index), Optimum)."""
    if kind == "hartmann":
        import bom_benchmarks as B
        pool = S.hartmann_pool()
        truth = np.array([B.hartmann6(p) for p in pool])
        return pool, truth, (lambda i: float(truth[i])), float(truth.min())
    if kind == "branin":
        field = S.BraninField(noise, seed)
    else:
        field = S.TerminalField(trucks, seed)
    grid = field.truth_grid()
    pts = S.grid_points(grid.shape)
    cand = S.to_unit(pts, grid.shape)
    flat = grid.reshape(-1)
    return cand, flat, (lambda i: field.observe(int(pts[i][0]), int(pts[i][1]))), float(flat.min())


def run_one(kind, model, settings, n, seed_offset=0, label="", n_start=None):
    """Ein Suchlauf mit `model` (None = Zufall). Reproduzierbar: Zufallsströme aus Seed und Offset."""
    trucks = C.DEFAULT_TRUCKS
    seed = settings.seed + seed_offset
    cand, truth, ev, _ = landscape(kind, settings.noise, trucks, seed)
    rng = np.random.default_rng(seed + 7)
    n_start = n_start or (C.HARTMANN_N_START if kind == "hartmann" else C.N_START)
    if model is None:
        idx, y = A.random_search(len(cand), ev, n, rng)
    else:
        idx, y = A.bo_search(cand, ev, n, rng, model, n_start=n_start)
    trace = [float(truth[idx[int(np.argmin(y[:t]))]]) for t in range(1, n + 1)]
    return Run(label, [int(i) for i in idx], [float(v) for v in y], trace)


def analyse(settings, choice, kind):
    """Zufall, Standardmodell und gewähltes Modell mit gleichem Startdesign (gleicher Seed) und gleicher Rauschreihe."""
    runs = {
        "random": run_one(kind, None, settings, settings.n, 11, "Zufall"),
        "default": run_one(kind, DEFAULT_MODEL, settings, settings.n, 0, "Standard"),
        "choice": run_one(kind, choice.model(), settings, settings.n, 0, "Gewähltes Modell"),
    }
    if kind == "hartmann":
        cand, truth, _, opt = landscape(kind)
        return Analysis(kind, settings, runs, np.zeros((1, 1)), (0, 0, opt), [], [])
    if kind == "branin":
        x1, x2 = S.BraninField.axes()
        truth = S.BraninField.truth_grid()
        xs, ys = list(map(float, x1)), list(map(float, x2))
    else:
        truth = S.TerminalField.truth_grid()
        xs, ys = list(C.SPEEDS), list(C.PLACES)
    i, j = np.unravel_index(int(np.argmin(truth)), truth.shape)
    return Analysis(kind, settings, runs, truth, (int(i), int(j), float(truth[i, j])), xs, ys)


def final_regret(a):
    return {k: r.truth_trace[-1] - a.optimum[2] for k, r in a.runs.items()}


# --- Vorgerechnete Studien -----------------------------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def study():
    with open(os.path.join(_HERE, "precomputed_study.json"), encoding="utf-8") as fh:
        return json.load(fh)


def study_key(kind, noise=None):
    return f"branin_{float(noise)}" if kind == "branin" else kind


def optimum_of(kind):
    st = study()["optimum"]
    return float(st["branin" if kind == "branin" else kind])


def regret_table(kind, noise=None):
    """Median und 90-%-Quantil des Regrets je Variante an den Prüfpunkten des Budgets (20 feste Läufe)."""
    raw = study()["runs"][study_key(kind, noise)]
    opt = optimum_of(kind)
    cps = C.STUDY_CHECKPOINTS[kind]
    out = {}
    for key, traces in raw.items():
        arr = np.array(traces)                       # Läufe x Budget
        out[key] = {cp: {"median": float(np.median(arr[:, cp - 1]) - opt), "q90": float(np.quantile(arr[:, cp - 1], 0.9) - opt)} for cp in cps}
    return out


def factor_spread(kind, checkpoint, noise=None):
    """Je Stellschraube: kleinster und größter Median-Regret über ihre Varianten (der Standard zählt mit) bei `checkpoint` Auswertungen."""
    tab = regret_table(kind, noise)
    out = {}
    for f in C.FACTOR_LABELS:
        keys = ["default"] + [k for k in tab if C.VARIANT_FACTOR.get(k) == f]
        vals = [tab[k][checkpoint]["median"] for k in keys]
        if len(vals) > 1:
            out[f] = {"min": min(vals), "max": max(vals), "best": keys[int(np.argmin(vals))], "worst": keys[int(np.argmax(vals))]}
    return out


def paired_stats(kind, key, checkpoint, noise=None, tol=1e-6):
    """Gepaarter Vergleich Variante gegen Standard über die 20 festen Läufe (gleiche Seeds, gleiches Startdesign, gleiche Rauschreihe):
    wie oft besser, schlechter, gleich (Unterschied unter `tol`), der Median der Differenz (negativ = Variante besser) und der zweiseitige Vorzeichentest-p-Wert
    über die nicht gleichen Läufe."""
    from scipy.stats import binomtest
    raw = study()["runs"][study_key(kind, noise)]
    d = np.array(raw["default"])[:, checkpoint - 1]
    v = np.array(raw[key])[:, checkpoint - 1]
    diff = v - d
    better = int((diff < -tol).sum())
    worse = int((diff > tol).sum())
    p = float(binomtest(better, better + worse, 0.5).pvalue) if better + worse else 1.0
    return {"better": better, "worse": worse, "equal": int(len(diff) - better - worse), "median_diff": float(np.median(diff)), "p": p, "n": int(len(diff))}


def verdict_of(stats, alpha=0.05):
    """'besser', 'schlechter' oder 'kein Unterschied nachweisbar' (Vorzeichentest, ohne Korrektur für die vielen Vergleiche)."""
    if stats["p"] < alpha:
        return "besser" if stats["better"] > stats["worse"] else "schlechter"
    return "kein Unterschied nachweisbar"


def choice_variant_key(choice):
    """Entspricht die Auswahl genau einer Studienvariante (eine Stellschraube gegen den Standard)? Schlüssel, 'default' oder None bei Kombination."""
    d = Choice()
    changed = [f for f in ("kernel", "lengthscale", "noise", "transform", "acquisition") if getattr(choice, f) != getattr(d, f)]
    if choice.acquisition == "ucb" and choice.beta != d.beta and "acquisition" not in changed:
        changed.append("acquisition")
    if not changed:
        return "default"
    if len(changed) > 1:
        return None
    f = changed[0]
    if f == "kernel":
        return f"kernel_{choice.kernel}"
    if f == "lengthscale":
        return {"0,1 fest": "ls_0.1", "0,3 fest": "ls_0.3", "1,0 fest": "ls_1.0"}.get(choice.lengthscale)
    if f == "noise":
        return {"winzig fest": "noise_tiny", "groß fest": "noise_big"}.get(choice.noise)
    if f == "transform":
        return "log"
    if choice.acquisition == "pi":
        return "acq_pi"
    if choice.acquisition == "ts":
        return "acq_ts"
    if choice.acquisition == "ucb":
        return {0.25: "acq_ucb_0.25", 4.0: "acq_ucb_4", 64.0: "acq_ucb_64"}.get(choice.beta)
    return None
