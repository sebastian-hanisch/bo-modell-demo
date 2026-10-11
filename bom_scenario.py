"""Landschaften: Branin mit Gauß-Rauschen, Terminal-Scheibe aus der Tabelle echter Simulationen, Hartmann6 auf einer Kandidatenmenge."""

import json
import os

import numpy as np

import bom_benchmarks as B
import bom_constants as C

_HERE = os.path.dirname(os.path.abspath(__file__))
_TABLE = None


def table():
    """Tabelle der Terminal-Scheibe: Wahrheit [Plätze, Tempo] und je Lauflänge 30 Wiederholungen je Einstellung (aus bayes-opt-demo)."""
    global _TABLE
    if _TABLE is None:
        with open(os.path.join(_HERE, "precomputed_slice.json"), encoding="utf-8") as fh:
            raw = json.load(fh)
        _TABLE = {"truth": np.array(raw["truth"], float), "reps": {int(k): np.array(v, float) for k, v in raw["reps"].items()}}
    return _TABLE


def grid_points(shape):
    """Alle Gitterpunkte als Array (Zeilen-Index, Spalten-Index)."""
    rows, cols = shape
    return np.array([(i, j) for i in range(rows) for j in range(cols)], dtype=int)


def to_unit(pts, shape):
    """Gitterindizes auf den Einheitswürfel."""
    rows, cols = shape
    pts = np.asarray(pts, float)
    return np.column_stack([pts[:, 0] / (rows - 1), pts[:, 1] / (cols - 1)])


class BraninField:
    """Branin auf 31 x 31 mit Gauß-Rauschen; Zeilen = x2, Spalten = x1."""

    _GRID = None

    def __init__(self, noise, seed):
        self.noise = float(noise)
        self.rng = np.random.default_rng(seed + 31)

    @classmethod
    def _grid(cls):
        if cls._GRID is None:
            cls._GRID = B.branin_grid(C.BRANIN_STEPS)
        return cls._GRID

    @classmethod
    def axes(cls):
        return cls._grid()[0], cls._grid()[1]

    @classmethod
    def truth_grid(cls):
        return cls._grid()[2]

    def observe(self, i, j):
        return float(self.truth_grid()[i, j] + self.noise * self.rng.standard_normal())


class TerminalField:
    """Terminal-Scheibe: je Einstellung eine der 30 echten Wiederholungen, ohne Zurücklegen (danach beginnt der Vorrat von vorn)."""

    def __init__(self, trucks, seed):
        self.trucks = trucks
        self.rng = np.random.default_rng(seed)
        self.used = {}

    @staticmethod
    def truth_grid():
        return table()["truth"]

    def observe(self, i, j):
        reps = table()["reps"][self.trucks][i, j]
        used = self.used.setdefault((i, j), [])
        free = [r for r in range(len(reps)) if r not in used]
        if not free:
            used.clear()
            free = list(range(len(reps)))
        r = int(self.rng.choice(free))
        used.append(r)
        return float(reps[r])


def hartmann_pool():
    """Kandidatenmenge für Hartmann6: gescrambelte Sobol-Punkte (fester Seed)."""
    from scipy.stats import qmc
    return qmc.Sobol(6, scramble=True, seed=5).random(C.HARTMANN_POOL)
