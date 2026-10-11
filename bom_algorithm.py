"""Kern der Demo: ein Gauß-Prozess mit wählbarem Kern, Längenmaß- und Rauschmodus, vier Akquisitionsfunktionen, Suche auf einer endlichen Kandidatenmenge.

Alles auf dem Einheitswürfel, Minimierung, nur numpy/scipy. Eine Einheit = eine Regel (DEMO-PLAYBOOK §1b): `kernel_matrix` (Kern), `GaussianProcess.fit/predict`
(Lernen, Vorhersage), `expected_improvement`, `probability_of_improvement`, `lower_confidence_bound` (Akquisition), `bo_search` (Auswahlregel), `random_search`.
Zufall nur über übergebene `numpy.random.Generator`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.stats import norm

KERNELS = ("matern52", "matern32", "rbf")
ACQUISITIONS = ("ei", "pi", "ucb", "ts")
TRANSFORMS = ("none", "log")


@dataclass(frozen=True)
class Model:
    """Alle Stellschrauben des Ersatzmodells und der Auswahl in einem Wert (unveränderlich, als Schlüssel für Zwischenspeicher geeignet)."""
    kernel: str = "matern52"
    lengthscale: float | None = None     # None: je Eingabe aus den Daten gelernt; sonst für alle Eingaben fest (Einheitswürfel)
    noise: float | None = None           # None: gelernt; sonst feste Rauschvarianz auf der standardisierten Ausgabe
    transform: str = "none"              # "log": der Gauß-Prozess modelliert log(Wert), Werte müssen positiv sein
    acquisition: str = "ei"
    beta: float = 4.0                    # nur UCB: Gewicht (sqrt(beta) Standardabweichungen)


def kernel_matrix(A, B, lengthscales, variance, kernel="matern52"):
    """Kernmatrix k(a, b). Matérn-5/2: (1 + √5 r + 5r²/3) e^(−√5 r); Matérn-3/2: (1 + √3 r) e^(−√3 r); RBF: e^(−r²/2). r mit je Eingabe eigenem Längenmaß."""
    r2 = np.maximum((((A[:, None, :] - B[None, :, :]) / lengthscales) ** 2).sum(-1), 0.0)
    if kernel == "rbf":
        return variance * np.exp(-0.5 * r2)
    r = np.sqrt(r2)
    if kernel == "matern32":
        s = np.sqrt(3.0) * r
        return variance * (1.0 + s) * np.exp(-s)
    s = np.sqrt(5.0) * r
    return variance * (1.0 + s + s * s / 3.0) * np.exp(-s)


class GaussianProcess:
    """Gauß-Prozess-Regression; Ausgabe standardisiert. Hyperparameter durch Maximieren der Randwahrscheinlichkeit (L-BFGS-B, feste Startpunkte)."""

    LS_BOUNDS = (0.05, 10.0)

    def __init__(self, kernel="matern52", lengthscale=None, noise=None, restarts=3):
        self.kernel = kernel
        self.ls_fixed = lengthscale
        self.noise_fixed = noise
        self.restarts = restarts

    def _unpack(self, theta, d):
        k = 0
        if self.ls_fixed is None:
            ls = np.exp(theta[:d])
            k = d
        else:
            ls = np.full(d, float(self.ls_fixed))
        var = np.exp(theta[k])
        nz = self.noise_fixed if self.noise_fixed is not None else np.exp(theta[k + 1])
        return ls, var, nz

    def _neg_log_marginal(self, theta, X, z):
        ls, var, nz = self._unpack(theta, X.shape[1])
        K = kernel_matrix(X, X, ls, var, self.kernel) + (nz + 1e-8) * np.eye(len(X))
        try:
            c = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            return 1e10
        a = cho_solve(c, z)
        return 0.5 * z @ a + np.log(np.diag(c[0])).sum() + 0.5 * len(z) * np.log(2 * np.pi)

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, float)
        self.X = X
        self.y_mean = float(y.mean())
        self.y_std = max(float(y.std()), 1e-9)
        z = (y - self.y_mean) / self.y_std
        d = X.shape[1]
        bounds = []
        if self.ls_fixed is None:
            bounds += [(np.log(self.LS_BOUNDS[0]), np.log(self.LS_BOUNDS[1]))] * d
        bounds.append((np.log(0.05), np.log(10.0)))
        if self.noise_fixed is None:
            bounds.append((np.log(1e-6), np.log(2.0)))
        starts = np.random.default_rng(0).uniform(0.15, 1.0, (self.restarts, d))
        best = None
        for r in range(self.restarts):
            th0 = []
            if self.ls_fixed is None:
                th0 += list(np.log(starts[r]))
            th0.append(0.0)
            if self.noise_fixed is None:
                th0.append(np.log(1e-2))
            res = minimize(self._neg_log_marginal, np.array(th0), args=(X, z), method="L-BFGS-B", bounds=bounds)
            if best is None or res.fun < best.fun:
                best = res
        self.lengthscales, self.variance, self.noise = self._unpack(best.x, d)
        K = kernel_matrix(X, X, self.lengthscales, self.variance, self.kernel) + (self.noise + 1e-8) * np.eye(len(X))
        self._chol = cho_factor(K, lower=True)
        self._alpha = cho_solve(self._chol, z)
        return self

    def predict(self, Xs, full_cov=False):
        """Mittelwert und Standardabweichung der **Funktion** (ohne Beobachtungsrauschen); mit full_cov: Mittelwert und Kovarianzmatrix."""
        Xs = np.asarray(Xs, float)
        Ks = kernel_matrix(Xs, self.X, self.lengthscales, self.variance, self.kernel)
        mu = Ks @ self._alpha
        v = cho_solve(self._chol, Ks.T)
        if full_cov:
            cov = kernel_matrix(Xs, Xs, self.lengthscales, self.variance, self.kernel) - Ks @ v
            return mu * self.y_std + self.y_mean, cov * self.y_std ** 2
        var = np.maximum(self.variance - (Ks * v.T).sum(1), 1e-12)
        return mu * self.y_std + self.y_mean, np.sqrt(var) * self.y_std

    def sample(self, Xs, rng):
        """Eine gemeinsame Ziehung der Funktion an den Stellen Xs (für Thompson-Sampling)."""
        mu, cov = self.predict(Xs, full_cov=True)
        w, V = np.linalg.eigh(cov + 1e-9 * self.y_std ** 2 * np.eye(len(Xs)))
        return mu + (V * np.sqrt(np.maximum(w, 0.0))) @ rng.standard_normal(len(Xs))


# --- Akquisition ------------------------------------------------------------------------------------------------------------------------------

def expected_improvement(mu, sd, best):
    """Erwartete Verbesserung (Minimierung): E[max(best − f, 0)], f ~ N(mu, sd²)."""
    sd = np.maximum(np.asarray(sd, float), 1e-12)
    z = (best - mu) / sd
    return (best - mu) * norm.cdf(z) + sd * norm.pdf(z)


def probability_of_improvement(mu, sd, best):
    """Wahrscheinlichkeit, besser als `best` zu sein: Φ((best − mu)/sd)."""
    return norm.cdf((best - mu) / np.maximum(np.asarray(sd, float), 1e-12))


def lower_confidence_bound(mu, sd, beta):
    """Zu maximierender Wert −(mu − sqrt(beta)·sd): kleine Mittelwerte und große Unsicherheit sind gut (UCB für Minimierung)."""
    return -(np.asarray(mu, float) - np.sqrt(beta) * np.asarray(sd, float))


# --- Suche ------------------------------------------------------------------------------------------------------------------------------------

def random_search(n_cand, evaluate, budget, rng):
    """Zufällig ohne Zurücklegen. Gibt (Indizes, Werte) zurück."""
    idx = [int(i) for i in rng.choice(n_cand, size=min(budget, n_cand), replace=False)]
    return idx, [evaluate(i) for i in idx]


def _transform(y, model):
    return np.log(np.asarray(y, float)) if model.transform == "log" else np.asarray(y, float)


def choose_next(gp, cand, idx, model, rng, ts_subset=500):
    """Die Auswahlregel: wählt den noch nicht ausgewerteten Kandidaten mit dem größten Akquisitionswert. `best` = bester Mittelwert an den Beobachtungen."""
    n = len(cand)
    free = np.ones(n, bool)
    free[idx] = False
    if model.acquisition == "ts":
        pool = np.flatnonzero(free)
        if len(pool) > ts_subset:
            pool = rng.choice(pool, size=ts_subset, replace=False)
        draw = gp.sample(cand[pool], rng)
        return int(pool[int(np.argmin(draw))])
    mu, sd = gp.predict(cand)
    if model.acquisition == "ei":
        a = expected_improvement(mu, sd, float(np.min(gp.predict(cand[idx])[0])))
    elif model.acquisition == "pi":
        a = probability_of_improvement(mu, sd, float(np.min(gp.predict(cand[idx])[0])))
    elif model.acquisition == "ucb":
        a = lower_confidence_bound(mu, sd, model.beta)
    else:
        raise ValueError(model.acquisition)
    a = np.where(free, a, -np.inf)
    return int(np.argmax(a))


def bo_search(cand, evaluate, budget, rng, model, n_start=5):
    """Bayessche Optimierung mit dem gegebenen Modell: Startdesign zufällig, danach je Schritt GP anpassen und `choose_next` ausführen."""
    n = len(cand)
    idx = [int(i) for i in rng.choice(n, size=min(n_start, budget), replace=False)]
    y = [evaluate(i) for i in idx]
    while len(idx) < budget:
        gp = GaussianProcess(model.kernel, model.lengthscale, model.noise).fit(cand[idx], _transform(y, model))
        i = choose_next(gp, cand, idx, model, rng)
        idx.append(i)
        y.append(evaluate(i))
    return idx, y


def gp_state(cand, idx, y, model):
    """Zustand des Ersatzmodells nach den Beobachtungen: Mittelwert und Standardabweichung auf der Wertskala (bei log: Rücktransformation des Mittelwerts) und
    erwartete Verbesserung je Kandidat. Für die Anschauung."""
    gp = GaussianProcess(model.kernel, model.lengthscale, model.noise).fit(cand[idx], _transform(y, model))
    mu, sd = gp.predict(cand)
    best = float(np.min(gp.predict(cand[idx])[0]))
    ei = expected_improvement(mu, sd, best)
    if model.transform == "log":
        return np.exp(mu), np.exp(mu) * sd, ei, gp
    return mu, sd, ei, gp


def recommend(idx, y):
    """Empfehlung: der Kandidat mit der besten (kleinsten) Beobachtung."""
    return idx[int(np.argmin(y))]
