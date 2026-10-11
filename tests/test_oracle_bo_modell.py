"""Orakel-Tests bo-modell-demo: Kern gegen unabhängige Rechenwege (ORACLE-PLAYBOOK).

* Gauß-Prozess: Vorhersage je Kern bei den gelernten Hyperparametern gegen scikit-learn.
* Akquisitionen: EI und PI gegen numerische Integration; Thompson-Ziehungen gegen Mittelwert und Standardabweichung der Vorhersage.
* Vorzeichentest: gegen die Binomialverteilung von Hand (9 von 10: p = 22/1024).
* Studie: ein Lauf wird neu gerechnet und liegt im Bereich der abgelegten Läufe derselben Variante.
"""

import numpy as np
import pytest

import bom_algorithm as A
import bom_constants as C
import bom_evaluation as ev


def test_gp_prediction_matches_scikit_learn_for_every_kernel():
    gpm = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

    rng = np.random.default_rng(11)
    X = rng.random((10, 2))
    y = np.sin(3 * X[:, 0]) + X[:, 1] ** 2 + 0.05 * rng.standard_normal(10)
    for kernel in A.KERNELS:
        gp = A.GaussianProcess(kernel=kernel).fit(X, y)
        z = (y - gp.y_mean) / gp.y_std
        base = RBF(gp.lengthscales, "fixed") if kernel == "rbf" else Matern(gp.lengthscales, "fixed", nu=2.5 if kernel == "matern52" else 1.5)
        sk = gpm.GaussianProcessRegressor(ConstantKernel(gp.variance, "fixed") * base, alpha=gp.noise + 1e-8, optimizer=None).fit(X, z)
        Xs = rng.random((12, 2))
        m_ref, s_ref = sk.predict(Xs, return_std=True)
        mu, sd = gp.predict(Xs)
        assert np.allclose((mu - gp.y_mean) / gp.y_std, m_ref, atol=1e-6), kernel
        assert np.allclose(sd / gp.y_std, s_ref, atol=1e-5), kernel


def test_fixed_hyperparameters_give_the_same_prediction_as_scikit_learn():
    gpm = pytest.importorskip("sklearn.gaussian_process")
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    rng = np.random.default_rng(4)
    X = rng.random((8, 2))
    y = np.cos(4 * X[:, 0]) + X[:, 1]
    gp = A.GaussianProcess(lengthscale=0.4, noise=0.01).fit(X, y)
    z = (y - gp.y_mean) / gp.y_std
    sk = gpm.GaussianProcessRegressor(ConstantKernel(gp.variance, "fixed") * Matern([0.4, 0.4], "fixed", nu=2.5), alpha=0.01 + 1e-8, optimizer=None).fit(X, z)
    Xs = rng.random((9, 2))
    assert np.allclose((gp.predict(Xs)[0] - gp.y_mean) / gp.y_std, sk.predict(Xs), atol=1e-6)


def test_ei_and_pi_match_numerical_integration():
    quad = pytest.importorskip("scipy.integrate").quad
    from scipy.stats import norm

    for mu, sd, best in [(0.3, 0.5, 0.0), (-0.2, 0.1, 0.0), (1.0, 2.0, 0.0), (0.0, 1.0, 0.5)]:
        lo, hi = mu - 12 * sd, mu + 12 * sd
        ei_ref, _ = quad(lambda f: max(best - f, 0.0) * norm.pdf(f, mu, sd), lo, hi, limit=200)
        pi_ref, _ = quad(lambda f: norm.pdf(f, mu, sd), lo, best, limit=200)
        assert A.expected_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(ei_ref, rel=1e-6, abs=1e-9)
        assert A.probability_of_improvement(np.array([mu]), np.array([sd]), best)[0] == pytest.approx(pi_ref, rel=1e-6, abs=1e-9)


def test_thompson_draws_have_the_predicted_mean_and_spread():
    rng = np.random.default_rng(9)
    X = rng.random((6, 1))
    gp = A.GaussianProcess(noise=0.05).fit(X, np.sin(5 * X[:, 0]))
    Xs = np.array([[0.1], [0.5], [0.9]])
    draws = np.array([gp.sample(Xs, np.random.default_rng(s)) for s in range(1500)])
    mu, sd = gp.predict(Xs)
    assert np.allclose(draws.mean(0), mu, atol=0.08 * gp.y_std + 0.03)
    assert np.allclose(draws.std(0), sd, rtol=0.12, atol=0.03)


def test_sign_test_by_hand():
    """9 von 10 besser: zweiseitig p = 2 (C(10,9) + C(10,10)) / 2^10 = 22/1024; 5 von 10: p = 1."""
    from scipy.stats import binomtest
    assert binomtest(9, 10, 0.5).pvalue == pytest.approx(22 / 1024)
    assert binomtest(5, 10, 0.5).pvalue == pytest.approx(1.0)


def test_paired_stats_counts_and_p_value_on_a_known_pair(monkeypatch):
    """Künstliche Studie: Standard immer 10, Variante 9 mal 9 und einmal 11: besser 9, schlechter 1, p = 22/1024."""
    fake = {"runs": {"branin_5.0": {"default": [[0.0] * 9 + [10.0]] * 10, "kernel_rbf": [[0.0] * 9 + [9.0]] * 9 + [[0.0] * 9 + [11.0]]}}}
    monkeypatch.setattr(ev, "study", lambda: fake)
    s = ev.paired_stats("branin", "kernel_rbf", 10, 5.0)
    assert (s["better"], s["worse"], s["equal"]) == (9, 1, 0)
    assert s["p"] == pytest.approx(22 / 1024)
    assert ev.verdict_of(s) == "besser"
    assert s["median_diff"] == pytest.approx(-1.0)


def test_study_run_is_reproducible_within_the_stored_range():
    st = ev.study()["runs"]["branin_5.0"]["kernel_rbf"]
    seed = C.SWEEP_SEEDS[0]
    run = ev.run_one("branin", ev.variant_model("kernel_rbf"), ev.Settings(n=C.BUDGETS["branin"], noise=5.0, seed=seed), C.BUDGETS["branin"])
    stored = np.array(st)[:, -1]
    assert stored.min() - 1e-9 <= run.truth_trace[-1] <= stored.max() + 1e-9
    assert run.truth_trace[0] == pytest.approx(st[0][0], abs=1e-9)             # dasselbe Startdesign: gleiche erste Empfehlung
