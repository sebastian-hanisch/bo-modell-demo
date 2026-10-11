"""Einheiten des Kerns an Mini-Instanzen (Handrechnung): Kerne, Akquisitionen, Auswahlregel, Suche."""

import math

import numpy as np
import pytest

import bom_algorithm as A


def k1(kernel, r, ls=1.0, var=1.0):
    return A.kernel_matrix(np.array([[0.0]]), np.array([[r]]), np.array([ls]), var, kernel)[0, 0]


def test_kernels_by_hand():
    """Matérn-5/2 bei sqrt(5) r = 1: (1 + 1 + 1/3) e^-1; Matérn-3/2 bei sqrt(3) r = 1: 2 e^-1; RBF bei r = 1: e^-0,5; alle k(x, x) = Varianz."""
    assert k1("matern52", 1 / math.sqrt(5)) == pytest.approx((7 / 3) * math.exp(-1), rel=1e-12)
    assert k1("matern32", 1 / math.sqrt(3)) == pytest.approx(2 * math.exp(-1), rel=1e-12)
    assert k1("rbf", 1.0) == pytest.approx(math.exp(-0.5), rel=1e-12)
    for k in A.KERNELS:
        assert k1(k, 0.0, var=2.5) == pytest.approx(2.5)


def test_kernel_ordering_at_equal_distance():
    """Bei gleichem Abstand ist der RBF-Kern am dichtesten an 1 (glatteste Annahme), der Matérn-3/2-Kern am niedrigsten (r = 0,5, Längenmaß 1)."""
    assert k1("rbf", 0.5) > k1("matern52", 0.5) > k1("matern32", 0.5)


def test_acquisitions_by_hand():
    one = np.array([1.0])
    assert A.expected_improvement(np.array([0.0]), one, 0.0)[0] == pytest.approx(0.3989422804, rel=1e-9)
    assert A.expected_improvement(np.array([1.0]), one, 0.0)[0] == pytest.approx(0.0833155, rel=1e-5)
    assert A.probability_of_improvement(np.array([0.0]), one, 0.0)[0] == pytest.approx(0.5)
    assert A.probability_of_improvement(np.array([1.0]), one, 0.0)[0] == pytest.approx(0.158655, rel=1e-5)
    assert A.lower_confidence_bound(np.array([3.0]), np.array([1.0]), 4.0)[0] == pytest.approx(-(3.0 - 2.0))
    assert A.lower_confidence_bound(np.array([3.0]), np.array([1.0]), 0.0)[0] == pytest.approx(-3.0)


def test_fixed_lengthscale_is_used_and_not_learned(toy):
    cand, f = toy
    gp = A.GaussianProcess(lengthscale=0.37).fit(cand[[0, 6, 12, 20]], f[[0, 6, 12, 20]])
    assert np.allclose(gp.lengthscales, 0.37)


def test_fixed_noise_is_used_and_not_learned(toy):
    cand, f = toy
    gp = A.GaussianProcess(noise=0.25).fit(cand[[0, 6, 12, 20]], f[[0, 6, 12, 20]])
    assert gp.noise == pytest.approx(0.25)


def test_gp_learns_a_smaller_lengthscale_on_a_wigglier_function():
    x = np.linspace(0, 1, 15)[:, None]
    smooth = A.GaussianProcess(noise=1e-6).fit(x, np.sin(2 * x[:, 0]))
    wiggly = A.GaussianProcess(noise=1e-6).fit(x, np.sin(14 * x[:, 0]))
    assert wiggly.lengthscales[0] < smooth.lengthscales[0]


def test_choose_next_never_returns_an_evaluated_candidate(toy):
    cand, f = toy
    idx = [0, 6, 12, 20]
    rng = np.random.default_rng(1)
    for acq in A.ACQUISITIONS:
        model = A.Model(acquisition=acq, noise=1e-6)
        gp = A.GaussianProcess(noise=1e-6).fit(cand[idx], f[idx])
        for _ in range(5):
            assert A.choose_next(gp, cand, idx, model, rng) not in idx, acq


def test_ei_picks_the_valley_and_ucb_with_large_beta_picks_unexplored_space():
    cand = np.linspace(0, 1, 41)[:, None]
    idx = [0, 4, 8, 12, 40]
    f = (cand[:, 0] - 0.3) ** 2
    gp = A.GaussianProcess(noise=1e-6).fit(cand[idx], f[idx])
    rng = np.random.default_rng(2)
    i_ei = A.choose_next(gp, cand, idx, A.Model(acquisition="ei", noise=1e-6), rng)
    i_ucb = A.choose_next(gp, cand, idx, A.Model(acquisition="ucb", beta=1e4, noise=1e-6), rng)
    assert abs(cand[i_ei, 0] - 0.3) < 0.2                # nah am Tal
    assert 0.5 < cand[i_ucb, 0] < 0.95                   # in der größten Lücke (zwischen 0,3 und 1,0)


def test_bo_search_uses_the_start_design_and_never_repeats(toy):
    cand, f = toy
    for model in (A.Model(), A.Model(kernel="rbf"), A.Model(acquisition="ts"), A.Model(lengthscale=0.3), A.Model(transform="log")):
        idx, y = A.bo_search(cand, lambda i: f[i], 9, np.random.default_rng(3), model, n_start=4)
        assert len(idx) == 9 == len(set(idx)), model
        assert np.allclose(y, f[idx])


def test_bo_finds_the_optimum_of_a_smooth_1d_function(toy):
    cand, f = toy
    for seed in range(5):
        idx, y = A.bo_search(cand, lambda i: f[i], 10, np.random.default_rng(seed), A.Model(noise=1e-6), n_start=3)
        assert abs(A.recommend(idx, y) - 6) <= 1, seed


def test_random_search_has_no_repeats(toy):
    cand, f = toy
    idx, y = A.random_search(len(cand), lambda i: f[i], 8, np.random.default_rng(1))
    assert len(idx) == 8 == len(set(idx)) and np.allclose(y, f[idx])


def test_log_transform_back_transforms_the_mean(toy):
    cand, f = toy
    mu, sd, ei, _ = A.gp_state(cand, [0, 6, 12, 20], f[[0, 6, 12, 20]], A.Model(transform="log", noise=1e-6))
    assert np.allclose(mu[[0, 6, 12, 20]], f[[0, 6, 12, 20]], rtol=2e-2)
