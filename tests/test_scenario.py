"""Landschaften: Form, Wahrheit, Ziehen ohne Zurücklegen, Reproduzierbarkeit."""

import numpy as np
import pytest

import bom_benchmarks as B
import bom_constants as C
import bom_evaluation as ev
import bom_scenario as S


def test_branin_grid_minimum_is_close_to_the_literature_value():
    _, _, truth = B.branin_grid(C.BRANIN_STEPS)
    assert truth.shape == (31, 31)
    assert float(truth.min()) == pytest.approx(0.427, abs=0.02)


def test_branin_has_three_equal_minima_by_hand():
    """Die drei bekannten Minimalstellen liefern denselben Wert 0,397887."""
    for x1, x2 in ((-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)):
        assert B.branin(x1, x2) == pytest.approx(B.BRANIN_MIN, abs=1e-5)


def test_hartmann6_known_minimum():
    x = np.array([0.20169, 0.150011, 0.476874, 0.275332, 0.311652, 0.6573])
    assert B.hartmann6(x) == pytest.approx(B.GLOBAL_MIN, abs=1e-4)


def test_terminal_table_shape_and_optimum():
    t = S.table()
    assert t["truth"].shape == (C.N_PLACES, C.N_SPEEDS)
    i, j = np.unravel_index(int(np.argmin(t["truth"])), t["truth"].shape)
    assert C.PLACES[i] == 5 and C.SPEEDS[j] == pytest.approx(0.7)


def test_terminal_observe_draws_without_replacement():
    f = S.TerminalField(15000, 1)
    vals = [f.observe(3, 4) for _ in range(30)]
    assert sorted(vals) == sorted(S.table()["reps"][15000][3, 4].tolist())


def test_branin_noise_has_the_requested_spread():
    f = S.BraninField(5.0, 3)
    draws = np.array([f.observe(10, 10) for _ in range(4000)])
    assert draws.mean() == pytest.approx(S.BraninField.truth_grid()[10, 10], abs=0.5)
    assert draws.std() == pytest.approx(5.0, rel=0.08)


def test_runs_are_reproducible_and_share_the_start_design():
    s = ev.Settings(n=10, noise=5.0, seed=7)
    a = ev.run_one("branin", ev.DEFAULT_MODEL, s, 10)
    b = ev.run_one("branin", ev.DEFAULT_MODEL, s, 10)
    assert a.idx == b.idx and a.y == b.y
    c = ev.run_one("branin", ev.variant_model("kernel_rbf"), s, 10)
    assert a.idx[:5] == c.idx[:5]                    # gleiches Startdesign, gleiche Rauschreihe: gepaarter Vergleich ist möglich


def test_optimum_values_match_the_landscapes():
    assert ev.optimum_of("branin") == pytest.approx(0.427, abs=0.02)
    assert ev.optimum_of("terminal") == pytest.approx(790.6, abs=1.0)
    assert ev.optimum_of("hartmann") == pytest.approx(-2.861, abs=0.01)
