"""Jede Zahl aus dem README, nachgerechnet über dieselben Auswertungsfunktionen wie die App (vorgerechnete Studie, 20 feste gepaarte Läufe).

Die Studie ist festgeschrieben (precomputed_study.json); die Bänder decken eine Neuberechnung auf anderer Plattform ab."""

import pytest

import bom_constants as C
import bom_evaluation as ev


def verdicts(kind, cp, noise=None):
    tab = ev.regret_table(kind, noise)
    return {k: ev.verdict_of(ev.paired_stats(kind, k, cp, noise)) for k in tab if k not in ("random", "default")}


def names(v, what):
    return sorted(k for k, x in v.items() if x == what)


def test_branin_without_noise_most_departures_from_the_default_are_worse():
    v = verdicts("branin", 30, 0.0)
    assert len(v) == 12
    assert len(names(v, "schlechter")) == 7 and not names(v, "besser")
    assert {"ls_0.1", "ls_0.3", "ls_1.0", "noise_big", "acq_ucb_64"} <= set(names(v, "schlechter"))
    s = ev.paired_stats("branin", "ls_0.1", 30, 0.0)
    assert (s["better"], s["worse"]) == (0, 19) and s["p"] < 0.001
    t = ev.regret_table("branin", 0.0)
    assert t["default"][30]["median"] == pytest.approx(0.0, abs=0.05)
    assert t["random"][30]["median"] == pytest.approx(0.89, abs=0.3)


def test_branin_with_noise_no_variant_is_distinguishable():
    for noise in (5.0, 20.0):
        v = verdicts("branin", 30, noise)
        assert not names(v, "besser") and not names(v, "schlechter"), noise
        assert min(ev.paired_stats("branin", k, 30, noise)["p"] for k in v) > 0.1


def test_hartmann6_only_the_extremes_are_worse():
    v = verdicts("hartmann", 40)
    assert names(v, "schlechter") == ["acq_ucb_64", "ls_1.0"] and not names(v, "besser")
    s = ev.paired_stats("hartmann", "acq_ucb_64", 40)
    assert (s["better"], s["worse"]) == (0, 19)
    t = ev.regret_table("hartmann")
    assert t["default"][40]["median"] == pytest.approx(0.08, abs=0.1)
    assert t["random"][40]["median"] == pytest.approx(1.33, abs=0.3)


def test_terminal_slice_nothing_is_distinguishable_at_the_largest_budget_but_pi_wins_at_20():
    assert not names(verdicts("terminal", 30), "besser") and not names(verdicts("terminal", 30), "schlechter")
    s = ev.paired_stats("terminal", "acq_pi", 20)
    assert (s["better"], s["worse"]) == (14, 2) and s["p"] < 0.01
    t = ev.regret_table("terminal")
    assert t["default"][30]["median"] == pytest.approx(9.1, abs=2.0)
    assert t["random"][30]["median"] == pytest.approx(8.2, abs=2.0)            # das Standardmodell ist hier nicht besser als Zufall


def test_study_geometry():
    st = ev.study()
    assert st["meta"]["seeds"] == 20
    for name, runs in st["runs"].items():
        for key, traces in runs.items():
            assert len(traces) == 20, (name, key)
    assert set(st["runs"]) == {"branin_0.0", "branin_5.0", "branin_20.0", "terminal", "hartmann"}


def test_log_transform_only_for_the_terminal():
    assert C.VARIANT_KINDS["log"] == ("terminal",)
    assert "log" in ev.study()["runs"]["terminal"] and "log" not in ev.study()["runs"]["branin_5.0"]
