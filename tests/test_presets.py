"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Einrasten."""

import bom_constants as C
import bom_evaluation as ev
import bom_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_control_bounds_and_choices():
    for preset in C.PRESETS.values():
        assert C.N_MIN <= preset["n"] <= C.N_MAX and preset["noise"] in C.BRANIN_NOISES and 0 <= preset["seed"] <= C.SEED_MAX
        assert preset["kernel"] in C.KERNEL_LABELS and preset["ls"] in C.LS_CHOICES and preset["noise_mode"] in C.NOISE_CHOICES
        assert preset["transform"] in C.TRANSFORM_CHOICES and preset["acq"] in C.ACQ_CHOICES and preset["beta"] in C.BETA_CHOICES


def test_default_preset_is_the_default_model():
    p = C.PRESETS["Standardmodell (Voreinstellung)"]
    assert ev.Settings(n=p["n"], noise=p["noise"], seed=p["seed"]) == ev.Settings()
    assert ev.Choice(p["kernel"], p["ls"], p["noise_mode"], p["transform"], p["acq"], p["beta"]) == ev.Choice()
    assert ev.Choice().model() == ev.DEFAULT_MODEL


def test_setting_specs_cover_every_preset_key():
    assert set(P.PRESET_KEYS.values()) <= set(P.SETTING_SPECS)


def test_permalink_values_snap_to_allowed_choices():
    assert P.snap_to_choice(7.0, C.BRANIN_NOISES, 5.0) == 5.0
    assert P.snap_to_choice(100.0, C.BRANIN_NOISES, 5.0) == 20.0
    assert P.snap_to_choice("unsinn", C.LS_CHOICES, "gelernt") == "gelernt"
    assert P.snap_to_choice("0,3 fest", C.LS_CHOICES, "gelernt") == "0,3 fest"


def test_every_variant_key_maps_to_a_model_and_back_to_a_choice():
    for key, _, _ in C.VARIANTS:
        ev.variant_model(key)
    assert ev.choice_variant_key(ev.Choice()) == "default"
    assert ev.choice_variant_key(ev.Choice(kernel="rbf")) == "kernel_rbf"
    assert ev.choice_variant_key(ev.Choice(lengthscale="0,3 fest")) == "ls_0.3"
    assert ev.choice_variant_key(ev.Choice(acquisition="ucb", beta=64.0)) == "acq_ucb_64"
    assert ev.choice_variant_key(ev.Choice(kernel="rbf", acquisition="pi")) is None
