"""Konstanten der Demo „Das Ersatzmodell entscheidet“: Landschaften, Modellvarianten, Regler, Presets, Farben.

Die Terminal-Scheibe (Plätze am Stapel x Tempo des Stapels, 12 x 17 Einstellungen) stammt als Tabelle echter Simulationen aus bayes-opt-demo
(`precomputed_slice.json`, dort erzeugt von `generate_precomputed.py`); Zeiten in Minuten, Kosten in EUR je Stunde.
"""

# --- Landschaften -----------------------------------------------------------------------------------------------------------------------
KINDS = ("branin", "terminal", "hartmann")
KIND_LABELS = {"branin": "Branin (zwei Eingaben)", "terminal": "Terminal-Scheibe (zwei Eingaben)", "hartmann": "Hartmann6 (sechs Eingaben)"}
BRANIN_STEPS = 31
BRANIN_NOISES = (0.0, 5.0, 20.0)
NOISE_LABEL = {0.0: "0 (rauschfrei)", 5.0: "5 (mäßig)", 20.0: "20 (stark)"}
DEFAULT_NOISE = 5.0
PLACES = tuple(range(2, 14))
SPEEDS = tuple(round(0.6 + 0.05 * i, 2) for i in range(17))
N_PLACES, N_SPEEDS = len(PLACES), len(SPEEDS)
TRUCK_CHOICES = (3000, 15000)
DEFAULT_TRUCKS = 15000
HARTMANN_POOL = 4096

# --- Modellvarianten (je Stellschraube eine; Standard in der Mitte) ---------------------------------------------------------------------
KERNEL_LABELS = {"matern52": "Matérn-5/2 (zweimal differenzierbar)", "matern32": "Matérn-3/2 (einmal differenzierbar)", "rbf": "RBF (unendlich glatt)"}
LS_CHOICES = ("gelernt", "0,1 fest", "0,3 fest", "1,0 fest")
LS_VALUE = {"gelernt": None, "0,1 fest": 0.1, "0,3 fest": 0.3, "1,0 fest": 1.0}
NOISE_CHOICES = ("gelernt", "winzig fest", "groß fest")
NOISE_VALUE = {"gelernt": None, "winzig fest": 1e-6, "groß fest": 0.3}
ACQ_CHOICES = ("ei", "pi", "ucb", "ts")
ACQ_LABELS = {"ei": "Erwartete Verbesserung (EI)", "pi": "Wahrscheinlichkeit der Verbesserung (PI)", "ucb": "Konfidenzschranke (UCB)", "ts": "Thompson-Sampling (TS)"}
BETA_CHOICES = (0.25, 4.0, 64.0)
TRANSFORM_CHOICES = ("none", "log")
TRANSFORM_LABELS = {"none": "keine", "log": "Logarithmus der Kosten"}

# --- Studie -----------------------------------------------------------------------------------------------------------------------------
SWEEP_SEEDS = tuple(range(400000, 400020))
BUDGETS = {"branin": 30, "terminal": 30, "hartmann": 40}
STUDY_CHECKPOINTS = {"branin": (10, 20, 30), "terminal": (10, 20, 30), "hartmann": (10, 20, 40)}
N_START = 5
HARTMANN_N_START = 10

# Varianten der Studie: (Schlüssel, Beschriftung, Stellschraube) – alle gegen den Standard (Matérn-5/2, Längenmaß und Rauschen gelernt, keine Transformation, EI)
VARIANTS = (
    ("default", "Standard", "standard"),
    ("ls_0.1", "Längenmaß 0,1 fest", "ls"),
    ("ls_0.3", "Längenmaß 0,3 fest", "ls"),
    ("ls_1.0", "Längenmaß 1,0 fest", "ls"),
    ("kernel_matern32", "Matérn-3/2", "kernel"),
    ("kernel_rbf", "RBF", "kernel"),
    ("noise_tiny", "Rauschen winzig fest", "noise"),
    ("noise_big", "Rauschen groß fest", "noise"),
    ("log", "log(Kosten)", "transform"),
    ("acq_pi", "PI", "acq"),
    ("acq_ucb_0.25", "UCB β = 0,25", "acq"),
    ("acq_ucb_4", "UCB β = 4", "acq"),
    ("acq_ucb_64", "UCB β = 64", "acq"),
    ("acq_ts", "Thompson", "acq"),
)
VARIANT_LABEL = {k: lab for k, lab, _ in VARIANTS}
VARIANT_FACTOR = {k: f for k, _, f in VARIANTS}
FACTOR_LABELS = {"ls": "Längenmaß", "kernel": "Kern", "noise": "Rauschannahme", "transform": "Transformation", "acq": "Akquisition"}
# Das log ist nur dort definiert, wo die Beobachtungen positiv bleiben (Terminal-Kosten; nicht Branin mit Gauß-Rauschen, nicht Hartmann6).
VARIANT_KINDS = {k: (("terminal",) if k == "log" else KINDS) for k, _, _ in VARIANTS}

# --- Regler -----------------------------------------------------------------------------------------------------------------------------
N_MIN, N_MAX, DEFAULT_N = 6, 30, 16
SEED_MAX, DEFAULT_SEED = 999999, 35

_STD = {"kernel": "matern52", "ls": "gelernt", "noise_mode": "gelernt", "transform": "none", "acq": "ei", "beta": 4.0}
PRESETS = {
    "Standardmodell (Voreinstellung)": {"n": DEFAULT_N, "noise": DEFAULT_NOISE, "seed": DEFAULT_SEED, **_STD},
    "Längenmaß zu klein": {"n": DEFAULT_N, "noise": DEFAULT_NOISE, "seed": DEFAULT_SEED, **{**_STD, "ls": "0,1 fest"}},
    "Rauschen nicht beachtet": {"n": DEFAULT_N, "noise": 20.0, "seed": DEFAULT_SEED, **{**_STD, "noise_mode": "winzig fest"}},
    "Zu viel Neugier (UCB)": {"n": DEFAULT_N, "noise": DEFAULT_NOISE, "seed": DEFAULT_SEED, **{**_STD, "acq": "ucb", "beta": 64.0}},
    "Thompson-Sampling": {"n": DEFAULT_N, "noise": DEFAULT_NOISE, "seed": DEFAULT_SEED, **{**_STD, "acq": "ts"}},
}
PRESET_HELP = {
    "Standardmodell (Voreinstellung)": "Matérn-5/2, Längenmaß und Rauschen aus den Daten gelernt, Erwartete Verbesserung.",
    "Längenmaß zu klein": "Das Längenmaß ist auf 0,1 festgelegt: das Modell hält weit entfernte Einstellungen für unabhängig und lernt kaum aus den Beobachtungen.",
    "Rauschen nicht beachtet": "Starkes Rauschen (20), aber das Modell nimmt winziges Rauschen an und passt sich jedem Messwert an.",
    "Zu viel Neugier (UCB)": "Konfidenzschranke mit β = 64: die Unsicherheit zählt so stark, dass das Modell vor allem Ränder und Unerforschtes absucht.",
    "Thompson-Sampling": "Statt eines festen Maßes zieht das Modell eine mögliche Landschaft und optimiert diese Ziehung.",
}

COLOR_OBS = "#2F6B65"
COLOR_NEXT = "#D68A2E"
COLOR_BAD = "#B3463A"
COLOR_DEFAULT = "#14233B"
COLOR_CHOICE = "#2F6B65"
COLOR_RANDOM = "#8c8c8c"
