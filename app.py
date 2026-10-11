"""Das Ersatzmodell entscheidet: Kern, Längenmaß, Rauschen, Transformation und Akquisition - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Linie Bayessche Optimierung. Dieselbe Suche wie im ersten Stück, aber jede Stellschraube des Ersatzmodells einzeln gegen den Standard geprüft
(20 feste Läufe, gleiche Startdesigns, gepaarter Vergleich).

Lauffähig mit: streamlit run app.py
"""

import streamlit as st

import bom_constants as C
import bom_evaluation as ev
from bom_evaluation import Choice, Settings, analyse
from bom_pdf_export import build_pdf
from bom_presets import (
    apply_preset,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sync_query_params,
)
from bom_visualization import build_convergence_chart, build_surface_chart, build_variant_chart

st.set_page_config(page_title="Das Ersatzmodell entscheidet – Sebastian Hanisch", layout="wide")


def de(x, nd=2):
    return f"{x:,.{nd}f}".replace(",", " ").replace(".", ",")


@st.cache_data(show_spinner=False)
def _analysis(settings, choice, kind):
    return analyse(settings, choice, kind)


st.title("🎚️ Das Ersatzmodell entscheidet: wie viel hängt an seinen Stellschrauben?")
st.markdown(
    """
Im ersten Stück der Linie lernte ein **Gauß-Prozess** aus wenigen Läufen eine Karte der Werte und setzte den nächsten Lauf dorthin, wo die erwartete Verbesserung am größten ist. Dabei
standen fünf Entscheidungen offen, die in der Praxis „einfach so“ getroffen werden: **welcher Kern** (wie glatt die Karte sein darf), ob das **Längenmaß** aus den Daten gelernt oder fest
gesetzt wird, ob das **Rauschen** mitgelernt wird, ob man die Werte vorher **logarithmiert** und welche **Akquisitionsfunktion** den nächsten Lauf wählt. Die Demo dreht **jede
Stellschraube einzeln** gegen das Standardmodell und fragt ehrlich: **Wie viel entscheidet sie?** Dazu laufen 20 feste Läufe je Variante auf drei Landschaften, **gepaart** (gleiches
Startdesign, gleiche Rauschreihe), und ein Vorzeichentest sagt, ob ein Unterschied mehr ist als Zufall. Die Demo ist das zweite Stück der Linie **Bayessche Optimierung**.

Weiter unten: drei Experimente (**🔬**), **🚧 Wo die Annahmen enden** und der Expander **📐 Mathematische Formulierung**.
"""
)

with st.expander("Die fünf Stellschrauben", expanded=True):
    st.markdown(
        """
| Stellschraube | Was sie festlegt | Varianten der Studie |
|---|---|---|
| **Kern** | Wie glatt die Karte sein darf: ob benachbarte Einstellungen ähnliche Werte haben und wie schnell sich das ändert. | Matérn-5/2 (Standard), Matérn-3/2, RBF |
| **Längenmaß** | Auf welcher Entfernung zwei Einstellungen noch etwas voneinander wissen. Klein: die Karte zerfällt in Inseln. Groß: sie ist fast eben. | gelernt (Standard), fest 0,1 / 0,3 / 1,0 (Einheitswürfel) |
| **Rauschannahme** | Wie sehr das Modell einem einzelnen Messwert traut. | gelernt (Standard), winzig fest, groß fest |
| **Transformation** | Ob die Werte vorher logarithmiert werden (dämpft Ausreißer nach oben, nur bei positiven Werten möglich). | keine (Standard), log (nur Terminal) |
| **Akquisition** | Wie aus Mittelwert und Unsicherheit der nächste Lauf wird. | EI (Standard), PI, UCB mit β = 0,25 / 4 / 64, Thompson-Sampling |
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
cols = st.columns(len(C.PRESETS))
for col, name in zip(cols, C.PRESETS):
    with col:
        st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Die Suche**")
    kind = st.radio("Landschaft", C.KINDS, format_func=lambda k: C.KIND_LABELS[k], key="kind_radio")
    n = st.slider("Auswertungen", *bounds("n_slider"), key="n_slider", step=1, help="Wie viele Läufe die Suche bezahlen darf (davon 5 bzw. bei Hartmann6 10 im Startdesign).")
    noise = st.select_slider("Rauschen (Branin)", C.BRANIN_NOISES, key="noise_select", format_func=lambda z: C.NOISE_LABEL[z],
                             help="Standardabweichung des Rauschens auf jeder Auswertung der Branin-Landschaft. Terminal: echte Simulationen, Hartmann6: rauschfrei.")
    st.markdown("**Das Ersatzmodell**")
    kernel = st.selectbox("Kern", list(C.KERNEL_LABELS), format_func=lambda k: C.KERNEL_LABELS[k], key="kernel_select")
    ls = st.select_slider("Längenmaß", C.LS_CHOICES, key="ls_select", help="„gelernt“: je Eingabe aus den Daten (Randwahrscheinlichkeit). Sonst für alle Eingaben fest.")
    noise_mode = st.select_slider("Rauschannahme", C.NOISE_CHOICES, key="noise_mode_select",
                                  help="„winzig fest“: das Modell nimmt rauschfreie Daten an. „groß fest“: es nimmt viel Rauschen an. „gelernt“: aus den Daten.")
    transform = st.selectbox("Transformation", C.TRANSFORM_CHOICES, format_func=lambda k: C.TRANSFORM_LABELS[k], key="transform_select", disabled=(kind != "terminal"),
                             help="Nur für das Terminal: dort sind die Kosten positiv. Branin mit Rauschen und Hartmann6 haben negative Werte.")
    acq = st.selectbox("Akquisition", C.ACQ_CHOICES, format_func=lambda k: C.ACQ_LABELS[k], key="acq_select")
    beta = st.select_slider("β der Konfidenzschranke", C.BETA_CHOICES, key="beta_select", disabled=(acq != "ucb"),
                            help="Nur für UCB: Gewicht der Unsicherheit (√β Standardabweichungen). Groß heißt: die Suche ist neugierig.")
    seed = st.number_input("Zufalls-Seed des Laufs", *bounds("seed_input"), key="seed_input", step=1)
    st.button("🎲 Neuen Lauf generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"kind_radio": kind, "n_slider": int(n), "noise_select": float(noise), "seed_input": int(seed), "kernel_select": kernel, "ls_select": ls,
                   "noise_mode_select": noise_mode, "transform_select": transform, "acq_select": acq, "beta_select": float(beta)})

settings = Settings(n=int(n), noise=float(noise), seed=int(seed))
choice = Choice(kernel=kernel, lengthscale=ls, noise=noise_mode, transform=("log" if (transform == "log" and kind == "terminal") else "none"),
                acquisition=acq, beta=float(beta))
a = _analysis(settings, choice, kind)
reg = ev.final_regret(a)

# --- Ergebnis ------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Ergebnis")
st.caption(f"Landschaft: **{C.KIND_LABELS[kind]}**. Zufall, Standardmodell und gewähltes Modell starten mit demselben Seed; Standard und Wahl teilen das Startdesign und die Rauschreihe.")
unit = " EUR/h" if kind == "terminal" else ""
m1, m2, m3 = st.columns(3)
m1.metric("Zufall", f"{de(reg['random'])}{unit}", help="Regret: wahrer Wert der empfohlenen Einstellung minus bester Wert der Landschaft. 0 wäre die beste Einstellung.")
m2.metric("Standardmodell", f"{de(reg['default'])}{unit}", delta=f"{reg['default'] - reg['random']:+.2f} gegen Zufall".replace(".", ","), delta_color="inverse")
m3.metric("Gewähltes Modell", f"{de(reg['choice'])}{unit}", delta=f"{reg['choice'] - reg['default']:+.2f} gegen Standard".replace(".", ","), delta_color="inverse")
st.plotly_chart(build_convergence_chart(a), width="stretch", key="convergence_chart")

if kind != "hartmann":
    st.markdown("**Was das gewählte Modell glaubt** – Schritt für Schritt nachvollziehen")
    c1, c2 = st.columns(2)
    with c1:
        step = st.slider("Nach wie vielen Auswertungen?", C.N_START, C.N_MAX, C.N_START + 3, key="surface_step",
                         help="Bis zur gewählten Zahl der Auswertungen; bei mehr als dem Budget gilt das Budget.")
    with c2:
        which = st.radio("Lauf", ("choice", "default"), format_func=lambda k: "Gewähltes Modell" if k == "choice" else "Standardmodell", key="surface_which", horizontal=True)
    step = int(min(step, settings.n))
    st.plotly_chart(build_surface_chart(a, step, which, choice.model() if which == "choice" else ev.DEFAULT_MODEL), width="stretch", key=f"surface_{kind}_{which}_{step}")
    st.caption(f"Nach {step} Auswertungen. Links die Wahrheit, in der Mitte der Mittelwert des Ersatzmodells (grüne Punkte, Nummer = Reihenfolge), rechts, wo es als Nächstes hinsähe.")
else:
    st.caption("Hartmann6 hat sechs Eingaben und lässt sich nicht als Fläche zeigen; die Verlaufskurve oben und die Studie unten gelten.")

st.download_button("📄 Auswertung als PDF", data=build_pdf(a, choice), file_name="ersatzmodell_auswertung.pdf", mime="application/pdf", key="pdf_button")

# --- Kernfrage -----------------------------------------------------------------------------------------------------------------------------

st.markdown("## ❓ Kernfrage: Lohnt es, am Modell zu drehen?")
key = ev.choice_variant_key(choice)
cp_last = C.STUDY_CHECKPOINTS[kind][-1]
if key is None:
    st.info("Diese Kombination mehrerer Stellschrauben steht nicht in der Studie, die jede Stellschraube einzeln prüft. Ein einzelner Lauf zeigt nur, was bei diesem Seed passiert ist.")
elif key == "default":
    st.info("Das ist das Standardmodell. Drehen Sie in der Seitenleiste an einer Stellschraube: darunter erscheint, was die 20 festen Läufe der Studie dazu sagen.")
elif key not in ev.study()["runs"][ev.study_key(kind, noise)]:
    st.info("Diese Variante gibt es für die gewählte Landschaft nicht in der Studie.")
else:
    s = ev.paired_stats(kind, key, cp_last, noise)
    v = ev.verdict_of(s)
    txt = (f"**{C.VARIANT_LABEL[key]}** gegen Standard nach {cp_last} Auswertungen in 20 gepaarten Läufen: in **{s['better']}** besser, in **{s['worse']}** schlechter, in "
           f"{s['equal']} gleich (Median der Differenz {de(s['median_diff'])}{unit}; Vorzeichentest p = {de(s['p'], 3)}).")
    (st.success if v == "besser" else st.warning if v == "schlechter" else st.info)(f"{txt} Ergebnis: **{v}**.")
st.caption("Ein einzelner Lauf oben zeigt nur, was bei diesem Seed geschah; das Urteil steht in den 20 gepaarten Läufen der Studie.")

st.markdown("---")

# --- Experimente ---------------------------------------------------------------------------------------------------------------------------


def study_block(kind_, noise_, title, number):
    st.subheader(f"🔬 Experiment {number}: {title}")
    tab = ev.regret_table(kind_, noise_)
    cps = C.STUDY_CHECKPOINTS[kind_]
    cp = st.select_slider("Prüfpunkt (Auswertungen)", cps, value=cps[-1], key=f"cp_{number}",
                          help="An welcher Zahl von Auswertungen die Varianten verglichen werden.")
    keys = [k for k in tab if k != "random"]
    st.plotly_chart(build_variant_chart(tab, keys, cp, tab["random"][cp]["median"]), width="stretch", key=f"variants_{number}_{cp}")
    rows = []
    for k in keys:
        if k == "default":
            continue
        s = ev.paired_stats(kind_, k, cp, noise_)
        rows.append(f"| {C.VARIANT_LABEL[k]} | {C.FACTOR_LABELS[C.VARIANT_FACTOR[k]]} | {s['better']} | {s['worse']} | {s['equal']} | {de(s['median_diff'])} | {de(s['p'], 3)} | {ev.verdict_of(s)} |")
    st.markdown(
        "Gepaarter Vergleich gegen das Standardmodell (20 Läufe; Differenz = Variante minus Standard, negativ ist besser):\n\n"
        "| Variante | Stellschraube | besser | schlechter | gleich | Median der Differenz | p | Urteil |\n|---|---|---|---|---|---|---|---|\n" + "\n".join(rows)
    )
    return tab, cp


st.caption("Je Experiment: Median des Regrets je Variante (Balken), das schlechteste Zehntel der Läufe (Raute) und der Zufall (Linie). Der Vorzeichentest zählt nur Läufe mit Unterschied; "
           "bei rund 13 Vergleichen je Tabelle können einzelne p-Werte unter 0,05 Zufall sein.")
tabB, cpB = study_block("branin", float(noise), f"Branin mit Rauschen {C.NOISE_LABEL[float(noise)]}", 1)
okB = {k: ev.verdict_of(ev.paired_stats("branin", k, cpB, float(noise))) for k in tabB if k not in ("random", "default")}
nbad, ngood = sum(v == "schlechter" for v in okB.values()), sum(v == "besser" for v in okB.values())
st.info(f"Bei {cpB} Auswertungen: {nbad} Varianten klar schlechter, {ngood} klar besser als das Standardmodell, {len(okB) - nbad - ngood} ohne nachweisbaren Unterschied. "
        + ("Ohne Rauschen verschlechtert **jede** Abweichung vom gelernten Modell oder von einer mäßigen Neugier das Ergebnis." if float(noise) == 0.0 else
           "Mit Rauschen streuen die Läufe so stark, dass 20 Läufe kaum etwas trennen."))

tabT, cpT = study_block("terminal", None, "Terminal-Scheibe (echte Simulationen)", 2)
sT = {k: ev.paired_stats("terminal", k, cpT) for k in tabT if k not in ("random", "default")}
good = [C.VARIANT_LABEL[k] for k, s in sT.items() if ev.verdict_of(s) == "besser"]
bad = [C.VARIANT_LABEL[k] for k, s in sT.items() if ev.verdict_of(s) == "schlechter"]
st.info(f"Bei {cpT} Auswertungen: klar besser: {', '.join(good) if good else 'keine Variante'}; klar schlechter: {', '.join(bad) if bad else 'keine Variante'}. "
        "Das Tal der Kosten ist flach und alle Verfahren liegen dicht am Optimum: **Unterschiede sind klein, auch wenn sie nachweisbar sind.**" if (good or bad)
        else "Das Tal der Kosten ist flach und alle Verfahren liegen dicht am Optimum: **hier entscheidet das Modell fast nichts.**")

tabH, cpH = study_block("hartmann", None, "Hartmann6, sechs Eingaben, rauschfrei", 3)
sH = {k: ev.paired_stats("hartmann", k, cpH) for k in tabH if k not in ("random", "default")}
badH = [C.VARIANT_LABEL[k] for k, s in sH.items() if ev.verdict_of(s) == "schlechter"]
goodH = [C.VARIANT_LABEL[k] for k, s in sH.items() if ev.verdict_of(s) == "besser"]
st.info(f"Bei {cpH} Auswertungen: klar schlechter: {', '.join(badH) if badH else 'keine Variante'}; klar besser: {', '.join(goodH) if goodH else 'keine Variante'}. "
        "Das gelernte Modell ist hier schwer zu schlagen; wer an einer Stellschraube dreht, kann nur verlieren, und am deutlichsten an den Rändern (sehr großes festes Längenmaß, sehr neugierige Schranke).")

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **20 Läufe reichen** | Bei starkem Rauschen streuen die Läufe so stark, dass auch echte Unterschiede nicht nachweisbar sind: „kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“. | Mehr Läufe, andere Streuungsmaße |
| **Eine Stellschraube nach der anderen** | Wechselwirkungen (kleines Längenmaß **und** winziges Rauschen) sind nicht geprüft; jede Variante dreht nur eine Schraube. | Faktorielle Versuchsplanung |
| **Drei Landschaften stehen für die Praxis** | Branin ist glatt, die Terminal-Scheibe ein flaches Tal mit Klippe, Hartmann6 rauschfrei. Wie sich ein Modell auf Ihrem Problem verhält, ist damit nicht gesagt. | Eigene Messung |
| **Das Längenmaß ist für alle Eingaben gleich, wenn es fest ist** | Das gelernte Modell darf je Eingabe ein eigenes wählen; „fest“ ist hier eine einzige Zahl für alle. | Eingabewarping, eigene Längenmaße |
| **Mehrere Tests, ein Niveau** | Je Tabelle werden rund 13 Varianten gegen den Standard geprüft, ohne Korrektur: einzelne p-Werte unter 0,05 sind bei dieser Zahl auch durch Zufall zu erwarten. | Korrektur für mehrere Vergleiche |
    """
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Kerne** (Abstand $r = \sqrt{\sum_d (x_d - x'_d)^2/\ell_d^2}$, Signalvarianz $\sigma_f^2$):

$$k_{5/2}(r) = \sigma_f^2\Bigl(1 + \sqrt5\, r + \tfrac53 r^2\Bigr)e^{-\sqrt5\, r}$$

$$k_{3/2}(r) = \sigma_f^2\bigl(1 + \sqrt3\, r\bigr)e^{-\sqrt3\, r}$$

$$k_{\mathrm{RBF}}(r) = \sigma_f^2\, e^{-r^2/2}$$

**Vorhersage** mit $K = k(X, X) + \sigma_n^2 I$:

$$\mu(x) = k(x, X)\,K^{-1}y, \qquad s^2(x) = k(x, x) - k(x, X)\,K^{-1}k(X, x)$$

**Akquisitionen** (Minimierung, $f^\*$ = kleinster Mittelwert an den Beobachtungen, $z = (f^\* - \mu)/s$):

$$\mathrm{EI}(x) = (f^\* - \mu)\,\Phi(z) + s\,\varphi(z), \qquad \mathrm{PI}(x) = \Phi(z), \qquad \mathrm{UCB}(x) = -\bigl(\mu - \sqrt\beta\, s\bigr)$$

**Thompson-Sampling.** Eine gemeinsame Ziehung $\tilde f \sim \mathcal{GP}$ an einer zufälligen Teilmenge von bis zu 500 Kandidaten; gewählt wird $\arg\min \tilde f$.

**Vorzeichentest.** Je Lauf die Differenz $\Delta = $ Wert der Variante minus Wert des Standards (gleiche Seeds); ohne Gleichstände ist die Zahl der Läufe mit $\Delta < 0$ unter
„kein Unterschied“ binomialverteilt mit $p = 1/2$, der zweiseitige $p$-Wert kommt daraus.

**Literatur.** C. E. Rasmussen, C. K. I. Williams (2006), *Gaussian Processes for Machine Learning*, MIT Press (Kerne, Randwahrscheinlichkeit); D. R. Jones, M. Schonlau,
W. J. Welch (1998), *Efficient Global Optimization of Expensive Black-Box Functions*, Journal of Global Optimization 13, 455–492 (EI).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html)."
)
