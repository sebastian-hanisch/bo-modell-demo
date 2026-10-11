# Das Ersatzmodell entscheidet: wie viel hängt an seinen Stellschrauben? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-bo-modell-demo.streamlit.app/)**

Zweites Stück der **Bayessche-Optimierung-Linie** in der „Konzepte“-Reihe für die Website „Sebastian Hanisch – Operations Research und Machine Learning“.
Es setzt [bayes-opt-demo](https://github.com/sebastian-hanisch/bayes-opt-demo) fort: dort entschied die Demo nur zwischen Suchverfahren, hier dreht sie **jede Stellschraube des
Ersatzmodells einzeln** (Kern, Längenmaß, Rauschannahme, Transformation, Akquisition) und fragt, wie viel sie entscheidet. Im Unterschied zu den Fall-Demos zeigt die Demo ein
einziges Verfahren an einem wachsenden Beispiel.

## Warum dieses Problem

Wer Bayessche Optimierung einsetzt, trifft fünf Entscheidungen „einfach so“. Die Kernfrage: **Wie viel hängt daran, und lohnt es, am Standardmodell zu drehen?** Die Antwort ist
unspektakulär und deshalb nützlich: Das gelernte Standardmodell ist schwer zu schlagen, und die Abweichungen, die nachweislich schaden, sind die Ränder.

## Befunde aus der Vorarbeit

Die Phase-0-Messreihe der Linie (Branin, Hartmann6, Terminal) hatte nahegelegt, dass die Akquisitionsfunktion wenig und die Modell-Einstellung mehr zählt, und dass eine feste Längenskala
je nach Problem besser oder schlechter ist. Die Studie dieser Demo bestätigt das **nur zum Teil**: Mit gepaarten Läufen und Vorzeichentest bleibt von den vermuteten Effekten wenig übrig,
sobald die Auswertungen rauschen.

## Modell

- **Ersatzmodell:** Gauß-Prozess, Kern Matérn-5/2 (Standard), Matérn-3/2 oder RBF; Längenmaß je Eingabe gelernt oder für alle Eingaben fest (0,1 / 0,3 / 1,0 im Einheitswürfel);
  Rauschvarianz gelernt oder fest (winzig, 0,3 auf der standardisierten Ausgabe); Ausgabe optional logarithmiert.
- **Akquisition:** erwartete Verbesserung (Standard), Wahrscheinlichkeit der Verbesserung, Konfidenzschranke (β = 0,25 / 4 / 64) oder Thompson-Sampling (Ziehung an bis zu 500 Kandidaten).
- **Suche:** Startdesign 5 zufällige Läufe (Hartmann6: 10), danach je Schritt anpassen und den noch nicht ausgewerteten Kandidaten mit dem größten Akquisitionswert wählen.
  Empfehlung = beste Beobachtung; bewertet wird sie an der **Wahrheit**.
- **Landschaften:** Branin auf 31 × 31 mit Gauß-Rauschen (0, 5, 20); Terminal-Scheibe (Plätze × Stapeltempo, 12 × 17, Auswertungen aus einer Tabelle echter Simulationen
  aus `bayes-opt-demo`); Hartmann6, rauschfrei, Kandidatenmenge von 4 096 Punkten.
- Formeln im 📐-Expander, Kern in `bom_algorithm.py`.

## Methodik

1. **Live:** ein Lauf mit gewählten Stellschrauben gegen das Standardmodell und den Zufall, auf der gewählten Landschaft; das Ersatzmodell lässt sich Schritt für Schritt ansehen.
2. **Studie** (vorgerechnet, `generate_study.py`): 14 Modellvarianten, je 20 feste Läufe, drei Landschaften; Prüfpunkte 10/20/30 (Hartmann6: 10/20/40). Alle Varianten **teilen je
   Seed** das Startdesign und die Rauschreihe: der Vergleich mit dem Standard ist **gepaart**.
3. **Urteil:** zweiseitiger Vorzeichentest über die Läufe mit Unterschied (Gleichstände zählen nicht), ohne Korrektur für die vielen Vergleiche; Niveau 0,05.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. „Regret“ = wahrer Wert der Empfehlung minus bester Wert der Landschaft; Median über 20 feste Läufe.

| Frage | Befund |
|---|---|
| Branin **ohne Rauschen**: schlägt eine Abweichung das Standardmodell? | **Nein, im Gegenteil**: bei 30 Auswertungen sind **7 von 12** Varianten klar schlechter und **keine** besser. Ein festes Längenmaß (0,1, 0,3, 1,0) ist in 19, 13 bzw. 10 von je 20 Läufen schlechter und nie besser; Standard-Regret 0,0 gegen 0,89 beim Zufall. |
| Branin **mit Rauschen** (5 und 20) | Bei 30 Auswertungen ist **keine** Variante nachweisbar besser oder schlechter (kleinster p-Wert über 0,1): Die Läufe streuen zu stark für 20 Wiederholungen. **Das heißt nicht „kein Unterschied“.** |
| Hartmann6, sechs Eingaben | Bei 40 Auswertungen sind nur **zwei** Varianten klar schlechter: ein festes Längenmaß von 1,0 (14 gegen 1 Lauf) und die **sehr neugierige Konfidenzschranke β = 64** (19 gegen 0). Kern, Rauschannahme, PI, UCB 0,25/4 und Thompson unterscheiden sich nicht nachweisbar. Standard-Regret 0,08 gegen 1,33 beim Zufall. |
| Terminal-Scheibe | Bei 30 Auswertungen ist **keine** Variante nachweisbar anders. Bei 20 Auswertungen ist **PI** besser als EI (14 gegen 2 Läufe, p < 0,01). Das Standardmodell ist hier **nicht besser als der Zufall** (Regret 9,1 gegen 8,2 EUR/h; flaches Tal). |
| Wo schadet eine Stellschraube sicher? | An den **Rändern**: sehr großes festes Längenmaß, sehr hohes β der Konfidenzschranke, sowie auf der rauschfreien Branin-Landschaft jede feste Längenmaß-Wahl und die Annahme von viel Rauschen. |

## Ehrliche Grenzen

- **„Kein Unterschied nachweisbar“ heißt nicht „kein Unterschied“.** Mit 20 Läufen und starkem Rauschen fehlt die Teststärke; die Mediane liegen bei Branin mit Rauschen 20 nach 30 Auswertungen
  zwischen 2,5 (Standard) und 10,2 (β = 64), und doch ist kein Unterschied nachweisbar.
- **Einzeln gedreht, nicht kombiniert.** Wechselwirkungen (kleines Längenmaß mit winzigem Rauschen) sind nicht geprüft.
- **Rund 13 Vergleiche je Tabelle ohne Korrektur.** Einzelne p-Werte unter 0,05 (etwa PI am Terminal bei 20 Auswertungen) können Zufall sein; belastbar sind die gehäuften Befunde
  (Branin ohne Rauschen, Hartmann6 an den Rändern).
- **Drei Landschaften.** Branin ist glatt, die Terminal-Scheibe ein flaches Tal mit Klippe, Hartmann6 rauschfrei und ohne Rauschen auf den Beobachtungen: das Bild auf Ihrem Problem
  kann anders aussehen.
- **Das fest gesetzte Längenmaß ist eine einzige Zahl für alle Eingaben**; das gelernte Modell darf je Eingabe ein eigenes wählen. Ein Teil des Vorteils des Standards liegt daran.
- **log nur am Terminal.** Auf Branin mit Rauschen und Hartmann6 sind Beobachtungen negativ; dort gibt es die Variante nicht.
- **Eigene Implementierung** (Kandidatenmenge, drei feste Startpunkte der Hyperparameter-Suche): Zahlen anderer Bibliotheken können abweichen.

## Verifikation

- **Kern:** Kerne und Akquisitionen an Mini-Instanzen von Hand, Auswahlregel, Suche (`tests/test_algorithm.py`).
- **Orakel** (`tests/test_oracle_bo_modell.py`, anderer Rechenweg):
  - Gauß-Prozess-Vorhersage je Kern gegen **scikit-learn** bei den gelernten Hyperparametern und bei festem Längenmaß und Rauschen.
  - EI und PI gegen **numerische Integration**; Thompson-Ziehungen gegen Mittelwert und Standardabweichung der Vorhersage.
  - **Vorzeichentest** gegen die Binomialverteilung von Hand (9 von 10: p = 22/1024).
  - Ein Studienlauf wird neu gerechnet und liegt im Bereich der abgelegten Läufe.
- **Landschaften:** Branin-Minima von Hand, Hartmann6-Minimum, Rauschen und Ziehen ohne Zurücklegen (`tests/test_scenario.py`).
- **Zahlen im README:** `tests/test_claims.py`.
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler an Minimum und Maximum) und Footer-Test.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `app.py` | Streamlit-App (Skelett nach DEMO-PLAYBOOK §2) |
| `bom_algorithm.py` | Kern: Gauß-Prozess (drei Kerne), vier Akquisitionen, Auswahlregel, Suche |
| `bom_scenario.py`, `bom_benchmarks.py` | Landschaften (Branin, Terminal-Tabelle, Hartmann6) |
| `bom_evaluation.py` | Läufe, gepaarte Statistik, Studienauswertung |
| `bom_constants.py` | Regler-Grenzen, Varianten, Farben, Presets |
| `bom_presets.py` | Permalink (Begrenzen/Einrasten), Presets, Zufalls-Seed-Button |
| `bom_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `bom_pdf_export.py` | PDF-Auswertung (fpdf2) |
| `generate_study.py`, `precomputed_study.json` | Studie (14 Varianten, 20 Läufe, drei Landschaften) |
| `precomputed_slice.json` | Tabelle der Terminal-Scheibe (aus `bayes-opt-demo`) |
| `requirements.txt`, `requirements-dev.txt` | Laufzeit bzw. Laufzeit plus pytest und scikit-learn |
| `tests/` | Einheiten, Orakel, Behauptungen, Presets, Landschaften, AppTest, Footer |

## Bewusst nicht umgesetzt

- **Kombinationen mehrerer Stellschrauben** (faktorielle Versuchsplanung) und **eigene Längenmaße je Eingabe** bei festen Werten.
- **Andere Akquisitionsfunktionen** (Wissensgradient, Entropie-Suche) und **andere Kerne** (rationale quadratische, periodische).
- **Korrektur für mehrere Vergleiche** (Holm, Bonferroni).

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und fpdf2.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zu den Demos: [Interaktive Demos](https://sebastianhanisch.net/demos.html).
