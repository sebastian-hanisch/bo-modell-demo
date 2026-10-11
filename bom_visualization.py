"""Plotly-Abbildungen (alle Achsen fest, damit Touch-Scrollen auf der Seite nicht am Diagramm hängen bleibt)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import bom_algorithm as A
import bom_constants as C
import bom_scenario as S

SURFACE = {
    "branin": {"zmin": 0.0, "zmax": 100.0, "x": "x1", "y": "x2"},
    "terminal": {"zmin": 750.0, "zmax": 1500.0, "x": "Tempo Stapel", "y": "Plätze am Stapel"},
}


def _fixed(fig, **layout):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10), legend=dict(orientation="h", y=-0.2), **layout)
    return fig


def _heat(a, z, colorscale, zmin=None, zmax=None):
    return go.Heatmap(z=z, x=a.x_values, y=a.y_values, colorscale=colorscale, zmin=zmin, zmax=zmax, showscale=False,
                      hovertemplate="%{x}<br>%{y}<br>%{z:.1f}<extra></extra>")


def build_surface_chart(a, step, which, model):
    """Wahrheit, Mittelwert des Ersatzmodells nach `step` Auswertungen und erwartete Verbesserung für den Lauf `which` ('default' oder 'choice')."""
    cfg = SURFACE[a.kind]
    run = a.runs[which]
    shape = a.truth.shape
    pts = S.grid_points(shape)
    cand = S.to_unit(pts, shape)
    idx = run.idx[:step]
    mu, sd, ei, _ = A.gp_state(cand, idx, run.y[:step], model)
    clip = lambda z: np.clip(z, cfg["zmin"], cfg["zmax"])
    fig = make_subplots(rows=1, cols=3, subplot_titles=("Wahrheit", "Ersatzmodell (Mittel)", "Erwartete Verbesserung"), horizontal_spacing=0.06)
    fig.add_trace(_heat(a, clip(a.truth), "YlGnBu_r", cfg["zmin"], cfg["zmax"]), row=1, col=1)
    fig.add_trace(_heat(a, clip(mu.reshape(shape)), "YlGnBu_r", cfg["zmin"], cfg["zmax"]), row=1, col=2)
    fig.add_trace(_heat(a, ei.reshape(shape), "Oranges"), row=1, col=3)
    px = [a.x_values[int(pts[i][1])] for i in idx]
    py = [a.y_values[int(pts[i][0])] for i in idx]
    for col in (1, 2, 3):
        fig.add_trace(go.Scatter(x=px, y=py, mode="markers+text", text=[str(k + 1) for k in range(len(idx))], textposition="top center", textfont=dict(size=9),
                                 marker=dict(size=8, color=C.COLOR_OBS, line=dict(color="white", width=1)), name="ausgewertet", showlegend=(col == 1),
                                 hovertemplate="Auswertung %{text}<extra></extra>"), row=1, col=col)
    oi, oj, _ = a.optimum
    fig.add_trace(go.Scatter(x=[a.x_values[oj]], y=[a.y_values[oi]], mode="markers", marker=dict(size=14, symbol="star", color=C.COLOR_NEXT, line=dict(color="white", width=1)),
                             name="Optimum der Gitterpunkte"), row=1, col=1)
    fig.update_xaxes(title_text=cfg["x"])
    fig.update_yaxes(title_text=cfg["y"], col=1)
    fig.update_annotations(font_size=12)
    return _fixed(fig, height=380)


def build_convergence_chart(a):
    """Wahrer Wert der Empfehlung nach t Auswertungen für Zufall, Standardmodell und gewähltes Modell (Regret; bei Terminal wahre Kosten)."""
    fig = go.Figure()
    ts = list(range(1, a.settings.n + 1))
    opt = a.optimum[2]
    spec = (("random", "Zufall", C.COLOR_RANDOM, 1.8), ("default", "Standardmodell", C.COLOR_DEFAULT, 2.5), ("choice", "Gewähltes Modell", C.COLOR_CHOICE, 3.0))
    for key, name, col, w in spec:
        tr = a.runs[key].truth_trace
        y = [max(v - opt, 1e-3) for v in tr] if a.kind != "terminal" else [min(v, 3000) for v in tr]
        fig.add_trace(go.Scatter(x=ts, y=y, mode="lines", name=name, line=dict(color=col, width=w, dash="dot" if key == "default" else "solid")))
    fig.update_xaxes(title_text="Auswertungen")
    if a.kind == "terminal":
        fig.add_hline(y=opt, line_dash="dash", line_color=C.COLOR_NEXT, annotation_text="wahres Optimum", annotation_position="bottom right")
        fig.update_yaxes(title_text="Wahre Kosten der Empfehlung (EUR/h)", range=[opt * 0.98, opt * 1.2])
    else:
        fig.update_yaxes(title_text="Regret der Empfehlung (log)", type="log")
    return _fixed(fig, height=340)


def build_variant_chart(table, keys, checkpoint, random_value=None):
    """Median des Regrets je Variante an einem Prüfpunkt; Standardmodell dunkel, Varianten grün, optional Zufall als Linie."""
    fig = go.Figure()
    names = [C.VARIANT_LABEL[k] for k in keys]
    vals = [max(table[k][checkpoint]["median"], 1e-3) for k in keys]
    q90 = [max(table[k][checkpoint]["q90"], 1e-3) for k in keys]
    fig.add_trace(go.Bar(x=names, y=vals, marker_color=[C.COLOR_DEFAULT if k == "default" else C.COLOR_CHOICE for k in keys], name="Median",
                         hovertemplate="%{x}<br>Median %{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=names, y=q90, mode="markers", marker=dict(symbol="diamond", size=9, color=C.COLOR_BAD), name="schlechtestes Zehntel der Läufe (90-%-Quantil)"))
    if random_value is not None:
        fig.add_hline(y=max(random_value, 1e-3), line_dash="dot", line_color=C.COLOR_RANDOM, annotation_text="Zufall", annotation_position="top left")
    fig.update_yaxes(title_text="Regret nach %d Auswertungen (log)" % checkpoint, type="log")
    return _fixed(fig, height=340)
