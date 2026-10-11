"""Vorgerechnete Studie: jede Stellschraube des Ersatzmodells einzeln gegen den Standard, 20 feste Läufe, drei Landschaften (rund 10 bis 20 Minuten auf 14 Kernen).

    python generate_study.py            # schreibt precomputed_study.json

Je Lauf wird die Folge der wahren Werte der Empfehlung (beste Beobachtung) nach 1 … N Auswertungen abgelegt; das Optimum der Gitterpunkte bzw. der Kandidatenmenge
steht in "optimum".
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import bom_constants as C
import bom_evaluation as ev


def job(args):
    kind, noise, key, seed = args
    st = ev.Settings(n=C.BUDGETS[kind], noise=noise, seed=seed)
    model = None if key == "random" else ev.variant_model(key)
    run = ev.run_one(kind, model, st, C.BUDGETS[kind])
    return kind, noise, key, seed, run.truth_trace


def main(workers=14):
    t0 = time.time()
    jobs = []
    for noise in C.BRANIN_NOISES:
        for key in ["random"] + [k for k, _, _ in C.VARIANTS if "branin" in C.VARIANT_KINDS[k]]:
            jobs += [("branin", noise, key, s) for s in C.SWEEP_SEEDS]
    for key in ["random"] + [k for k, _, _ in C.VARIANTS if "terminal" in C.VARIANT_KINDS[k]]:
        jobs += [("terminal", C.DEFAULT_NOISE, key, s) for s in C.SWEEP_SEEDS]
    for key in ["random"] + [k for k, _, _ in C.VARIANTS if "hartmann" in C.VARIANT_KINDS[k]]:
        jobs += [("hartmann", C.DEFAULT_NOISE, key, s) for s in C.SWEEP_SEEDS]
    runs = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for kind, noise, key, seed, trace in ex.map(job, jobs, chunksize=2):
            name = f"branin_{float(noise)}" if kind == "branin" else kind
            runs.setdefault(name, {}).setdefault(key, []).append(trace)
    opt = {}
    for kind in C.KINDS:
        _, _, _, o = ev.landscape(kind)
        opt[kind] = o
    out = {"runs": runs, "optimum": opt, "meta": {"seeds": len(C.SWEEP_SEEDS), "seconds": round(time.time() - t0), "jobs": len(jobs)}}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "precomputed_study.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    print("fertig", len(jobs), "Läufe in", round(time.time() - t0), "s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
