"""Candidate-versus-production comparison over named frozen suites."""
from __future__ import annotations
import numpy as np

def compare_predictions(candidate, production, truth, tie_tolerance=.05):
    candidate=np.asarray(candidate,dtype=float); production=np.asarray(production,dtype=float); truth=np.asarray(truth,dtype=float)
    c=np.abs(candidate-truth); p=np.abs(production-truth); delta=float(np.mean(p)-np.mean(c))
    verdict="statistical_tie" if abs(delta)<=tie_tolerance else ("candidate_better" if delta>0 else "production_better")
    def metrics(error): return {"mean_x_error":float(np.mean(error)),"median_x_error":float(np.median(error)),"p95_x_error":float(np.percentile(error,95)),"maximum_x_error":float(np.max(error))}
    return {"candidate":metrics(c),"production":metrics(p),"mean_improvement":delta,"verdict":verdict}

def suite_report(cases):
    suites={}
    for case in cases:
        result=compare_predictions(case["candidate"],case["production"],case["truth"])
        for suite in case.get("suites",["general"]): suites.setdefault(suite,[]).append(result)
    return {name:{"case_count":len(rows),"candidate_better":sum(r["verdict"]=="candidate_better" for r in rows),"production_better":sum(r["verdict"]=="production_better" for r in rows),"mean_improvement":float(np.mean([r["mean_improvement"] for r in rows]))} for name,rows in suites.items()}
