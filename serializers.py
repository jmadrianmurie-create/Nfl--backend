"""
serializers.py — conversión de resultados del motor NFL a JSON.

Por qué existe este archivo aparte:
  GameAnalysis / EnsembleResult / MonteCarloResult / EdgeResult son
  dataclasses anidados, y SubmodelResult.confidence es un Enum
  (Confidence). Ninguno de los dos es JSON-serializable por defecto
  en Flask/json estándar -> hay que aplanar explícitamente.

No se usa NumPy en nfl_engine.py (solo stdlib), así que no hay
arrays ndarray que convertir; el único problema real es el Enum.
"""
from __future__ import annotations
from dataclasses import asdict
from typing import Any

from nfl_engine import GameAnalysis, SubmodelResult


def serialize_submodel(s: SubmodelResult) -> dict[str, Any]:
    d = asdict(s)
    d["confidence"] = s.confidence.value  # Enum -> str plano
    return d


def serialize_analysis(a: GameAnalysis) -> dict[str, Any]:
    out: dict[str, Any] = {
        "home": a.home,
        "away": a.away,
        "ready_to_bet": a.ready_to_bet,
        "rejection_reasons": a.rejection_reasons,
        "ensemble": {
            "prob_home_win": a.ensemble.prob_home_win,
            "confidence_flag": a.ensemble.confidence_flag.value,
            "submodels_used": [serialize_submodel(s) for s in a.ensemble.submodels_used],
            "submodels_excluded": [serialize_submodel(s) for s in a.ensemble.submodels_excluded],
            "warnings": a.ensemble.warnings,
        },
        "monte_carlo": None,
        "edge": None,
    }
    if a.monte_carlo:
        mc = a.monte_carlo
        out["monte_carlo"] = {
            "iterations": mc.iterations,
            "prob_home_win": mc.prob_home_win,
            "prob_away_win": mc.prob_away_win,
            "prob_tie": mc.prob_tie,
            "mean_home_score": mc.mean_home_score,
            "mean_away_score": mc.mean_away_score,
            "spread_distribution": mc.spread_distribution,
            "total_distribution": mc.total_distribution,
        }
    if a.edge:
        e = a.edge
        out["edge"] = {
            "our_prob": e.our_prob,
            "market_implied_prob": e.market_implied_prob,
            "edge_pct": e.edge_pct,
            "ev_per_unit": e.ev_per_unit,
            "kelly_full": e.kelly_full,
            "kelly_recommended": e.kelly_recommended,
            "flags": e.flags,   # motivos de RECHAZO
            "info": e.info,     # informativo (ej. Kelly) — nunca bloquea
        }

    # Bloque de conveniencia: moneyline (ganador) y spread (cobertura de
    # línea) como DOS métricas numéricas separadas, explícitas. Números
    # reales (float), NUNCA strings con "%" — el consumidor decide el
    # formato de presentación, el backend nunca lo fija de antemano.
    # No reemplaza los campos planos de arriba (ensemble/monte_carlo/edge):
    # es una vista adicional sobre los mismos datos, para no romper nada
    # que ya consuma el shape existente.
    moneyline_favorite = None
    win_probability = None
    if a.ensemble.prob_home_win is not None and a.ensemble.prob_home_win == a.ensemble.prob_home_win:  # not NaN
        is_home_fav = a.ensemble.prob_home_win > 0.5
        moneyline_favorite = a.home if is_home_fav else a.away
        win_probability = round((a.ensemble.prob_home_win if is_home_fav
                                 else 1 - a.ensemble.prob_home_win) * 100, 2)

    spread_favorite = a.spread_favorite      # None si no hay market_spread o es pick'em
    home_cover = a.home_cover_prob           # None si no hay market_spread o no hubo Monte Carlo
    away_cover = a.away_cover_prob
    cover_probability = None
    if spread_favorite is not None:
        cover_probability = home_cover if spread_favorite == a.home else away_cover

    out["predictions"] = {
        "moneyline": {
            "favorite": moneyline_favorite,
            "win_probability": win_probability,   # ej. 62.5  (NO "62.5%")
        },
        "spread": {
            "line": a.market_spread,              # None si no se declaró línea
            "cover_favorite": spread_favorite,     # None si pick'em o sin línea
            "cover_probability": cover_probability,  # del FAVORITO de spread, no siempre del local
            "home_cover_prob": home_cover,
            "away_cover_prob": away_cover,
        },
    }
    return out
