"""
NFL_ENGINE — pipeline propio, sin depender de un modelo externo único.

Mismo estándar de rigor que MLB:
  - ensemble de submodelos independientes, con pesos configurables
  - shrinkage bayesiano hacia el prior (mismo principio que MLB/NCAAF)
  - Monte Carlo (>=10,000 iteraciones) para spread, moneyline y totals
  - comparación explícita contra el mercado (edge) con banderas de alerta
  - NADA se declara "listo para apostar" sin pasar los umbrales de confianza

Este módulo NO usa DAEPA ni ningún proveedor externo como fuente de verdad.
Los submodelos aquí definidos se alimentan de datos propios (o, mientras no
existan, de placeholders explícitamente marcados como NO_DATA — nunca de un
número inventado).

Versión: NFL_ENGINE_v1
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ============================================================
# CONFIGURACIÓN — PRE-ESPECIFICADA, no calibrada con resultados
# de la temporada que se está evaluando (mismo principio que NCAAF).
# ============================================================

class NFLConfig:
    """Parámetros del motor. Cambiar esto exige nueva versión, no ajuste silencioso."""
    VERSION = "NFL_ENGINE_CONFIG_v1"

    # Shrinkage: k = "partidos equivalentes" del prior. La NFL juega solo
    # 17 partidos por temporada -> muestra intrínsecamente pequeña.
    # k más alto que MLB (70) porque hay ~8x menos partidos para calibrar.
    SHRINKAGE_K: float = 8.0
    PRIOR_RATING: float = 0.0          # rating neutro en escala de puntos

    # Ventaja de local en la NFL (consenso de literatura pública: ~1.5-2.5 pts)
    HOME_ADVANTAGE_PTS: float = 2.0

    # Desviación estándar del margen final vs esperado (consenso ~13-14 pts)
    MARGIN_SD: float = 13.5

    # Monte Carlo
    MC_ITERATIONS: int = 10_000

    # Umbral de confianza mínima para NO marcar "closeCall"
    MIN_CONFIDENCE_PCT: float = 58.0

    # Tamaño de muestra mínimo (partidos jugados) antes de confiar en un
    # submodelo basado en resultados de la temporada actual
    MIN_SAMPLE_GAMES: int = 4

    # --- Ajuste por ritmo (pace) ---
    # Posesiones/juego promedio de la liga. PRE-ESPECIFICADO (consenso
    # público histórico NFL ~10.5-11 por equipo); no se recalibra con
    # los datos de la temporada que se está evaluando.
    LEAGUE_AVG_POSSESSIONS: float = 10.8

    # --- Números clave NFL ---
    # Distribución empírica pública de márgenes finales más frecuentes
    # en la NFL (fenómeno de estructura de anotación: TD+XP=7, FG=3,
    # TD+2pt=8, etc.). Pesos APROXIMADOS de literatura de apuestas
    # deportivas, PRE-ESPECIFICADOS — no ajustados a esta temporada.
    # Fuente: frecuencia histórica pública de márgenes NFL (Sharp Football,
    # Pro-Football-Reference, análisis agregados de +40 temporadas).
    KEY_NUMBERS: dict[int, float] = {
        3: 0.095, 7: 0.072, 6: 0.052, 10: 0.048, 4: 0.042, 14: 0.031, 1: 0.030,
    }
    # Peso de mezcla: cuánto se inclina la distribución simulada hacia el
    # patrón empírico de números clave. Bajo a propósito — el Monte Carlo
    # sigue siendo la fuente principal, esto solo corrige un sesgo conocido
    # de Poisson puro (que no reproduce la estructura discreta de anotación).
    KEY_NUMBER_BLEND: float = 0.12

    # --- Kelly ---
    # Edge mínimo para siquiera calcular/mostrar una fracción de Kelly.
    MIN_EDGE_FOR_KELLY_PCT: float = 2.0
    # Fracción de Kelly recomendada por defecto (conservadora). Kelly
    # completo es agresivo y sensible a errores de calibración del modelo;
    # 1/4 es el estándar defensivo en gestión de bankroll cuantitativa.
    KELLY_FRACTION_DEFAULT: float = 0.25


class Confidence(str, Enum):
    OK = "OK"
    CLOSE_CALL = "CLOSE_CALL"          # < MIN_CONFIDENCE_PCT
    INSUFFICIENT_SAMPLE = "INSUFFICIENT_SAMPLE"
    NO_DATA = "NO_DATA"                # falta el submodelo por completo


# ============================================================
# SUBMODELOS — cada uno es independiente y auditable por separado.
# Ninguno se inventa: si falta el dato, el submodelo se omite y
# se declara explícitamente, nunca se rellena con un promedio.
# ============================================================

@dataclass
class SubmodelResult:
    name: str
    prob_home_win: Optional[float]     # None si no hay dato suficiente
    weight: float
    confidence: Confidence
    sample_size: int = 0
    note: str = ""


@dataclass
class TeamInputs:
    """
    Todo lo que el motor puede recibir de un equipo. Los campos son
    Optional a propósito: si un dato no existe, el submodelo que lo usa
    queda marcado NO_DATA en vez de fabricar un valor.
    """
    name: str
    games_played: int = 0

    # Métricas de eficiencia (EPA/play). None = sin dato.
    epa_off_per_play: Optional[float] = None
    epa_def_per_play: Optional[float] = None

    # Power rating externo verificable (ej. FPI, DVOA) — se usa como UN
    # submodelo más, nunca como la única fuente, y siempre con su fuente
    # y timestamp para auditoría.
    power_rating: Optional[float] = None
    power_rating_source: Optional[str] = None

    # Situacional: descanso (días), viaje (millas), lesiones clave (conteo)
    rest_days: Optional[int] = None
    travel_miles: Optional[float] = None
    key_injuries: int = 0

    # Ritmo: posesiones por partido. None = se usa el promedio de liga
    # (LEAGUE_AVG_POSSESSIONS) para AMBOS equipos, que equivale a no
    # ajustar nada — nunca se inventa un ritmo específico del equipo.
    possessions_per_game: Optional[float] = None

    # --- Prior de temporada anterior (opcional) ---
    # Mejora el PUNTO DE PARTIDA del shrinkage cuando hay pocos partidos
    # de la temporada actual. NO cuenta como partidos jugados: la NFL
    # tiene alta rotación año-a-año (QB nuevo, coordinador nuevo, roster
    # turnover), así que un buen prior no reemplaza evidencia de ESTA
    # temporada — solo la hace arrancar desde un lugar más informado que
    # el neutro (0.0) mientras games_played sigue siendo bajo.
    prior_epa_off_per_play: Optional[float] = None
    prior_epa_def_per_play: Optional[float] = None
    prior_season_label: Optional[str] = None  # ej. "2025 EPA/play, temporada completa"


def submodel_epa(home: TeamInputs, away: TeamInputs, cfg: NFLConfig) -> SubmodelResult:
    """
    Submodelo de eficiencia: diferencial de EPA/play ofensivo-defensivo,
    con shrinkage hacia 0 según el tamaño de muestra de la temporada.
    """
    def has_usable_data(t: TeamInputs) -> bool:
        # Usable si hay EPA de esta temporada, O si hay prior declarado
        # (con 0 partidos actuales, el peso del prior será 100% de todos
        # modos — ver net_epa más abajo).
        has_current = t.epa_off_per_play is not None
        has_prior = (t.prior_epa_off_per_play is not None
                    and t.prior_epa_def_per_play is not None)
        return has_current or has_prior

    if not has_usable_data(home) or not has_usable_data(away):
        return SubmodelResult("epa_efficiency", None, weight=0.35,
                              confidence=Confidence.NO_DATA,
                              note="Falta EPA/play de esta temporada Y prior de al menos un equipo")

    n = min(home.games_played, away.games_played)
    if n < cfg.MIN_SAMPLE_GAMES:
        conf = Confidence.INSUFFICIENT_SAMPLE
    else:
        conf = Confidence.OK

    def net_epa(t: TeamInputs) -> float:
        off = t.epa_off_per_play or 0.0
        deff = t.epa_def_per_play or 0.0
        raw = off - deff
        # El prior de shrinkage es el rating de temporada anterior si se
        # declaró explícitamente; si no, el neutro (0.0) de siempre.
        # Esto SOLO cambia el punto de partida — el peso n/(n+k) sigue
        # dependiendo de partidos REALES de esta temporada, así que con
        # pocos partidos el resultado se parece más al prior, y conforme
        # avanza la temporada la evidencia nueva lo va desplazando, igual
        # que el shrinkage de MLB/NCAAF.
        if t.prior_epa_off_per_play is not None and t.prior_epa_def_per_play is not None:
            base = t.prior_epa_off_per_play - t.prior_epa_def_per_play
        else:
            base = cfg.PRIOR_RATING
        w = n / (n + cfg.SHRINKAGE_K)
        return base + (raw - base) * w

    diff = net_epa(home) - net_epa(away)
    # conversión heurística EPA-diff -> probabilidad (logística), pendiente
    # calibrable pero PRE-ESPECIFICADA, no ajustada a resultados de hoy.
    prob = 1.0 / (1.0 + math.exp(-diff * 8.0))

    return SubmodelResult("epa_efficiency", prob, weight=0.35, confidence=conf,
                          sample_size=n)


def submodel_power_rating(home: TeamInputs, away: TeamInputs,
                           cfg: NFLConfig) -> SubmodelResult:
    """
    Submodelo de rating externo (ej. FPI/DVOA), tratado como UN input más
    del ensemble — nunca como la respuesta única del sistema.
    Requiere que la fuente esté explícitamente declarada.
    """
    if home.power_rating is None or away.power_rating is None:
        return SubmodelResult("power_rating", None, weight=0.25,
                              confidence=Confidence.NO_DATA,
                              note="Sin rating externo declarado para ambos equipos")
    if not (home.power_rating_source and away.power_rating_source):
        return SubmodelResult("power_rating", None, weight=0.25,
                              confidence=Confidence.NO_DATA,
                              note="Rating presente pero SIN fuente declarada — se rechaza por auditoría")

    diff_pts = (home.power_rating - away.power_rating) + cfg.HOME_ADVANTAGE_PTS
    prob = 1.0 / (1.0 + math.exp(-diff_pts / 7.0))
    return SubmodelResult("power_rating", prob, weight=0.25, confidence=Confidence.OK)


def submodel_situational(home: TeamInputs, away: TeamInputs,
                          cfg: NFLConfig) -> SubmodelResult:
    """
    Submodelo situacional: descanso, viaje, lesiones clave.
    Efecto pequeño y acotado a propósito (evita que una sola variable
    blanda domine el ensemble).
    """
    if home.rest_days is None and away.rest_days is None and \
       home.key_injuries == 0 and away.key_injuries == 0:
        return SubmodelResult("situational", None, weight=0.15,
                              confidence=Confidence.NO_DATA,
                              note="Sin datos situacionales cargados")

    adj = 0.0
    if home.rest_days is not None and away.rest_days is not None:
        adj += max(-3, min(3, (home.rest_days - away.rest_days))) * 0.15
    adj -= home.key_injuries * 0.4
    adj += away.key_injuries * 0.4

    prob = 1.0 / (1.0 + math.exp(-adj / cfg.MARGIN_SD))
    return SubmodelResult("situational", prob, weight=0.15, confidence=Confidence.OK)


def submodel_market_implied(market_home_odds: Optional[float],
                             market_away_odds: Optional[float]) -> SubmodelResult:
    """
    El propio mercado como submodelo de referencia SOLO para contraste
    de calibración — su peso es bajo a propósito: no queremos que el
    ensemble termine reproduciendo la línea (eso sería edge=0 por diseño,
    el mismo error que evitamos en NCAAF).
    """
    if market_home_odds is None or market_away_odds is None:
        return SubmodelResult("market_reference", None, weight=0.0,
                              confidence=Confidence.NO_DATA,
                              note="Sin cuotas de mercado (no participa en el ensemble)")
    imp_home = 1.0 / market_home_odds
    imp_away = 1.0 / market_away_odds
    total = imp_home + imp_away
    return SubmodelResult("market_reference", imp_home / total, weight=0.0,
                          confidence=Confidence.OK,
                          note="Peso 0.0 deliberado: solo referencia, nunca insumo del ensemble")


# ============================================================
# ENSEMBLE — combina submodelos con pesos configurables.
# Pesos normalizados SOLO entre submodelos con dato real disponible.
# ============================================================

@dataclass
class EnsembleResult:
    prob_home_win: float
    submodels_used: list[SubmodelResult]
    submodels_excluded: list[SubmodelResult]
    confidence_flag: Confidence
    warnings: list[str] = field(default_factory=list)


def run_ensemble(home: TeamInputs, away: TeamInputs,
                  cfg: NFLConfig = NFLConfig(),
                  market_home_odds: Optional[float] = None,
                  market_away_odds: Optional[float] = None) -> EnsembleResult:
    """
    Corre los submodelos, excluye los NO_DATA, renormaliza pesos,
    y devuelve la probabilidad combinada + las banderas de advertencia.
    """
    candidates = [
        submodel_epa(home, away, cfg),
        submodel_power_rating(home, away, cfg),
        submodel_situational(home, away, cfg),
    ]
    market_ref = submodel_market_implied(market_home_odds, market_away_odds)

    used = [s for s in candidates if s.prob_home_win is not None]
    excluded = [s for s in candidates if s.prob_home_win is None]

    warnings: list[str] = []

    if not used:
        warnings.append("SIN NINGÚN SUBMODELO CON DATOS: no se puede generar probabilidad propia.")
        return EnsembleResult(prob_home_win=float("nan"), submodels_used=[],
                              submodels_excluded=candidates,
                              confidence_flag=Confidence.NO_DATA,
                              warnings=warnings)

    total_w = sum(s.weight for s in used)
    prob = sum(s.prob_home_win * (s.weight / total_w) for s in used)

    # Bandera de muestra insuficiente si algún submodelo usado tiene poca muestra
    insuf = [s for s in used if s.confidence == Confidence.INSUFFICIENT_SAMPLE]
    if insuf:
        warnings.append(
            f"Muestra insuficiente en: {', '.join(s.name for s in insuf)} "
            f"(< {cfg.MIN_SAMPLE_GAMES} partidos)"
        )

    if excluded:
        warnings.append(
            "Submodelos excluidos por falta de datos: "
            + ", ".join(f"{s.name} ({s.note})" for s in excluded)
        )

    # Referencia de mercado, solo para contraste — nunca entra al promedio
    if market_ref.prob_home_win is not None:
        gap = abs(prob - market_ref.prob_home_win) * 100
        if gap > 15:
            warnings.append(
                f"El ensemble propio difiere {gap:.1f} pts del mercado — "
                f"revisar antes de confiar en cualquiera de los dos"
            )

    conf_pct = max(prob, 1 - prob) * 100
    conf_flag = Confidence.CLOSE_CALL if conf_pct < cfg.MIN_CONFIDENCE_PCT else Confidence.OK
    if conf_flag == Confidence.CLOSE_CALL:
        warnings.append(f"Confianza {conf_pct:.1f}% por debajo del umbral ({cfg.MIN_CONFIDENCE_PCT}%)")

    return EnsembleResult(prob_home_win=prob, submodels_used=used,
                          submodels_excluded=excluded, confidence_flag=conf_flag,
                          warnings=warnings)


# ============================================================
# MONTE CARLO — distribución real de resultados, no solo un punto.
# Simula puntuación de ambos equipos vía Poisson (aprox. razonable
# para totales de puntos en NFL), a partir del margen esperado y el
# total implícito por el ensemble.
# ============================================================

@dataclass
class MonteCarloResult:
    iterations: int
    prob_home_win: float
    prob_away_win: float
    prob_tie: float
    mean_home_score: float
    mean_away_score: float
    spread_distribution: dict[str, float]   # e.g. {"-3.5": 0.62, ...}
    total_distribution: dict[str, float]    # e.g. {"o44.5": 0.51, ...}


def _poisson_sample(rng: random.Random, lam: float) -> int:
    """Poisson vía Knuth — suficiente para este propósito, sin dependencias externas."""
    L = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        k += 1
        p *= rng.random()
        if p <= L:
            return k - 1


def _pace_multiplier(home: Optional[TeamInputs], away: Optional[TeamInputs],
                      cfg: NFLConfig) -> float:
    """
    Ajuste por ritmo: si NINGÚN equipo trae posesiones/juego declaradas,
    el multiplicador es exactamente 1.0 (sin efecto) — nunca se inventa
    ritmo específico. Si al menos uno trae dato real, se ajusta la media
    de Poisson proporcional a las posesiones combinadas vs el promedio
    de liga (más posesiones -> más oportunidades de anotar -> lambda sube).
    """
    hp = home.possessions_per_game if home else None
    ap = away.possessions_per_game if away else None
    if hp is None and ap is None:
        return 1.0
    hp = hp if hp is not None else cfg.LEAGUE_AVG_POSSESSIONS
    ap = ap if ap is not None else cfg.LEAGUE_AVG_POSSESSIONS
    combined = (hp + ap) / 2
    return combined / cfg.LEAGUE_AVG_POSSESSIONS


def _blend_key_numbers(spread_covers_raw: dict[str, int], n: int,
                        spread_lines: list[float], cfg: NFLConfig) -> dict[str, float]:
    """
    Corrige un sesgo conocido de Poisson puro: no reproduce la estructura
    discreta de anotación de la NFL (los márgenes de 3, 7, 6, 10, 4 son
    más frecuentes de lo que un Poisson continuo predice, por la propia
    aritmética del deporte: FG=3, TD+XP=7, etc.).

    Se mezcla la probabilidad simulada con un ajuste basado en si la línea
    cae exactamente sobre o cerca de un número clave. KEY_NUMBER_BLEND
    controla cuánto peso tiene esta corrección (bajo, deliberadamente).
    """
    out: dict[str, float] = {}
    for s in spread_lines:
        raw_p = spread_covers_raw[f"{s:+.1f}"] / n
        # línea entera más cercana a la magnitud del spread
        nearest_key = round(abs(s))
        key_weight = cfg.KEY_NUMBERS.get(nearest_key, 0.0)
        # si la línea cae justo en un número clave (ej. -3.0, -7.0), el
        # efecto de "cruzarlo" es más determinante -> mayor corrección
        is_exact = abs(abs(s) - nearest_key) < 1e-9
        blend = cfg.KEY_NUMBER_BLEND * (1.5 if is_exact else 1.0)
        # la corrección empuja levemente hacia 0.5 cuando la línea está
        # justo sobre un número clave con alta densidad histórica (más
        # incertidumbre real de la que Poisson puro sugiere)
        adjusted = raw_p * (1 - blend * key_weight * 10) + 0.5 * (blend * key_weight * 10)
        out[f"{s:+.1f}"] = max(0.0, min(1.0, adjusted))
    return out


def run_monte_carlo(prob_home_win: float,
                     expected_total_points: float = 44.0,
                     spread_lines: Optional[list[float]] = None,
                     total_lines: Optional[list[float]] = None,
                     cfg: NFLConfig = NFLConfig(),
                     seed: Optional[int] = None,
                     home_inputs: Optional[TeamInputs] = None,
                     away_inputs: Optional[TeamInputs] = None) -> MonteCarloResult:
    """
    A partir de la probabilidad de victoria del ensemble, deriva lambdas
    de anotación consistentes (vía margen esperado implícito por logit)
    y simula >=10,000 partidos completos.

    No usa una proyección puntual fija: cada iteración es un resultado
    completo, y de ahí se derivan TODAS las probabilidades (ganador,
    cobertura de spread, total).

    Si se pasan home_inputs/away_inputs con possessions_per_game, se
    aplica el ajuste de ritmo (pace) a la media de Poisson. Sin ese
    dato, el ajuste es neutro (multiplicador 1.0) — nunca se asume ritmo.
    """
    rng = random.Random(seed)
    spread_lines = spread_lines or [-2.5, -3.5, -6.5, -7.5]
    total_lines = total_lines or [42.5, 44.5, 47.5]

    # Margen esperado implícito por la probabilidad (inversión de la logística
    # usada en el ensemble), consistente con MARGIN_SD.
    p = min(max(prob_home_win, 1e-6), 1 - 1e-6)
    expected_margin = -cfg.MARGIN_SD * math.log((1 - p) / p) / 1.7  # aprox logit->normal

    pace_mult = _pace_multiplier(home_inputs, away_inputs, cfg)
    adj_total = expected_total_points * pace_mult

    lam_home = max(3.0, adj_total / 2 + expected_margin / 2)
    lam_away = max(3.0, adj_total / 2 - expected_margin / 2)

    n = cfg.MC_ITERATIONS
    home_wins = away_wins = ties = 0
    home_scores: list[int] = []
    away_scores: list[int] = []
    spread_covers = {f"{s:+.1f}": 0 for s in spread_lines}
    total_overs = {f"o{t:.1f}": 0 for t in total_lines}

    for _ in range(n):
        hs = _poisson_sample(rng, lam_home)
        aws = _poisson_sample(rng, lam_away)
        home_scores.append(hs)
        away_scores.append(aws)

        if hs > aws:
            home_wins += 1
        elif aws > hs:
            away_wins += 1
        else:
            ties += 1

        margin = hs - aws
        for s in spread_lines:
            if margin > -s:
                spread_covers[f"{s:+.1f}"] += 1

        total_pts = hs + aws
        for t in total_lines:
            if total_pts > t:
                total_overs[f"o{t:.1f}"] += 1

    return MonteCarloResult(
        iterations=n,
        prob_home_win=home_wins / n,
        prob_away_win=away_wins / n,
        prob_tie=ties / n,
        mean_home_score=statistics.mean(home_scores),
        mean_away_score=statistics.mean(away_scores),
        spread_distribution=_blend_key_numbers(spread_covers, n, spread_lines, cfg),
        total_distribution={k: v / n for k, v in total_overs.items()},
    )


# ============================================================
# COMPARACIÓN DE MERCADO Y EDGE
# ============================================================

@dataclass
class EdgeResult:
    our_prob: float
    market_implied_prob: Optional[float]
    edge_pct: Optional[float]           # our_prob - market_implied, en puntos %
    ev_per_unit: Optional[float]        # EV esperado apostando 1 unidad
    kelly_full: Optional[float] = None       # fracción de Kelly completa (0-1)
    kelly_recommended: Optional[float] = None  # KELLY_FRACTION_DEFAULT * kelly_full
    flags: list[str] = field(default_factory=list)   # RECHAZO — sí bloquean ready_to_bet
    info: list[str] = field(default_factory=list)    # INFORMATIVO — nunca bloquea


def compute_kelly_fraction(our_prob: float, market_odds_decimal: float) -> float:
    """
    Fracción de Kelly: f* = (b*p - q) / b
      b = ganancia neta por unidad apostada (odds decimales - 1)
      p = nuestra probabilidad de ganar
      q = 1 - p
    Se trunca a 0 si el resultado es negativo (no apostar) y a 1 como
    tope defensivo (nunca recomendar apostar más del 100% del bankroll,
    aunque matemáticamente Kelly pudiera exceder eso en casos extremos).

    Esto es información de gestión de bankroll, no asesoría financiera
    personalizada: el usuario decide su propio tamaño de apuesta.
    """
    b = market_odds_decimal - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - our_prob
    f = (b * our_prob - q) / b
    return max(0.0, min(1.0, f))


def compute_edge(our_prob: float, market_odds_decimal: Optional[float],
                  cfg: NFLConfig = NFLConfig(),
                  sample_size: int = 0) -> EdgeResult:
    """
    Compara nuestra probabilidad (del Monte Carlo, no del ensemble puntual)
    contra la cuota de mercado. Marca banderas si la muestra o la confianza
    no alcanzan el umbral. Si el edge supera MIN_EDGE_FOR_KELLY_PCT, calcula
    también la fracción de Kelly (completa y la recomendada/conservadora).
    """
    flags: list[str] = []
    conf_pct = max(our_prob, 1 - our_prob) * 100

    if conf_pct < cfg.MIN_CONFIDENCE_PCT:
        flags.append(f"CONFIANZA INSUFICIENTE: {conf_pct:.1f}% < {cfg.MIN_CONFIDENCE_PCT}%")

    if sample_size < cfg.MIN_SAMPLE_GAMES:
        flags.append(f"MUESTRA INSUFICIENTE: {sample_size} partidos < {cfg.MIN_SAMPLE_GAMES}")

    if market_odds_decimal is None:
        flags.append("SIN CUOTA DE MERCADO: edge no calculable")
        return EdgeResult(our_prob=our_prob, market_implied_prob=None,
                          edge_pct=None, ev_per_unit=None, flags=flags)

    implied = 1.0 / market_odds_decimal
    edge = (our_prob - implied) * 100
    ev = our_prob * (market_odds_decimal - 1) - (1 - our_prob)

    if edge < 0:
        flags.append(f"EDGE NEGATIVO ({edge:.1f} pts): el mercado ve más probable lo contrario que nosotros")

    info: list[str] = []
    kelly_full = kelly_rec = None
    if edge >= cfg.MIN_EDGE_FOR_KELLY_PCT:
        kelly_full = compute_kelly_fraction(our_prob, market_odds_decimal)
        kelly_rec = kelly_full * cfg.KELLY_FRACTION_DEFAULT
        info.append(
            f"Kelly informativo: completo {kelly_full*100:.1f}% del bankroll, "
            f"recomendado (1/4 Kelly) {kelly_rec*100:.1f}%. No es asesoría "
            f"financiera personalizada — gestión de riesgo bajo tu propio criterio."
        )

    return EdgeResult(our_prob=our_prob, market_implied_prob=implied,
                      edge_pct=edge, ev_per_unit=ev,
                      kelly_full=kelly_full, kelly_recommended=kelly_rec,
                      flags=flags, info=info)


# ============================================================
# PIPELINE COMPLETO — orquesta todo lo anterior en una sola llamada,
# devolviendo un resultado auditable de punta a punta.
# ============================================================

@dataclass
class GameAnalysis:
    home: str
    away: str
    ensemble: EnsembleResult
    monte_carlo: Optional[MonteCarloResult]
    edge: Optional[EdgeResult]
    ready_to_bet: bool
    rejection_reasons: list[str]
    market_spread: Optional[float] = None   # línea del LOCAL declarada, si la hubo

    @property
    def home_cover_prob(self) -> Optional[float]:
        """% de las 10,000 simulaciones donde el LOCAL cubre market_spread."""
        if self.monte_carlo is None or self.market_spread is None:
            return None
        key = f"{self.market_spread:+.1f}"
        p = self.monte_carlo.spread_distribution.get(key)
        return round(p * 100, 2) if p is not None else None

    @property
    def away_cover_prob(self) -> Optional[float]:
        """Complemento del anterior (push tratado aparte, no se resta de 100)."""
        hp = self.home_cover_prob
        return round(100 - hp, 2) if hp is not None else None

    @property
    def spread_favorite(self) -> Optional[str]:
        """
        Quién es favorito EN LA LÍNEA (no en moneyline): local si
        market_spread < 0, visitante si > 0. Con línea 0 (pick'em) no
        hay favorito de spread — se declara explícitamente como tal,
        no se asigna arbitrariamente a uno de los dos.
        """
        if self.market_spread is None:
            return None
        if self.market_spread < 0:
            return self.home
        if self.market_spread > 0:
            return self.away
        return None  # pick'em — declarar, no adivinar


def analyze_game(home: TeamInputs, away: TeamInputs,
                  cfg: NFLConfig = NFLConfig(),
                  market_home_odds: Optional[float] = None,
                  market_away_odds: Optional[float] = None,
                  market_spread: Optional[float] = None,
                  expected_total_points: float = 44.0) -> GameAnalysis:
    """
    Punto de entrada único. Un partido entra, un análisis completo sale,
    con criterio explícito de por qué SÍ o por qué NO se recomienda apostar.

    market_spread: línea de puntos del LOCAL (convención ya usada en todo
    el módulo: negativo = local favorito, ej. -3.5). Si se declara, el
    Monte Carlo mide la cobertura EXACTA de esa línea (además de las 4
    líneas genéricas de referencia) y el resultado expone home_cover_prob
    / away_cover_prob / spread_favorite ya resueltos — sin que el llamador
    tenga que adivinar el signo ni parsear spread_distribution a mano.

    "ready_to_bet" es deliberadamente conservador: por defecto False.
    """
    ens = run_ensemble(home, away, cfg, market_home_odds, market_away_odds)
    rejection: list[str] = []

    if ens.confidence_flag == Confidence.NO_DATA:
        rejection.append("Sin datos suficientes para generar ensemble propio")
        return GameAnalysis(home=home.name, away=away.name, ensemble=ens,
                            monte_carlo=None, edge=None, ready_to_bet=False,
                            rejection_reasons=rejection, market_spread=market_spread)

    # Si se declaró la línea real del partido, se incluye en spread_lines
    # (junto con las 4 genéricas de referencia) para que el Monte Carlo
    # calcule su cobertura exacta con el MISMO mecanismo ya validado
    # (incluida la corrección de números clave), no uno paralelo.
    default_lines = [-2.5, -3.5, -6.5, -7.5]
    lines = list(default_lines)
    if market_spread is not None and market_spread not in lines:
        lines.append(market_spread)

    mc = run_monte_carlo(ens.prob_home_win, expected_total_points, cfg=cfg,
                         spread_lines=lines,
                         seed=hash((home.name, away.name)) & 0xFFFFFFFF,
                         home_inputs=home, away_inputs=away)

    market_odds = market_home_odds  # simplificación: análisis desde la óptica local
    n_sample = min((s.sample_size for s in ens.submodels_used if s.sample_size), default=0)
    edge = compute_edge(mc.prob_home_win, market_odds, cfg, sample_size=n_sample)

    if ens.confidence_flag == Confidence.CLOSE_CALL:
        rejection.append("Confianza del ensemble por debajo del umbral mínimo")
    if edge.flags:
        rejection.extend(edge.flags)
    if len(ens.submodels_used) < 2:
        rejection.append(
            f"Solo {len(ens.submodels_used)} submodelo(s) con datos — "
            f"ensemble poco robusto, no se recomienda confiar en solitario"
        )

    ready = len(rejection) == 0

    return GameAnalysis(home=home.name, away=away.name, ensemble=ens,
                        monte_carlo=mc, edge=edge, ready_to_bet=ready,
                        rejection_reasons=rejection, market_spread=market_spread)


# ============================================================
# UTILIDAD DE REPORTE — texto plano para consola/logs, auditable.
# ============================================================

def format_report(analysis: GameAnalysis) -> str:
    lines = [
        f"=== {analysis.away} @ {analysis.home} ===",
        f"Submodelos usados: {[s.name for s in analysis.ensemble.submodels_used]}",
        f"Submodelos excluidos (sin dato): {[s.name for s in analysis.ensemble.submodels_excluded]}",
    ]
    if analysis.monte_carlo:
        mc = analysis.monte_carlo
        lines.append(
            f"Monte Carlo ({mc.iterations:,} iter): "
            f"P(local)={mc.prob_home_win*100:.1f}% "
            f"P(visita)={mc.prob_away_win*100:.1f}% "
            f"P(empate)={mc.prob_tie*100:.1f}%"
        )
        lines.append(f"Marcador esperado: {mc.mean_home_score:.1f} - {mc.mean_away_score:.1f}")
    if analysis.edge:
        e = analysis.edge
        edge_txt = f"{e.edge_pct:+.1f} pts" if e.edge_pct is not None else "N/A"
        lines.append(f"Edge vs mercado: {edge_txt}")
        for i in e.info:
            lines.append(f"  ℹ {i}")
    for w in analysis.ensemble.warnings:
        lines.append(f"  ⚠ {w}")
    lines.append(f"LISTO PARA APOSTAR: {'SÍ' if analysis.ready_to_bet else 'NO'}")
    for r in analysis.rejection_reasons:
        lines.append(f"  ✗ {r}")
    return "\n".join(lines)
