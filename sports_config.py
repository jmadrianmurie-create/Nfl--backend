"""
REGISTRO DE DEPORTES — capa específica por deporte sobre un núcleo común.

Lo que COMPARTEN todos: motor de predicción, filtros de calidad,
clasificación BET/SMALL BET/NO BET, almacenamiento, captura de cuotas,
CLV, resultados, histórico, control de cuota, proveedores, versionado.

Lo que es PROPIO de cada uno: mercados disponibles, ventaja de local,
escala del logístico, constante de shrinkage, base rate, umbrales y
si admite empate.

Activar o desactivar un deporte es cambiar `enabled`. El núcleo no se toca.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict


# --------------------------------------------------------------- mercados
@dataclass(frozen=True)
class Market:
    key: str
    label: str
    n_outcomes: int              # 2 = binario · 3 = con empate
    needs_line: bool = False     # spread/handicap/total llevan línea
    enabled: bool = True


# Catálogo. Cada deporte elige los suyos; no se fuerzan los mismos a todos.
M_MONEYLINE = Market("moneyline", "Moneyline", 2)
M_1X2       = Market("1x2",       "1X2",       3)
M_RUNLINE   = Market("run_line",  "Run line",  2, needs_line=True)
M_SPREAD    = Market("spread",    "Spread",    2, needs_line=True)
M_HANDICAP  = Market("handicap",  "Hándicap asiático", 2, needs_line=True)
M_TOTALS    = Market("totals",    "Totales O/U", 2, needs_line=True)
# Preparados, desactivados hasta que haya datos gratis que los sostengan
M_BTTS      = Market("btts",      "Ambos anotan", 2, enabled=False)
M_PROPS     = Market("player_props", "Props de jugador", 2, needs_line=True, enabled=False)


@dataclass
class LeagueConfig:
    """Una competición concreta. Fútbol tiene varias; los demás una."""
    key: str
    label: str
    enabled: bool = True
    games_played: int = 0            # muestra acumulada de la temporada
    data_quality: str = "estimado"   # confirmado | parcial | estimado
    provider_key: str = ""           # id de la liga en la API de cuotas


@dataclass
class SportConfig:
    key: str
    label: str
    icon: str
    enabled: bool

    markets: list[Market]
    leagues: list[LeagueConfig]

    # Parámetros del modelo, propios de cada deporte
    home_advantage: float
    scale: float
    has_draws: bool
    shrinkage_k: float               # partidos equivalentes del prior
    base_rate_home: float            # % histórico de victoria local
    no_pick_threshold: float
    draw_peak: float = 0.0
    draw_width: float = 22.0

    # Qué modelos del ensemble aplican. No todos sirven en todos los deportes.
    models: list[str] = field(default_factory=lambda: ["strength"])

    notes: str = ""

    def market(self, key: str) -> Market | None:
        for m in self.markets:
            if m.key == key:
                return m
        return None

    def supports(self, market_key: str) -> bool:
        m = self.market(market_key)
        return bool(m and m.enabled)

    def league(self, key: str) -> LeagueConfig | None:
        for lg in self.leagues:
            if lg.key == key or lg.label == key:
                return lg
        return None

    def active_leagues(self) -> list[LeagueConfig]:
        return [lg for lg in self.leagues if lg.enabled]


# ------------------------------------------------------------- el registro
SPORTS: dict[str, SportConfig] = {

    "mlb": SportConfig(
        key="mlb", label="MLB", icon="⚾", enabled=True,
        markets=[M_MONEYLINE, M_RUNLINE, M_TOTALS, M_PROPS],
        leagues=[LeagueConfig("mlb", "MLB", True, 139, "parcial", "baseball_mlb")],
        home_advantage=1.5, scale=65, has_draws=False,
        shrinkage_k=70, base_rate_home=0.540, no_pick_threshold=0.55,
        models=["record_log5", "pythag", "strength", "monte_carlo"],
        notes="Pitagórico con exponente 1.83. Monte Carlo Poisson sobre carreras. "
              "Falta el abridor del día: es la variable de mayor impacto no cubierta."),

    "nba": SportConfig(
        key="nba", label="NBA", icon="🏀", enabled=True,
        markets=[M_MONEYLINE, M_SPREAD, M_TOTALS, M_PROPS],
        leagues=[LeagueConfig("nba", "NBA", True, 0, "estimado", "basketball_nba")],
        home_advantage=2.5, scale=60, has_draws=False,
        shrinkage_k=25, base_rate_home=0.585, no_pick_threshold=0.55,
        models=["strength"],
        notes="Temporada sin arrancar: 0 partidos. Bloqueado para picks hasta que "
              "haya juegos. Pitagórico de puntos y Monte Carlo requieren datos reales."),

    "nfl": SportConfig(
        key="nfl", label="NFL", icon="🏈", enabled=True,
        markets=[M_MONEYLINE, M_SPREAD, M_TOTALS, M_PROPS],
        leagues=[LeagueConfig("nfl", "NFL", True, 0, "estimado", "americanfootball_nfl")],
        home_advantage=2.0, scale=45, has_draws=False,
        shrinkage_k=6, base_rate_home=0.560, no_pick_threshold=0.55,
        models=["strength"],
        notes="Muestra muy corta por diseño (17 juegos). k bajo. Sin partidos aún."),

    "soccer": SportConfig(
        key="soccer", label="Fútbol", icon="⚽", enabled=True,
        markets=[M_1X2, M_HANDICAP, M_TOTALS, M_BTTS],
        leagues=[
            LeagueConfig("liga_mx",  "Liga MX",         True,  6, "parcial",  "soccer_mexico_ligamx"),
            LeagueConfig("la_liga",  "La Liga",         True,  3, "estimado", "soccer_spain_la_liga"),
            LeagueConfig("epl",      "Premier League",  False, 0, "estimado", "soccer_epl"),
            LeagueConfig("serie_a",  "Serie A",         False, 0, "estimado", "soccer_italy_serie_a"),
            LeagueConfig("bundesliga","Bundesliga",     False, 0, "estimado", "soccer_germany_bundesliga"),
            LeagueConfig("ligue_1",  "Ligue 1",         False, 0, "estimado", "soccer_france_ligue_one"),
            LeagueConfig("ucl",      "Champions League",False, 0, "estimado", "soccer_uefa_champs_league"),
        ],
        home_advantage=5.0, scale=40, has_draws=True,
        shrinkage_k=10, base_rate_home=0.455, no_pick_threshold=0.55,
        draw_peak=0.30, draw_width=22,
        models=["strength"],
        notes="Único deporte con empate: 1X2 tiene 3 resultados. Varias ligas bajo "
              "la misma configuración; activar más es cambiar enabled. "
              "k=10 porque las temporadas son cortas."),
}


# -------------------------------------------------------------- utilidades
def enabled_sports() -> list[SportConfig]:
    return [s for s in SPORTS.values() if s.enabled]


# Nombres genéricos que usan las APIs de cuotas y los clientes existentes.
ALIASES = {
    "baseball": "mlb", "mlb": "mlb",
    "basketball": "nba", "nba": "nba",
    "americanfootball": "nfl", "american_football": "nfl",
    "football": "nfl", "nfl": "nfl",
    "soccer": "soccer", "futbol": "soccer", "fútbol": "soccer",
}


def sport_for_league(league_key: str) -> SportConfig | None:
    """Resuelve el deporte desde la liga. Más fiable que el nombre genérico."""
    if not league_key:
        return None
    for s in SPORTS.values():
        if s.league(league_key):
            return s
    return None


def get_sport(key: str) -> SportConfig | None:
    k = (key or "").strip().lower()
    if k in SPORTS:
        return SPORTS[k]
    if k in ALIASES:
        return SPORTS[ALIASES[k]]
    for s in SPORTS.values():
        if s.label.lower() == k:
            return s
    return sport_for_league(key)


def resolve(sport_key: str, league_key: str = None):
    """Devuelve (SportConfig, LeagueConfig) o (None, None)."""
    s = get_sport(sport_key) or sport_for_league(league_key)
    if not s:
        return None, None
    if league_key:
        return s, s.league(league_key)
    act = s.active_leagues()
    return s, (act[0] if act else None)


def validate(sport_key: str, league_key: str, market_key: str) -> tuple[bool, str]:
    """
    Puerta de entrada. La API la usa antes de guardar.
    Rechaza combinaciones que el deporte no declara — no las adivina.
    """
    s = get_sport(sport_key) or sport_for_league(league_key)
    if not s:
        return False, f"deporte desconocido: {sport_key} (liga: {league_key})"
    if not s.enabled:
        return False, f"{s.label} está desactivado en la configuración"
    lg = s.league(league_key) if league_key else None
    if league_key and not lg:
        return False, f"liga desconocida para {s.label}: {league_key}"
    if lg and not lg.enabled:
        return False, f"liga desactivada: {lg.label}"
    if not s.supports(market_key):
        disp = [m.key for m in s.markets if m.enabled]
        return False, f"{s.label} no admite el mercado '{market_key}'. Disponibles: {disp}"
    return True, "ok"


def is_operational(sport_key: str, league_key: str = None) -> tuple[bool, str]:
    """Sin partidos jugados no se generan picks: tener rating no es tener datos."""
    s, lg = resolve(sport_key, league_key)
    if not s:
        return False, "deporte desconocido"
    if not lg:
        return False, "sin liga activa"
    if lg.games_played <= 0:
        return False, f"{lg.label}: 0 partidos jugados esta temporada"
    return True, f"{lg.label}: {lg.games_played} partidos de muestra"


def summary() -> dict:
    out = {}
    for s in SPORTS.values():
        out[s.key] = {
            "label": s.label, "enabled": s.enabled,
            "markets": [m.key for m in s.markets if m.enabled],
            "markets_pending": [m.key for m in s.markets if not m.enabled],
            "leagues": [{"key": lg.key, "label": lg.label, "enabled": lg.enabled,
                         "games_played": lg.games_played, "data_quality": lg.data_quality,
                         "operational": lg.games_played > 0}
                        for lg in s.leagues],
            "models": s.models,
            "params": {"home_advantage": s.home_advantage, "scale": s.scale,
                       "shrinkage_k": s.shrinkage_k, "base_rate_home": s.base_rate_home,
                       "has_draws": s.has_draws,
                       "no_pick_threshold": s.no_pick_threshold},
        }
    return out


if __name__ == "__main__":
    import json
    print("DEPORTES CONFIGURADOS\n" + "=" * 62)
    for s in SPORTS.values():
        est = "ACTIVO" if s.enabled else "desactivado"
        print(f"\n{s.icon} {s.label}  [{est}]")
        print(f"   mercados   : {', '.join(m.key for m in s.markets if m.enabled)}")
        pend = [m.key for m in s.markets if not m.enabled]
        if pend:
            print(f"   pendientes : {', '.join(pend)}")
        print(f"   modelos    : {', '.join(s.models)}")
        print(f"   params     : local +{s.home_advantage} · escala {s.scale} · "
              f"k={s.shrinkage_k} · base rate {s.base_rate_home:.3f} · "
              f"empate={'sí' if s.has_draws else 'no'}")
        for lg in s.leagues:
            ok, msg = is_operational(s.key, lg.key)
            flag = "operativa" if ok else "NO operativa"
            act = "on " if lg.enabled else "off"
            print(f"     [{act}] {lg.label:<18} {lg.games_played:>3} j.  "
                  f"{lg.data_quality:<10} {flag}")
