lo"""
backend/nfl_data_loader.py — ingesta oficial vía nflreadpy (nflverse).

Requiere conexión a internet y la librería instalada:
    pip install nflreadpy

CAMBIO DE LIBRERÍA (importante): este módulo usaba `nfl_data_py`
originalmente. Se migró a `nflreadpy` porque `nfl_data_py` fue
DEPRECADA y su repositorio en GitHub fue ARCHIVADO el 25 de septiembre
de 2025 — "No further nfl_data_py maintenance or updates are planned"
(palabras del propio equipo de nflverse). En producción, `nfl_data_py`
lanzaba un NameError interno ("name 'Error' is not defined") al
procesar datos de la temporada 2026 — un bug real de la librería que
nadie va a corregir porque está archivada. `nflreadpy` es el reemplazo
oficial, mantenido activamente por el mismo equipo de nflverse.

Diferencia de interfaz: `nflreadpy.load_pbp(seasons=...)` devuelve un
DataFrame de Polars, no pandas, y no acepta filtrar columnas en la
descarga (a diferencia de `nfl_data_py.import_pbp_data(columns=...)`).
Se convierte con `.to_pandas()` justo después de descargar, y de ahí
en adelante todo el módulo sigue trabajando en pandas sin cambios —
la lógica de agregación por equipo ya estaba probada y se conservó.

CORRECCIÓN respecto al borrador original: el conteo de partidos jugados
debía ser POR EQUIPO, no un promedio de liga. Un promedio de liga permite
que un equipo con muestra real chica (ej. 2 partidos) pase el filtro de
seguridad MIN_SAMPLE_GAMES=4 usando el número de otro equipo — exactamente
el tipo de atajo que la regla de rechazo existe para evitar.

Limitación a declarar siempre: nflverse actualiza el play-by-play hasta
~48h después de cada partido (no es un feed en vivo). No es un riesgo de
leakage hacia partidos futuros, pero sí implica que el dato del partido
de HOY puede no estar disponible hasta 1-2 días después.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Optional

import nflreadpy as nfl
import pandas as pd

# Columnas mínimas necesarias — reduce la descarga frente al PBP completo
# (~370 columnas normalmente). Estas son las únicas que usa este módulo.
_PBP_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "epa", "season_type"]

# Cache en memoria del PROCESO (nflreadpy tiene su propio cache_mode
# configurable, pero este dict es independiente y explícito). TTL corto
# porque nflverse solo actualiza cada ~48h, pero evita recalcular en
# cada petición HTTP mientras el proceso vive.
_CACHE: dict[int, tuple[float, dict, dict]] = {}
_CACHE_TTL_SECONDS = 3600  # 1 hora

LAST_ERROR: Optional[str] = None  # el texto real de la última excepción,
                                   # para que app.py lo muestre en vez de
                                   # un mensaje genérico


def get_last_error() -> Optional[str]:
    return LAST_ERROR


def _cache_get(year: int) -> Optional[tuple[dict, dict]]:
    entry = _CACHE.get(year)
    if entry is None:
        return None
    ts, stats, games = entry
    if time.time() - ts > _CACHE_TTL_SECONDS:
        return None
    return stats, games


def _cache_set(year: int, stats: dict, games: dict) -> None:
    _CACHE[year] = (time.time(), stats, games)


def clear_cache() -> None:
    """Fuerza una recarga en la siguiente llamada, ignorando el TTL."""
    _CACHE.clear()


def get_team_epa_stats(year: Optional[int] = None,
                        season_type: str = "REG",
                        use_cache: bool = True) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Descarga el play-by-play de la temporada y calcula EPA/play ofensivo
    y defensivo POR EQUIPO, junto con el número real de partidos jugados
    POR EQUIPO (no un promedio de liga).

    use_cache=True (default): reutiliza el resultado si se pidió el mismo
    año hace menos de _CACHE_TTL_SECONDS. Evita descargar/recalcular en
    cada petición HTTP del endpoint.

    Retorna:
        stats: {team: {"epa_offense": float, "epa_defense": float}}
        games_played: {team: int}   <- conteo real, uno por equipo

    Si la descarga falla, retorna ({}, {}) — nunca datos parciales o
    inventados. El llamador debe tratar un resultado vacío como NO_DATA,
    igual que el motor ya hace cuando faltan campos.
    """
    global LAST_ERROR
    if year is None:
        year = datetime.now().year

    if use_cache:
        cached = _cache_get(year)
        if cached is not None:
            return cached

    try:
        # nflreadpy no filtra columnas al descargar (a diferencia de
        # nfl_data_py) — se selecciona con pandas después de convertir.
        # .to_pandas(): nflreadpy usa Polars internamente; el resto de
        # este módulo ya está probado sobre pandas, así que se convierte
        # aquí y se reutiliza esa lógica sin cambios.
        pbp_pl = nfl.load_pbp(seasons=year)
        pbp = pbp_pl.to_pandas()
        missing_cols = [c for c in _PBP_COLUMNS if c not in pbp.columns]
        if missing_cols:
            raise KeyError(f"columnas esperadas ausentes en nflreadpy: {missing_cols}")
        pbp = pbp[_PBP_COLUMNS]
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        print(f"[nfl_data_loader] Error al descargar datos de nflverse: {e}")
        return {}, {}


    if pbp.empty:
        LAST_ERROR = f"nflverse devolvió 0 filas para la temporada {year} (aún no hay datos publicados, o el año es incorrecto)"
        print(f"[nfl_data_loader] Sin datos para la temporada {year} todavía.")
        return {}, {}

    # Filtro de temporada regular: el k=8 de shrinkage en NFLConfig asume
    # el ritmo de una temporada regular de 17 juegos. Mezclar pretemporada
    # o playoffs distorsionaría esa calibración.
    if "season_type" in pbp.columns:
        pbp = pbp[pbp["season_type"] == season_type]

    valid_plays = pbp[
        pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notnull()
    ]

    if valid_plays.empty:
        LAST_ERROR = f"nflverse tiene datos de {year} pero 0 jugadas pass/run con EPA para season_type={season_type}"
        print(f"[nfl_data_loader] Sin jugadas válidas para {year}/{season_type}.")
        return {}, {}

    # --- EPA por equipo ---
    offense_epa = valid_plays.groupby("posteam")["epa"].mean().to_dict()
    # epa_defense = EPA promedio de las jugadas del RIVAL contra este equipo
    # (mientras más alto, peor defensa) — convención consistente con
    # nfl_engine.submodel_epa, que calcula net = off - def.
    defense_epa = valid_plays.groupby("defteam")["epa"].mean().to_dict()

    # --- Partidos jugados POR EQUIPO (la corrección real) ---
    # Un equipo aparece en un game_id ya sea como posteam o defteam en
    # distintas jugadas del mismo partido; se cuenta game_id único por
    # cada rol y se unen ambos conjuntos para no perder partidos donde
    # el equipo casi no tuvo jugadas de un tipo.
    games_as_off = valid_plays.groupby("posteam")["game_id"].apply(set)
    games_as_def = valid_plays.groupby("defteam")["game_id"].apply(set)

    teams = set(offense_epa.keys()) | set(defense_epa.keys())
    games_played: dict[str, int] = {}
    for team in teams:
        g = games_as_off.get(team, set()) | games_as_def.get(team, set())
        games_played[team] = len(g)

    stats: dict[str, dict] = {}
    for team in teams:
        stats[team] = {
            "epa_offense": float(offense_epa.get(team, 0.0)) if team in offense_epa else None,
            "epa_defense": float(defense_epa.get(team, 0.0)) if team in defense_epa else None,
        }

    if use_cache:
        _cache_set(year, stats, games_played)

    return stats, games_played


def get_team_epa(team: str, year: Optional[int] = None) -> Optional[dict]:
    """
    Atajo para un solo equipo. Devuelve None (no un valor inventado) si
    el equipo no aparece en los datos descargados.
    """
    stats, games = get_team_epa_stats(year)
    if team not in stats:
        return None
    return {
        "epa_off_per_play": stats[team]["epa_offense"],
        "epa_def_per_play": stats[team]["epa_defense"],
        "games_played": games.get(team, 0),
    }


if __name__ == "__main__":
    # Prueba manual (requiere internet + librería instalada).
    stats, games = get_team_epa_stats()
    if not stats:
        print("Sin datos (revisa conexión, año, o si la temporada ya empezó).")
    else:
        print(f"{'EQUIPO':<6}{'EPA OF':>10}{'EPA DEF':>10}{'PARTIDOS':>10}")
        for team in sorted(stats.keys()):
            s = stats[team]
            print(f"{team:<6}{s['epa_offense']:>10.3f}{s['epa_defense']:>10.3f}"
                  f"{games.get(team, 0):>10d}")
