"""
backend/nfl_data_loader.py - ingesta oficial via nflreadpy (nflverse).

Requiere conexion a internet y la libreria instalada:
    pip install nflreadpy

CAMBIO DE LIBRERIA: se usa `nflreadpy` porque `nfl_data_py` fue
DEPRECADA y archivada por nflverse (lanzaba NameError con datos 2026).

REVISION 7 (22-sep-2026) - correcciones para Render plan gratis (512 MB):

1. Ya NO se convierte a pandas. `polars.DataFrame.to_pandas()` exige
   `pyarrow`, que no esta en requirements.txt: en produccion eso es un
   ImportError en cada llamada. Ahora todo el agregado se hace en Polars
   (que ya viene con nflreadpy) y solo salen dicts pequenos.
2. Se reduce a 6 columnas ANTES de filtrar y agregar, y se libera el
   DataFrame completo (~370 columnas) en cuanto se selecciona.
3. Se APAGA el cache en memoria de nflreadpy. Por defecto guarda el
   play-by-play completo en RAM; con la temporada actual + la anterior
   (el prior) eso puede tumbar el servicio por falta de memoria. El
   cache propio de este modulo guarda solo los agregados (32 equipos).
4. Filas con posteam/defteam nulo se excluyen explicitamente. pandas las
   descartaba solo al agrupar; Polars las agrupa como un "equipo" nulo.
5. La temporada anterior esta cerrada: su agregado se cachea 24 h en vez
   de 1 h, para no volver a descargarla en cada peticion.

Logica de negocio SIN cambios: EPA/play ofensivo y defensivo por equipo,
partidos jugados POR EQUIPO (game_id unicos, union ofensa/defensa), solo
temporada regular, y ({}, {}) si algo falla - nunca datos inventados.

Limitacion a declarar siempre: nflverse actualiza el play-by-play hasta
~48h despues de cada partido (no es un feed en vivo).
"""
from __future__ import annotations

import gc
import os
import time
from datetime import datetime
from typing import Optional

# Apagar el cache en RAM de nflreadpy ANTES de importarlo (variable de
# entorno documentada) y otra vez por codigo despues, por si la version
# instalada lee la configuracion en otro momento.
os.environ.setdefault("NFLREADPY_CACHE", "off")
os.environ.setdefault("NFLREADPY_VERBOSE", "false")

import nflreadpy as nfl
import polars as pl

try:
    from nflreadpy.config import update_config
    update_config(cache_mode="off", verbose=False)
except Exception as _cfg_err:  # la variable de entorno ya cubre este caso
    print(f"[nfl_data_loader] update_config no disponible: {_cfg_err}")

# Unicas columnas que usa este modulo.
_PBP_COLUMNS = ["game_id", "posteam", "defteam", "play_type", "epa", "season_type"]

# Cache de agregados en memoria del proceso.
# En Render gratis el proceso se reinicia al dormirse; solo evita recalcular
# mientras el servicio esta despierto.
_CACHE: dict[int, tuple[float, dict, dict]] = {}
_TTL_CURRENT_SECONDS = 3600          # temporada en curso: 1 hora
_TTL_CLOSED_SECONDS = 24 * 3600      # temporada cerrada: 24 horas

LAST_ERROR: Optional[str] = None     # texto real de la ultima excepcion


def get_last_error() -> Optional[str]:
    return LAST_ERROR


def _current_nfl_season() -> int:
    """La temporada NFL arranca en septiembre: en ene-ago sigue siendo la del ano anterior."""
    now = datetime.now()
    return now.year if now.month >= 9 else now.year - 1


def _ttl_for(year: int) -> int:
    return _TTL_CLOSED_SECONDS if year < _current_nfl_season() else _TTL_CURRENT_SECONDS


def _cache_get(year: int) -> Optional[tuple[dict, dict]]:
    entry = _CACHE.get(year)
    if entry is None:
        return None
    ts, stats, games = entry
    if time.time() - ts > _ttl_for(year):
        return None
    return stats, games


def _cache_set(year: int, stats: dict, games: dict) -> None:
    _CACHE[year] = (time.time(), stats, games)


def clear_cache() -> None:
    """Fuerza una recarga en la siguiente llamada, ignorando el TTL."""
    _CACHE.clear()


# Archivo oficial de nflverse por temporada (el mismo que usa nflreadpy).
_PBP_URL = ("https://github.com/nflverse/nflverse-data/releases/download/"
            "pbp/play_by_play_{year}.parquet")


def _load_pbp_columns(year: int) -> "pl.DataFrame":
    """
    Via 1 (preferida): leer el parquet oficial pidiendo SOLO 6 columnas.
    Parquet es columnar: las otras ~370 ni se decodifican. La temporada
    completa anterior (el prior) cabe asi en pocos MB en vez de cientos,
    que es lo que puede tumbar el plan gratis de Render (512 MB).

    Via 2 (respaldo): nflreadpy.load_pbp completo y recorte inmediato.
    Si ambas fallan, se propaga el error de la via 2 junto con el de la 1.
    """
    url = _PBP_URL.format(year=year)
    try:
        df = pl.read_parquet(url, columns=_PBP_COLUMNS)
        return df
    except Exception as e1:
        first = f"{type(e1).__name__}: {e1}"
        print(f"[nfl_data_loader] lectura directa fallo ({first}); uso nflreadpy")
    try:
        full = nfl.load_pbp(seasons=year)
        missing_cols = [c for c in _PBP_COLUMNS if c not in full.columns]
        if missing_cols:
            raise KeyError(f"columnas esperadas ausentes en nflreadpy: {missing_cols}")
        df = full.select(_PBP_COLUMNS)
        del full
        gc.collect()
        return df
    except Exception as e2:
        raise RuntimeError(f"directa: {first} | nflreadpy: {type(e2).__name__}: {e2}")


def get_team_epa_stats(year: Optional[int] = None,
                       season_type: str = "REG",
                       use_cache: bool = True) -> tuple[dict[str, dict], dict[str, int]]:
    """
    Calcula EPA/play ofensivo y defensivo POR EQUIPO y partidos jugados
    POR EQUIPO para la temporada `year`.

    Retorna:
        stats: {team: {"epa_offense": float|None, "epa_defense": float|None}}
        games_played: {team: int}

    Si la descarga o el calculo fallan, retorna ({}, {}) y deja el error
    real en LAST_ERROR. Nunca datos parciales o inventados.
    """
    global LAST_ERROR
    if year is None:
        year = _current_nfl_season()

    if use_cache:
        cached = _cache_get(year)
        if cached is not None:
            return cached

    # --- descarga: solo las 6 columnas necesarias ---
    try:
        pbp = _load_pbp_columns(year)
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__}: {e}"
        print(f"[nfl_data_loader] Error al descargar datos de nflverse: {LAST_ERROR}")
        return {}, {}

    if pbp.height == 0:
        LAST_ERROR = (f"nflverse devolvio 0 filas para la temporada {year} "
                      "(aun no hay datos publicados, o el ano es incorrecto)")
        print(f"[nfl_data_loader] {LAST_ERROR}")
        return {}, {}

    # --- agregacion en Polars (sin pandas, sin pyarrow) ---
    try:
        # Temporada regular: el k=8 de shrinkage en NFLConfig asume ese ritmo.
        valid = pbp.filter(
            (pl.col("season_type") == season_type)
            & pl.col("play_type").is_in(["pass", "run"])
            & pl.col("epa").is_not_null()
        )
        del pbp

        if valid.height == 0:
            LAST_ERROR = (f"nflverse tiene datos de {year} pero 0 jugadas pass/run "
                          f"con EPA para season_type={season_type}")
            print(f"[nfl_data_loader] {LAST_ERROR}")
            return {}, {}

        # epa_defense = EPA promedio del RIVAL contra este equipo (mas alto =
        # peor defensa), convencion de nfl_engine.submodel_epa (net = off - def).
        off = (valid.filter(pl.col("posteam").is_not_null())
                    .group_by("posteam")
                    .agg(pl.col("epa").mean().alias("epa"),
                         pl.col("game_id").unique().alias("games")))
        dfn = (valid.filter(pl.col("defteam").is_not_null())
                    .group_by("defteam")
                    .agg(pl.col("epa").mean().alias("epa"),
                         pl.col("game_id").unique().alias("games")))
        del valid

        offense_epa: dict[str, float] = {}
        games_off: dict[str, set] = {}
        for row in off.iter_rows(named=True):
            offense_epa[row["posteam"]] = float(row["epa"])
            games_off[row["posteam"]] = set(row["games"])

        defense_epa: dict[str, float] = {}
        games_def: dict[str, set] = {}
        for row in dfn.iter_rows(named=True):
            defense_epa[row["defteam"]] = float(row["epa"])
            games_def[row["defteam"]] = set(row["games"])
        del off, dfn
        gc.collect()
    except Exception as e:
        LAST_ERROR = f"{type(e).__name__} al agregar EPA: {e}"
        print(f"[nfl_data_loader] {LAST_ERROR}")
        return {}, {}

    # --- partidos jugados POR EQUIPO: union de game_id como ofensa y defensa ---
    teams = set(offense_epa) | set(defense_epa)
    games_played: dict[str, int] = {
        t: len(games_off.get(t, set()) | games_def.get(t, set())) for t in teams
    }
    stats: dict[str, dict] = {
        t: {"epa_offense": offense_epa.get(t), "epa_defense": defense_epa.get(t)}
        for t in teams
    }

    LAST_ERROR = None
    if use_cache:
        _cache_set(year, stats, games_played)
    return stats, games_played


def get_team_epa(team: str, year: Optional[int] = None) -> Optional[dict]:
    """Atajo para un solo equipo. None (no un valor inventado) si no aparece."""
    stats, games = get_team_epa_stats(year)
    if team not in stats:
        return None
    return {
        "epa_off_per_play": stats[team]["epa_offense"],
        "epa_def_per_play": stats[team]["epa_defense"],
        "games_played": games.get(team, 0),
    }


if __name__ == "__main__":
    # Prueba manual (requiere internet + libreria instalada).
    stats, games = get_team_epa_stats()
    if not stats:
        print(f"Sin datos: {get_last_error()}")
    else:
        def fmt(v):
            return f"{v:>10.3f}" if v is not None else f"{'—':>10}"
        print(f"{'EQUIPO':<6}{'EPA OF':>10}{'EPA DEF':>10}{'PARTIDOS':>10}")
        for team in sorted(stats):
            s = stats[team]
            print(f"{team:<6}{fmt(s['epa_offense'])}{fmt(s['epa_defense'])}"
                  f"{games.get(team, 0):>10d}")


# Nombre publico usado por app.py
current_nfl_season = _current_nfl_season
