"""
TIPSTER - Backend minimo de validacion.
Objetivo unico: acumular una base historica real de predicciones
para poder validar o refutar el cerebro con evidencia.

NO calcula probabilidades. Eso lo hace el motor.
Este servicio persiste, versiona y mide.

BASE DE DATOS (revision 22-sep-2026):
  - Si existe la variable DATABASE_URL (postgres://...), usa POSTGRES
    (Neon, plan gratis permanente) con schema_pg.sql.
  - Si no existe, usa SQLite local (tipster.db) con schema.sql, como antes.
  Motivo: en Render gratis el disco es efimero; SQLite se borraba cada vez
  que el servicio se dormia, se reiniciaba o se redesplegaba.

Correr local:  python3 app.py
"""
import json, math, os, sqlite3, uuid, hashlib
from datetime import datetime, timezone
from flask import Flask, request, jsonify, g
import sports_config as SC

HERE    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.environ.get("TIPSTER_DB", os.path.join(HERE, "tipster.db"))
PORT    = int(os.environ.get("TIPSTER_PORT", "5055"))
SCHEMA_SQLITE = os.path.join(HERE, "schema.sql")
SCHEMA_PG     = os.path.join(HERE, "schema_pg.sql")

DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()
USE_PG = DATABASE_URL.startswith(("postgres://", "postgresql://"))

if USE_PG:
    import psycopg
    from psycopg.rows import dict_row
    IntegrityError = psycopg.IntegrityError
else:
    IntegrityError = sqlite3.IntegrityError

# Criterios de validacion (del informe tecnico)
N_MIN_CALIB, N_MIN_CLV = 200, 150
BRIER_BASELINE, LOGLOSS_BASELINE = 0.25, 0.6931
Z95, Z80 = 1.959964, 0.841621

app = Flask(__name__)

# CORS - el motor corre en el navegador, la API en otro origen.
ALLOWED_ORIGIN = os.environ.get("TIPSTER_ALLOWED_ORIGIN", "*")

# Autenticacion de escritura. Si TIPSTER_API_KEY esta definida, todo POST la exige.
WRITE_KEY = os.environ.get("TIPSTER_API_KEY", "")

app.config["MAX_CONTENT_LENGTH"] = 256 * 1024


@app.before_request
def require_key():
    if request.method != "POST":
        return None
    if not WRITE_KEY:
        return None
    sent = (request.headers.get("Authorization", "").replace("Bearer ", "").strip()
            or request.headers.get("X-Api-Key", "").strip())
    if sent != WRITE_KEY:
        return jsonify({"error": "no autorizado",
                        "detail": "falta o no coincide la clave de escritura"}), 401
    return None

@app.after_request
def add_cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    # X-Api-Key agregado: require_key lo acepta, pero sin declararlo aqui
    # el navegador bloqueaba la peticion en el preflight.
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Api-Key"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp

@app.route("/<path:_any>", methods=["OPTIONS"])
@app.route("/", methods=["OPTIONS"])
def cors_preflight(_any=None):
    return ("", 204)


# ----------------------------------------------------------------- DB
class DB:
    """
    Capa minima comun a SQLite y Postgres. El SQL de la app se escribe con
    placeholders '?' (estilo SQLite); en Postgres se traducen a '%s'.
    Todas las filas salen como dict en ambos motores.
    """
    def __init__(self):
        if USE_PG:
            self.con = psycopg.connect(DATABASE_URL, row_factory=dict_row,
                                       connect_timeout=15)
        else:
            self.con = sqlite3.connect(DB_PATH)
            self.con.row_factory = sqlite3.Row
            self.con.execute("PRAGMA foreign_keys = ON")

    @staticmethod
    def _sql(sql, args):
        if USE_PG and args:
            return sql.replace("%", "%%").replace("?", "%s")
        return sql

    def _exec(self, sql, args=()):
        args = tuple(args or ())
        try:
            if args:
                return self.con.execute(self._sql(sql, args), args)
            return self.con.execute(sql)
        except Exception:
            self.rollback()
            raise

    def all(self, sql, args=()):
        return [dict(r) for r in self._exec(sql, args).fetchall()]

    def one(self, sql, args=()):
        r = self._exec(sql, args).fetchone()
        return dict(r) if r is not None else None

    def scalar(self, sql, args=()):
        r = self._exec(sql, args).fetchone()
        if r is None:
            return None
        return list(r.values())[0] if USE_PG else r[0]

    def run(self, sql, args=()):
        self._exec(sql, args)

    def commit(self):
        self.con.commit()

    def rollback(self):
        try:
            self.con.rollback()
        except Exception:
            pass

    def close(self):
        self.con.close()


def db() -> DB:
    if "db" not in g:
        g.db = DB()
    return g.db


@app.teardown_appcontext
def close_db(exc):
    d = g.pop("db", None)
    if d is not None:
        d.close()


def init_db():
    if USE_PG:
        with psycopg.connect(DATABASE_URL, connect_timeout=30) as con:
            with open(SCHEMA_PG) as f:
                con.execute(f.read())   # sin parametros: admite varias sentencias
            con.commit()
    else:
        con = sqlite3.connect(DB_PATH)
        with open(SCHEMA_SQLITE) as f:
            con.executescript(f.read())
        con.commit()
        con.close()


# SIEMPRE al importar: gunicorn (Render) nunca entra a __main__.
# Ambos esquemas son idempotentes (IF NOT EXISTS / OR REPLACE).
# Si la base no responde (Neon dormido, URL mal copiada), el servicio NO se
# cae: arranca, /health reporta el error real y los endpoints NFL siguen vivos.
DB_INIT_ERROR = None
try:
    init_db()
except Exception as _e:
    DB_INIT_ERROR = f"{type(_e).__name__}: {_e}"
    print(f"[app] ERROR inicializando la base: {DB_INIT_ERROR}")


def _json_safe(x):
    """NaN/inf no son JSON valido: el navegador falla en JSON.parse. -> None."""
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    if isinstance(x, dict):
        return {k: _json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_json_safe(v) for v in x]
    return x


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def snapshot_hash(payload: dict) -> str:
    """Hash del snapshot: mismo input -> mismo hash -> reproducibilidad."""
    canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode()).hexdigest()[:32]


def _is_full_bet(decision: str) -> bool:
    dd = (decision or "").upper()
    return ("BET" in dd and "NO BET" not in dd
            and "SMALL" not in dd and "RECHAZ" not in dd)


def _brain_is_validated() -> bool:
    """
    El cerebro solo se considera validado si supera las puertas criticas
    con muestra suficiente. Por defecto: NO. Es deliberado.
    """
    try:
        rows = db().all("SELECT prob_final, outcome, clv FROM v_predictions_full")
    except Exception:
        return False
    graded = [r for r in rows if r["outcome"] in ("win", "loss")]
    clvs = [r["clv"] for r in rows if r["clv"] is not None]
    if len(graded) < N_MIN_CALIB or len(clvs) < N_MIN_CLV:
        return False
    brier = sum((r["prob_final"] - (1 if r["outcome"] == "win" else 0)) ** 2
                for r in graded) / len(graded)
    return brier < BRIER_BASELINE and (sum(clvs) / len(clvs)) > 0


# ------------------------------------------------- POST /predictions
@app.post("/predictions")
def create_prediction():
    d = request.get_json(force=True)

    required = ["event_id", "sport", "league", "event_name", "team_home",
                "team_away", "market", "selection", "prob_final",
                "model_version", "decision"]
    missing = [k for k in required if d.get(k) is None]
    if missing:
        return jsonify({"error": "campos obligatorios ausentes", "missing": missing}), 400

    p = float(d["prob_final"])
    if not (0.0 < p < 1.0):
        return jsonify({"error": "prob_final debe estar entre 0 y 1 (exclusivo)"}), 400

    # Capa por deporte: cada uno declara sus mercados. No se fuerzan los mismos.
    sport_key = d.get("sport_key") or d["sport"]
    ok, motivo = SC.validate(sport_key, d.get("league_key") or d.get("league"), d["market"])
    if not ok:
        return jsonify({"error": "combinacion no admitida", "detail": motivo}), 422

    # Sin partidos jugados no se generan picks apostables
    op, op_msg = SC.is_operational(sport_key, d.get("league_key") or d.get("league"))
    decision = d["decision"]

    # CONDICION 9 - Guarda conservadora del servidor.
    # Mientras el cerebro no este VALIDADO, ningun pick puede ser BET pleno,
    # exista o no una cuota favorable. Se degrada a SMALL BET y se deja constancia.
    if _is_full_bet(decision) and not _brain_is_validated():
        d["decision_reason"] = ((d.get("decision_reason") or "") +
            " | DEGRADADO por el servidor: el cerebro no esta validado "
            "(muestra o calibracion insuficientes). Una cuota favorable "
            "no basta para elevar a BET.").strip()
        decision = "SMALL BET"
        d["decision"] = decision
    if not op and ("BET" in decision.upper() and "NO BET" not in decision.upper()):
        return jsonify({"error": "liga no operativa",
                        "detail": op_msg + " - no puede producir apuestas"}), 422

    pid = d.get("prediction_id") or str(uuid.uuid4())
    ts_pred = d.get("ts_prediction") or now_iso()

    market_snap = d.get("market_snapshot")   # None si no hay cuotas: NO se inventa

    # UNPRICED: la prediccion nace sin cuota. Entra al historico, NO al
    # contador de validacion. Se puede enriquecer despues si aparece la cuota.
    is_priced = bool(market_snap and market_snap.get("sharp_odds"))
    pricing_status = "PRICED" if is_priced else "UNPRICED"
    lineage = d.get("data_lineage") or {}

    # Metricas economicas: solo si existe cuota real. Si no, NULL.
    fair_odds = round(1.0 / p, 4)
    implied = devig = edge = ev = None
    if market_snap and market_snap.get("odds"):
        o = float(market_snap["odds"])
        implied = round(1.0 / o, 5)
        devig = market_snap.get("devig_prob")
        edge = round(p - implied, 5)
        ev = round(p * (o - 1.0) - (1.0 - p), 5)

    core = {
        "event_id": d["event_id"], "market": d["market"], "selection": d["selection"],
        "prob_final": p, "prob_by_model": d.get("prob_by_model", {}),
        "model_version": d["model_version"], "ts_prediction": ts_pred,
        "market_snapshot": market_snap, "config": d.get("config", {}),
    }
    shash = snapshot_hash(core)

    row = (
        pid, d["event_id"], d["sport"], d["league"], d.get("season"),
        d["event_name"], d["team_home"], d["team_away"], d["market"], d["selection"],
        ts_pred, d.get("timezone", "UTC"), d.get("ts_odds_capture"), d.get("ts_data_update"),
        d["model_version"], d.get("ensemble_version", d["model_version"]),
        d.get("feature_version", "f1.0"),
        json.dumps(d.get("config", {}), ensure_ascii=False),
        json.dumps(d.get("prob_by_model", {}), ensure_ascii=False),
        p, d.get("uncertainty_lo"), d.get("uncertainty_hi"), d.get("confidence"),
        fair_odds, implied, devig, edge, ev,
        d["decision"], d.get("decision_reason"), d.get("stake_recommended"), d.get("risk_level"),
        d.get("redteam_verdict"), d.get("redteam_severity"),
        1 if d.get("redteam_rejected") else 0, d.get("redteam_reason"),
        d.get("robustness_score"), d.get("robustness_verdict"),
        json.dumps(market_snap, ensure_ascii=False) if market_snap else None,
        json.dumps(lineage, ensure_ascii=False), shash,
        d.get("provenance", "live"), pricing_status,
    )

    sql = """INSERT INTO predictions (
      prediction_id, event_id, sport, league, season, event_name, team_home, team_away,
      market, selection, ts_prediction, timezone, ts_odds_capture, ts_data_update,
      model_version, ensemble_version, feature_version, config_json, prob_by_model_json,
      prob_final, uncertainty_lo, uncertainty_hi, confidence,
      fair_odds, implied_prob, devig_prob, edge, ev,
      decision, decision_reason, stake_recommended, risk_level,
      redteam_verdict, redteam_severity, redteam_rejected, redteam_reason,
      robustness_score, robustness_verdict, market_snapshot_json,
      data_lineage_json, snapshot_hash, provenance, pricing_status
    ) VALUES (""" + ",".join(["?"] * 43) + ")"

    c = db()
    try:
        c.run(sql, row)
        c.commit()
    except IntegrityError as e:
        return jsonify({"error": "prediccion duplicada", "detail": str(e)}), 409

    n = c.scalar("SELECT COUNT(*) FROM predictions")
    return jsonify({"prediction_id": pid, "snapshot_hash": shash,
                    "fair_odds": fair_odds, "edge": edge, "ev": ev,
                    "pricing_status": pricing_status,
                    "validatable": is_priced,
                    "note": None if is_priced else
                            "UNPRICED: sin cuota sharp de entrada. Entra al historico "
                            "pero NO al contador de validacion CLV.",
                    "total_predictions": n}), 201


# -------------------------------------------------- GET /predictions
@app.get("/predictions")
def list_predictions():
    q = "SELECT * FROM v_predictions_full WHERE 1=1"
    args = []
    for f in ("league", "decision", "market", "provenance"):
        if request.args.get(f):
            q += f" AND {f} = ?"
            args.append(request.args[f])
    if request.args.get("has_clv") == "true":
        q += " AND clv IS NOT NULL"
    if request.args.get("has_result") == "true":
        q += " AND outcome IS NOT NULL"
    q += " ORDER BY ts_prediction DESC LIMIT ?"
    args.append(int(request.args.get("limit", 100)))

    rows = db().all(q, args)
    return jsonify({"count": len(rows), "predictions": rows})


@app.get("/predictions/<pid>")
def get_prediction(pid):
    out = db().one("SELECT * FROM v_predictions_full WHERE prediction_id = ?", (pid,))
    if not out:
        return jsonify({"error": "no encontrada"}), 404
    out["events"] = db().all(
        "SELECT event_type, ts_event, payload_json FROM prediction_events "
        "WHERE prediction_id = ? ORDER BY event_seq", (pid,))
    return jsonify(out)


# ------------------------------- POST /predictions/:id/closing-line
@app.post("/predictions/<pid>/closing-line")
def post_closing_line(pid):
    d = request.get_json(force=True)
    if d.get("closing_odds") is None:
        return jsonify({"error": "closing_odds obligatorio"}), 400

    pred = db().one(
        "SELECT market_snapshot_json FROM predictions WHERE prediction_id = ?", (pid,))
    if not pred:
        return jsonify({"error": "prediccion no encontrada"}), 404

    closing = float(d["closing_odds"])
    payload = {
        "closing_odds": closing,
        "closing_line": d.get("closing_line"),
        "closing_sportsbook": d.get("closing_sportsbook"),
        "source": d.get("source"),
    }

    # CONDICION 8: el CLV se calcula SOLO sharp contra sharp.
    # Nunca se mezcla con consenso ni con mejor precio disponible.
    clv = None
    snap = json.loads(pred["market_snapshot_json"]) if pred["market_snapshot_json"] else None
    sharp_entry = snap.get("sharp_odds") if snap else None
    is_sharp_close = bool(d.get("is_sharp", True))

    payload["closing_is_sharp"] = is_sharp_close
    if snap:
        payload["odds_taken_best"] = snap.get("odds")          # se guarda, NO entra al CLV
        payload["sharp_entry"] = sharp_entry

    if sharp_entry and closing > 0 and is_sharp_close:
        clv = round(float(sharp_entry) / closing - 1.0, 6)
        payload["clv"] = clv
        payload["clv_pct"] = round(clv * 100, 3)
        payload["clv_basis"] = "sharp_entry / sharp_close"
        payload["movement"] = round(closing - float(sharp_entry), 4)
    else:
        payload["clv"] = None
        payload["clv_basis"] = None
        payload["clv_unavailable_reason"] = (
            "sin cuota sharp de entrada" if not sharp_entry
            else "la cuota de cierre no es de una casa sharp")

    try:
        c = db()
        c.run("INSERT INTO prediction_events (prediction_id, event_type, ts_event, payload_json)"
              " VALUES (?,?,?,?)",
              (pid, "CLOSING_LINE", d.get("ts_closing") or now_iso(),
               json.dumps(payload, ensure_ascii=False)))
        c.commit()
    except IntegrityError:
        return jsonify({"error": "ya existe closing line para esta prediccion"}), 409

    return jsonify({"prediction_id": pid, "clv": clv,
                    "clv_pct": payload.get("clv_pct"),
                    "clv_basis": payload.get("clv_basis"),
                    "note": None if clv is not None
                            else payload.get("clv_unavailable_reason")}), 201


# ------------------------------------ POST /predictions/:id/result
@app.post("/predictions/<pid>/result")
def post_result(pid):
    d = request.get_json(force=True)
    outcome = d.get("outcome")
    if outcome not in ("win", "loss", "push", "void"):
        return jsonify({"error": "outcome debe ser win|loss|push|void"}), 400

    pred = db().one(
        "SELECT prob_final, market_snapshot_json FROM predictions WHERE prediction_id = ?",
        (pid,))
    if not pred:
        return jsonify({"error": "prediccion no encontrada"}), 404

    p = pred["prob_final"]
    y = 1 if outcome == "win" else 0
    payload = {
        "outcome": outcome,
        "score_home": d.get("score_home"),
        "score_away": d.get("score_away"),
        "prob_error": round(p - y, 5) if outcome in ("win", "loss") else None,
        "brier_contrib": round((p - y) ** 2, 6) if outcome in ("win", "loss") else None,
        "error_type": d.get("error_type"),   # variance|model|data|market|event
        "notes": d.get("notes"),
    }

    # ROI hipotetico solo si hubo cuota real
    snap = json.loads(pred["market_snapshot_json"]) if pred["market_snapshot_json"] else None
    if snap and snap.get("odds") and outcome in ("win", "loss"):
        o = float(snap["odds"])
        payload["roi_units"] = round(o - 1.0, 4) if outcome == "win" else -1.0

    try:
        c = db()
        c.run("INSERT INTO prediction_events (prediction_id, event_type, ts_event, payload_json)"
              " VALUES (?,?,?,?)",
              (pid, "RESULT", d.get("ts_result") or now_iso(),
               json.dumps(payload, ensure_ascii=False)))
        c.commit()
    except IntegrityError:
        return jsonify({"error": "ya existe resultado para esta prediccion"}), 409

    return jsonify({"prediction_id": pid, **payload}), 201


# ---------------------------------------- GET /validation/summary
def wilson(w, n):
    if not n:
        return {"p": None, "lo": None, "hi": None}
    p = w / n
    dd = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / dd
    m = (Z95 / dd) * math.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n))
    return {"p": round(p, 4), "lo": round(max(0, c - m), 4), "hi": round(min(1, c + m), 4)}


def n_for_proportion(p_real, p_base):
    if p_real <= p_base:
        return None
    num = Z95 * math.sqrt(p_base * (1 - p_base)) + Z80 * math.sqrt(p_real * (1 - p_real))
    return math.ceil(num ** 2 / (p_real - p_base) ** 2)


def n_for_mean(delta, sigma):
    return math.ceil((Z95 + Z80) ** 2 * sigma ** 2 / delta ** 2)


@app.get("/validation/summary")
def validation_summary():
    c = db()
    total = c.scalar("SELECT COUNT(*) FROM predictions")
    bets = c.scalar("SELECT COUNT(*) FROM predictions WHERE decision LIKE '%BET%'"
                    " AND decision NOT LIKE 'NO BET%'")
    rows = c.all("SELECT prob_final, outcome, clv FROM v_predictions_full")

    graded = [r for r in rows if r["outcome"] in ("win", "loss")]
    n_res = len(graded)
    wins = sum(1 for r in graded if r["outcome"] == "win")

    brier = logloss = None
    if n_res:
        brier = round(sum((r["prob_final"] - (1 if r["outcome"] == "win" else 0)) ** 2
                          for r in graded) / n_res, 4)
        eps = 1e-9
        logloss = round(-sum(
            (1 if r["outcome"] == "win" else 0) * math.log(max(r["prob_final"], eps)) +
            (0 if r["outcome"] == "win" else 1) * math.log(max(1 - r["prob_final"], eps))
            for r in graded) / n_res, 4)

    clvs = [r["clv"] for r in rows if r["clv"] is not None]
    clv_mean = round(sum(clvs) / len(clvs), 5) if clvs else None

    # STATUS: nunca VALIDATED sin cumplir criterios
    if total == 0:
        status = "NO DATA"
    elif len(clvs) < N_MIN_CLV or n_res < N_MIN_CALIB:
        status = "INSUFFICIENT SAMPLE"
    elif brier is not None and brier < BRIER_BASELINE and clv_mean and clv_mean > 0:
        status = "VALIDATED"
    else:
        status = "UNDERPERFORMING"

    priced = c.scalar("SELECT COUNT(*) FROM predictions WHERE pricing_status='PRICED'")
    return jsonify({
        "total_predictions": total,
        "priced": priced,
        "unpriced": total - priced,
        "bet_decisions": bets,
        "validation_sample": n_res,
        "clv_sample": len(clvs),
        "result_sample": n_res,
        "record": f"{wins}-{n_res - wins}" if n_res else None,
        "win_rate": wilson(wins, n_res),
        "brier_score": brier,
        "brier_baseline": BRIER_BASELINE,
        "log_loss": logloss,
        "log_loss_baseline": LOGLOSS_BASELINE,
        "current_clv": clv_mean,
        "status": status,
        "gates": {
            "calibration": {"n": n_res, "n_required": N_MIN_CALIB,
                            "met": bool(n_res >= N_MIN_CALIB and brier is not None
                                        and brier < BRIER_BASELINE)},
            "clv": {"n": len(clvs), "n_required": N_MIN_CLV,
                    "met": bool(len(clvs) >= N_MIN_CLV and clv_mean and clv_mean > 0)},
        },
        "still_needed": {
            "clv_observations": max(0, N_MIN_CLV - len(clvs)),
            "calibration_observations": max(0, N_MIN_CALIB - n_res),
            "n_to_prove_beats_market_by_winrate": n_for_proportion(0.57, 0.55),
            "n_to_prove_positive_clv": n_for_mean(0.01, 0.04),
        },
    })


SIGMA_ASSUMED = 0.04      # supuesto inicial, se reemplaza al haber datos
CLV_MIN_N_FOR_SIGMA = 30  # antes de 30 obs. el sigma medido no es fiable


def clv_required_n(observed_sigma=None, n_obs=0, delta=0.01):
    """
    Cuantas observaciones hacen falta para detectar un CLV medio de `delta`.
    Con >= 30 observaciones usa el sigma REAL. Antes, el supuesto.
    Devuelve tambien de donde salio, para no esconder la suposicion.
    """
    if observed_sigma and n_obs >= CLV_MIN_N_FOR_SIGMA:
        sigma, basis = observed_sigma, f"sigma observado ({observed_sigma:.4f}, n={n_obs})"
    else:
        sigma, basis = SIGMA_ASSUMED, f"sigma SUPUESTO ({SIGMA_ASSUMED}) - aun sin medir"
    n = math.ceil((Z95 + Z80) ** 2 * sigma ** 2 / delta ** 2)
    return n, basis, sigma


@app.get("/validation/clv")
def validation_clv():
    rows = db().all(
        "SELECT prediction_id, league, market, selection, sharp_entry, odds_taken_best,"
        " closing_odds, clv, clv_basis, outcome"
        " FROM v_predictions_full WHERE clv IS NOT NULL ORDER BY ts_prediction DESC")
    clvs = [r["clv"] for r in rows]
    n = len(clvs)
    mean = sd = t = None
    if n:
        mean = sum(clvs) / n
        if n > 1:
            sd = math.sqrt(sum((x - mean) ** 2 for x in clvs) / (n - 1))
            if sd > 0:
                t = mean / (sd / math.sqrt(n))
    req_n, basis, sigma_used = clv_required_n(sd, n)
    return jsonify({
        "n": n, "mean_clv": round(mean, 5) if mean is not None else None,
        "sd": round(sd, 5) if sd else None,
        "required_n": req_n, "required_n_basis": basis,
        "sigma_used": round(sigma_used, 5),
        "still_needed": max(0, req_n - n),
        "t_stat": round(t, 3) if t else None,
        "significant_at_95": bool(t and abs(t) > 1.96),
        "clv_basis": "sharp_entry / sharp_close (nunca consenso ni mejor precio)",
        "status": "INSUFFICIENT SAMPLE" if n < N_MIN_CLV else
                  ("VALIDATED" if (t and t > 1.96) else "UNDERPERFORMING"),
        "rows": rows,
    })


@app.get("/validation/calibration")
def validation_calibration():
    rows = db().all(
        "SELECT prob_final, outcome FROM v_predictions_full"
        " WHERE outcome IN ('win','loss')")
    buckets, n = [], len(rows)
    for lo, hi in [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.00)]:
        sub = [r for r in rows if lo <= r["prob_final"] < hi]
        if sub:
            pred = sum(r["prob_final"] for r in sub) / len(sub)
            real = sum(1 for r in sub if r["outcome"] == "win") / len(sub)
            buckets.append({"range": f"{int(lo*100)}-{int(hi*100)}%", "n": len(sub),
                            "predicted": round(pred * 100, 1), "actual": round(real * 100, 1),
                            "gap_pp": round((pred - real) * 100, 1)})
    ece = round(sum(b["n"] / n * abs(b["predicted"] - b["actual"]) for b in buckets) / 100, 4) if n else None
    return jsonify({"n": n, "ece": ece, "ece_target": 0.04,
                    "n_required": N_MIN_CALIB,
                    "status": "INSUFFICIENT SAMPLE" if n < N_MIN_CALIB else
                              ("VALIDATED" if ece and ece < 0.04 else "UNDERPERFORMING"),
                    "buckets": buckets})


@app.get("/sports")
def sports_catalog():
    """Configuracion viva de deportes, ligas y mercados."""
    return jsonify(SC.summary())


@app.get("/health")
def health():
    global DB_INIT_ERROR
    if DB_INIT_ERROR:
        try:                      # reintento: Neon puede haber despertado
            init_db()
            DB_INIT_ERROR = None
        except Exception as e:
            DB_INIT_ERROR = f"{type(e).__name__}: {e}"
            return jsonify({"ok": False,
                            "db": "postgres" if USE_PG else "sqlite",
                            "error": DB_INIT_ERROR}), 503
    n = db().scalar("SELECT COUNT(*) FROM predictions")
    # Nunca se devuelve DATABASE_URL: contiene la contrasena de la base.
    return jsonify({"ok": True,
                    "db": "postgres" if USE_PG else f"sqlite ({DB_PATH})",
                    "persistent": USE_PG,
                    "total_predictions": n})




# ================= NFL ENGINE - endpoint =================
from nfl_engine import TeamInputs, NFLConfig, analyze_game
from serializers import serialize_analysis


# nflverse identifica a los equipos por ABREVIATURA en el play-by-play
# (posteam/defteam). La app manda nombres completos ("Buffalo Bills").
# Sin esta traduccion, stats.get("Buffalo Bills") siempre daba None y el
# motor respondia NO_DATA para los 32 equipos aunque la descarga funcionara.
# Codigos oficiales de nflverse: Rams = "LA", Jaguars = "JAX", Commanders = "WAS".
NFL_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}
# Abreviaturas alternas que usan otras fuentes (ESPN, CBS, historicas).
_NFL_ABBR_ALIASES = {"LAR": "LA", "JAC": "JAX", "WSH": "WAS", "OAK": "LV",
                     "SD": "LAC", "STL": "LA"}
_NFL_CODES = set(NFL_ABBR.values())


def nfl_team_code(name: str):
    """Nombre completo o abreviatura -> codigo nflverse. None si no se reconoce."""
    if not name:
        return None
    n = str(name).strip()
    if n in NFL_ABBR:
        return NFL_ABBR[n]
    up = n.upper()
    if up in _NFL_CODES:
        return up
    if up in _NFL_ABBR_ALIASES:
        return _NFL_ABBR_ALIASES[up]
    low = n.lower()
    for full, code in NFL_ABBR.items():
        if full.lower() == low:
            return code
    return None


@app.post("/api/nfl/predict")
def nfl_predict():
    """
    NO recibe nombres de equipo como texto libre para "buscar" sus stats:
    eso obligaria a inventar una base de datos de equipos que no existe.
    Recibe los INSUMOS explicitos (EPA, power rating + fuente, etc.).
    Si faltan, el motor los excluye y lo declara - nunca los rellena.
    """
    d = request.get_json(force=True) or {}

    def team_from(payload, name_key):
        return TeamInputs(
            name=payload.get(name_key, ""),
            games_played=payload.get("games_played", 0),
            epa_off_per_play=payload.get("epa_off_per_play"),
            epa_def_per_play=payload.get("epa_def_per_play"),
            power_rating=payload.get("power_rating"),
            power_rating_source=payload.get("power_rating_source"),
            rest_days=payload.get("rest_days"),
            travel_miles=payload.get("travel_miles"),
            key_injuries=payload.get("key_injuries", 0),
        )

    home_payload = d.get("home", {})
    away_payload = d.get("away", {})
    if not home_payload.get("name") or not away_payload.get("name"):
        return jsonify({"error": "faltan home.name y/o away.name"}), 400

    home = team_from(home_payload, "name")
    away = team_from(away_payload, "name")

    result = analyze_game(
        home, away,
        cfg=NFLConfig(),
        market_home_odds=d.get("market_home_odds"),
        market_away_odds=d.get("market_away_odds"),
        market_spread=d.get("market_spread"),
        expected_total_points=d.get("expected_total_points", 44.0),
    )
    return jsonify(_json_safe(serialize_analysis(result)))


@app.post("/api/nfl/predict/auto")
def nfl_predict_auto():
    """
    Igual que /api/nfl/predict, pero arma TeamInputs automaticamente desde
    nfl_data_loader (nflverse) en vez de requerir que el cliente mande
    EPA a mano. Requiere internet en ESTE servidor (no en el cliente).

    Si nfl_data_loader no esta instalado o la descarga falla, responde
    NO_DATA explicito - nunca inventa numeros para poder continuar.
    """
    d = request.get_json(force=True) or {}
    home_name = d.get("home_team")
    away_name = d.get("away_team")
    if not home_name or not away_name:
        return jsonify({"error": "faltan home_team y/o away_team"}), 400

    home_code, away_code = nfl_team_code(home_name), nfl_team_code(away_name)
    unknown = [n for n, c in ((home_name, home_code), (away_name, away_code)) if c is None]
    if unknown:
        return jsonify({"error": "equipo NFL no reconocido",
                        "detail": f"sin codigo nflverse para: {unknown}"}), 400

    try:
        from nfl_data_loader import get_team_epa_stats
    except ImportError as e:
        return jsonify({
            "error": "nfl_data_loader no disponible en este servidor",
            "detail": f"{type(e).__name__}: {e}",
            "hint": "revisa este 'detail' - dice exactamente que fallo al importar"
        }), 503
    except Exception as e:
        return jsonify({
            "error": "Error inesperado cargando nfl_data_loader",
            "detail": f"{type(e).__name__}: {e}"
        }), 500

    from nfl_data_loader import current_nfl_season
    # La temporada NFL arranca en septiembre: en ene-feb sigue la del ano anterior.
    current_year = int(d.get("year") or current_nfl_season())
    prior_year = current_year - 1

    stats, games = get_team_epa_stats(year=current_year)
    if not stats:
        from nfl_data_loader import get_last_error
        real_error = get_last_error() or "sin detalle capturado"
        return jsonify({
            "error": "Sin datos de nflverse disponibles",
            "detail": real_error
        }), 502

    # Prior de temporada anterior: falla TOLERANTE. Si 2025 no se puede
    # descargar (o el equipo no aparece), el motor ya sabe operar sin
    # prior (usa el neutro 0.0) - no se hard-failea el endpoint por esto,
    # porque el prior es una mejora, no un requisito.
    try:
        prior_stats, _ = get_team_epa_stats(year=prior_year)
    except Exception:
        prior_stats = {}

    def build_team(name, code):
        s = stats.get(code)
        p = prior_stats.get(code) if prior_stats else None
        return TeamInputs(
            name=name,                                # nombre completo: es lo que ve la app
            games_played=games.get(code, 0),          # SIEMPRE temporada actual, POR EQUIPO
            epa_off_per_play=s["epa_offense"] if s else None,
            epa_def_per_play=s["epa_defense"] if s else None,
            prior_epa_off_per_play=p["epa_offense"] if p else None,
            prior_epa_def_per_play=p["epa_defense"] if p else None,
            prior_season_label=f"{prior_year} EPA/play, temporada regular" if p else None,
            power_rating=d.get(f"power_rating_{name}"),
            power_rating_source=d.get("power_rating_source"),
        )

    home = build_team(home_name, home_code)
    away = build_team(away_name, away_code)

    result = analyze_game(
        home, away, cfg=NFLConfig(),
        market_home_odds=d.get("market_home_odds"),
        market_away_odds=d.get("market_away_odds"),
        market_spread=d.get("market_spread"),
        expected_total_points=d.get("expected_total_points", 44.0),
    )
    out = serialize_analysis(result)
    out["data_source"] = {
        "provider": "nflverse via nflreadpy",
        "current_season": current_year,
        "prior_season": prior_year,
        "prior_available": {"home": home.prior_epa_off_per_play is not None,
                            "away": away.prior_epa_off_per_play is not None},
        "home_games_played_real": games.get(home_code, 0),
        "away_games_played_real": games.get(away_code, 0),
        "team_codes": {"home": home_code, "away": away_code},
        "current_available": {"home": home_code in stats, "away": away_code in stats},
        # Sede neutral: se registra tal como la manda la app. El motor no la usa
        # porque hoy ningun submodelo activo aplica ventaja de local (ver nota).
        "neutral_site": bool(d.get("neutral_site")),
        "note": "games_played es POR EQUIPO y SIEMPRE de la temporada actual "
                f"({current_year}), tomado de game_id unicos reales - nunca un "
                "promedio de liga ni inflado por el prior. El prior de "
                f"{prior_year} solo ajusta el punto de partida del shrinkage "
                "(net_epa); el gate N>=4 sigue atado exclusivamente a "
                "partidos reales de la temporada actual.",
    }
    return jsonify(_json_safe(out))


# Al final del archivo: antes las rutas NFL se definian DESPUES de este
# bloque, y en local (python3 app.py) app.run() arrancaba sin ellas.
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)
