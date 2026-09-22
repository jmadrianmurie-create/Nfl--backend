-- ============================================================
-- TIPSTER — Base histórica de predicciones
-- SQLite para desarrollo · portable a Postgres (ver NOTAS al final)
-- ============================================================

PRAGMA foreign_keys = ON;

-- ------------------------------------------------------------
-- 1. PREDICCIONES — núcleo INMUTABLE
--    Se escribe una vez. Nunca se actualiza.
--    Todo estado posterior va en prediction_events.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id      TEXT PRIMARY KEY,

  -- Identificación
  event_id           TEXT NOT NULL,
  sport              TEXT NOT NULL,
  league             TEXT NOT NULL,
  season             TEXT,
  event_name         TEXT NOT NULL,
  team_home          TEXT NOT NULL,
  team_away          TEXT NOT NULL,
  market             TEXT NOT NULL,        -- moneyline | spread | total | prop
  selection          TEXT NOT NULL,

  -- Tiempo
  ts_prediction      TEXT NOT NULL,        -- ISO 8601 con offset
  timezone           TEXT NOT NULL,
  ts_odds_capture    TEXT,
  ts_data_update     TEXT,

  -- Modelo
  model_version      TEXT NOT NULL,
  ensemble_version   TEXT NOT NULL,
  feature_version    TEXT NOT NULL,
  config_json        TEXT NOT NULL,
  prob_by_model_json TEXT NOT NULL,        -- {"record":0.60,"pythag":0.57,...}
  prob_final         REAL NOT NULL CHECK (prob_final > 0 AND prob_final < 1),
  uncertainty_lo     REAL,
  uncertainty_hi     REAL,
  confidence         INTEGER,

  -- Evaluación económica (NULL cuando no hay cuotas: no se inventa)
  fair_odds          REAL,
  implied_prob       REAL,
  devig_prob         REAL,
  edge               REAL,
  ev                 REAL,

  -- Decisión
  decision           TEXT NOT NULL,        -- BET | SMALL BET | WAIT | NO BET | REJECTED
  decision_reason    TEXT,
  stake_recommended  REAL,
  risk_level         TEXT,

  -- Red team / robustez (se guardan TAMBIÉN los rechazados)
  redteam_verdict    TEXT,
  redteam_severity   INTEGER,
  redteam_rejected   INTEGER DEFAULT 0,
  redteam_reason     TEXT,
  robustness_score   REAL,
  robustness_verdict TEXT,

  -- Mercado en el momento de decidir (snapshot completo)
  market_snapshot_json TEXT,

  -- Linaje de datos
  data_lineage_json  TEXT NOT NULL,
  snapshot_hash      TEXT NOT NULL,        -- reproducibilidad

  -- Procedencia: live = capturada en vivo · backfill_audited = del log auditado
  provenance         TEXT NOT NULL DEFAULT 'live',

  -- PRICED   = tiene cuota sharp de entrada -> puede completar el ciclo CLV
  -- UNPRICED = nace sin cuota -> entra al histórico, NO al contador de validación
  pricing_status     TEXT NOT NULL DEFAULT 'UNPRICED',

  created_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_pred_event  ON predictions(event_id);
CREATE INDEX IF NOT EXISTS ix_pred_ts     ON predictions(ts_prediction);
CREATE INDEX IF NOT EXISTS ix_pred_league ON predictions(league);
CREATE INDEX IF NOT EXISTS ix_pred_dec    ON predictions(decision);

-- Una misma decisión no se duplica; nueva info => nueva predicción (v2, v3)
CREATE UNIQUE INDEX IF NOT EXISTS ux_pred_unique
  ON predictions(event_id, market, selection, model_version, ts_prediction);

-- ------------------------------------------------------------
-- 2. INMUTABILIDAD — bloqueo a nivel de base de datos
--    No depende de que el código de la app se porte bien.
-- ------------------------------------------------------------
CREATE TRIGGER IF NOT EXISTS trg_predictions_immutable
BEFORE UPDATE ON predictions
BEGIN
  SELECT RAISE(ABORT,
    'INMUTABLE: una predicción no se modifica. Los estados posteriores van en prediction_events.');
END;

CREATE TRIGGER IF NOT EXISTS trg_predictions_no_delete
BEFORE DELETE ON predictions
BEGIN
  SELECT RAISE(ABORT, 'INMUTABLE: las predicciones no se borran.');
END;

-- ------------------------------------------------------------
-- 3. EVENTOS — append-only. Aquí viven closing line, resultado, post-mortem.
--    Misma prediction_id => trazabilidad completa del ciclo de vida.
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_events (
  event_seq      INTEGER PRIMARY KEY AUTOINCREMENT,
  prediction_id  TEXT NOT NULL REFERENCES predictions(prediction_id),
  event_type     TEXT NOT NULL,   -- CLOSING_LINE | RESULT | POST_MORTEM | EXECUTION
  ts_event       TEXT NOT NULL,
  payload_json   TEXT NOT NULL,
  created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS ix_ev_pred ON prediction_events(prediction_id, event_type);

CREATE TRIGGER IF NOT EXISTS trg_events_immutable
BEFORE UPDATE ON prediction_events
BEGIN
  SELECT RAISE(ABORT, 'INMUTABLE: los eventos son append-only.');
END;

-- Un solo cierre y un solo resultado por predicción
CREATE UNIQUE INDEX IF NOT EXISTS ux_ev_closing
  ON prediction_events(prediction_id) WHERE event_type = 'CLOSING_LINE';
CREATE UNIQUE INDEX IF NOT EXISTS ux_ev_result
  ON prediction_events(prediction_id) WHERE event_type = 'RESULT';

-- ------------------------------------------------------------
-- 4. CUOTAS — serie temporal del mercado
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
  id           INTEGER PRIMARY KEY AUTOINCREMENT,
  event_id     TEXT NOT NULL,
  market       TEXT NOT NULL,
  selection    TEXT NOT NULL,
  sportsbook   TEXT NOT NULL,
  odds         REAL NOT NULL,
  line         REAL,
  ts           TEXT NOT NULL,
  is_opening   INTEGER DEFAULT 0,
  is_closing   INTEGER DEFAULT 0,
  source       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_ev ON odds_snapshots(event_id, market, ts);

-- ------------------------------------------------------------
-- 5. REGISTRO DE MODELOS — nunca se sobrescribe una versión
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
  model_version  TEXT PRIMARY KEY,
  params_json    TEXT NOT NULL,
  features_json  TEXT NOT NULL,
  created_at     TEXT NOT NULL,
  metrics_json   TEXT,
  is_champion    INTEGER DEFAULT 0,
  notes          TEXT
);

CREATE TRIGGER IF NOT EXISTS trg_model_no_overwrite
BEFORE UPDATE OF params_json, features_json, created_at ON model_registry
BEGIN
  SELECT RAISE(ABORT, 'INMUTABLE: una versión de modelo no se reescribe. Crea una nueva.');
END;

-- ------------------------------------------------------------
-- 6. VISTAS DE VALIDACIÓN
--    CLV = (cuota_tomada / cuota_cierre) - 1
-- ------------------------------------------------------------
CREATE VIEW IF NOT EXISTS v_predictions_full AS
SELECT
  p.*,
  json_extract(c.payload_json, '$.closing_odds')      AS closing_odds,
  json_extract(c.payload_json, '$.closing_line')      AS closing_line,
  json_extract(c.payload_json, '$.closing_sportsbook') AS closing_sportsbook,
  c.ts_event                                          AS ts_closing,
  json_extract(p.market_snapshot_json, '$.odds')      AS odds_taken_best,
  -- CLV: SOLO sharp de entrada contra sharp de cierre.
  -- Si falta cualquiera de las dos, queda NULL. No se sustituye por
  -- consenso ni por mejor precio: seria mezclar referencias distintas.
  json_extract(c.payload_json, '$.clv')               AS clv,
  json_extract(c.payload_json, '$.clv_basis')         AS clv_basis,
  json_extract(p.market_snapshot_json, '$.sharp_odds') AS sharp_entry,
  json_extract(r.payload_json, '$.outcome')           AS outcome,
  json_extract(r.payload_json, '$.score_home')        AS score_home,
  json_extract(r.payload_json, '$.score_away')        AS score_away,
  r.ts_event                                          AS ts_result
FROM predictions p
LEFT JOIN prediction_events c
  ON c.prediction_id = p.prediction_id AND c.event_type = 'CLOSING_LINE'
LEFT JOIN prediction_events r
  ON r.prediction_id = p.prediction_id AND r.event_type = 'RESULT';

-- ============================================================
-- NOTAS PARA MIGRAR A POSTGRES
--   TEXT ts            -> TIMESTAMPTZ
--   TEXT *_json        -> JSONB
--   AUTOINCREMENT      -> BIGSERIAL
--   json_extract(x,'$.k') -> x->>'k'
--   Triggers RAISE(ABORT) -> RULE ... DO INSTEAD NOTHING, o
--                            REVOKE UPDATE, DELETE ON predictions FROM app_user;
--   El filtro anti-leakage vive SIEMPRE en SQL:
--     WHERE ts < :ts_decision      -- nunca NOW()
-- ============================================================
