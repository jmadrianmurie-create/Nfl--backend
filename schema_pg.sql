-- ============================================================
-- TIPSTER — Base histórica de predicciones · versión POSTGRES
-- Traducción 1:1 de schema.sql (SQLite). Mismas tablas, mismas
-- columnas, mismos índices únicos, misma vista, misma inmutabilidad.
-- Se ejecuta al arrancar app.py: todo es idempotente (IF NOT EXISTS /
-- OR REPLACE), así que repetirlo no altera datos.
--
-- Decisiones de traducción:
--  * JSON y timestamps se guardan como TEXT, igual que en SQLite: la app
--    ya los manda serializados y así no cambia nada del lado de app.py.
--    La vista los lee con ::jsonb.
--  * confidence y redteam_severity -> DOUBLE PRECISION. SQLite aceptaba
--    cualquier número; Postgres INTEGER rechazaría un decimal y se
--    perdería la predicción completa.
--  * Inmutabilidad con triggers PL/pgSQL (equivalente a RAISE(ABORT)).
-- ============================================================

-- Conversión segura texto -> número. Si el valor no es numérico devuelve
-- NULL en vez de romper TODA la consulta de la vista.
CREATE OR REPLACE FUNCTION tip_num(v TEXT) RETURNS DOUBLE PRECISION
LANGUAGE sql IMMUTABLE AS $$
  SELECT CASE WHEN v ~ '^\s*-?[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?\s*$'
              THEN v::double precision END
$$;

-- ------------------------------------------------------------
-- 1. PREDICCIONES — núcleo INMUTABLE
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS predictions (
  prediction_id        TEXT PRIMARY KEY,
  event_id             TEXT NOT NULL,
  sport                TEXT NOT NULL,
  league               TEXT NOT NULL,
  season               TEXT,
  event_name           TEXT NOT NULL,
  team_home            TEXT NOT NULL,
  team_away            TEXT NOT NULL,
  market               TEXT NOT NULL,
  selection            TEXT NOT NULL,
  ts_prediction        TEXT NOT NULL,
  timezone             TEXT NOT NULL,
  ts_odds_capture      TEXT,
  ts_data_update       TEXT,
  model_version        TEXT NOT NULL,
  ensemble_version     TEXT NOT NULL,
  feature_version      TEXT NOT NULL,
  config_json          TEXT NOT NULL,
  prob_by_model_json   TEXT NOT NULL,
  prob_final           DOUBLE PRECISION NOT NULL CHECK (prob_final > 0 AND prob_final < 1),
  uncertainty_lo       DOUBLE PRECISION,
  uncertainty_hi       DOUBLE PRECISION,
  confidence           DOUBLE PRECISION,
  fair_odds            DOUBLE PRECISION,
  implied_prob         DOUBLE PRECISION,
  devig_prob           DOUBLE PRECISION,
  edge                 DOUBLE PRECISION,
  ev                   DOUBLE PRECISION,
  decision             TEXT NOT NULL,
  decision_reason      TEXT,
  stake_recommended    DOUBLE PRECISION,
  risk_level           TEXT,
  redteam_verdict      TEXT,
  redteam_severity     DOUBLE PRECISION,
  redteam_rejected     INTEGER DEFAULT 0,
  redteam_reason       TEXT,
  robustness_score     DOUBLE PRECISION,
  robustness_verdict   TEXT,
  market_snapshot_json TEXT,
  data_lineage_json    TEXT NOT NULL,
  snapshot_hash        TEXT NOT NULL,
  provenance           TEXT NOT NULL DEFAULT 'live',
  pricing_status       TEXT NOT NULL DEFAULT 'UNPRICED',
  created_at           TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
);

CREATE INDEX IF NOT EXISTS ix_pred_event  ON predictions(event_id);
CREATE INDEX IF NOT EXISTS ix_pred_ts     ON predictions(ts_prediction);
CREATE INDEX IF NOT EXISTS ix_pred_league ON predictions(league);
CREATE INDEX IF NOT EXISTS ix_pred_dec    ON predictions(decision);
CREATE UNIQUE INDEX IF NOT EXISTS ux_pred_unique
  ON predictions(event_id, market, selection, model_version, ts_prediction);

-- ------------------------------------------------------------
-- 2. INMUTABILIDAD — bloqueo en la base, no en el código de la app
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION tip_block_predictions() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  IF TG_OP = 'DELETE' THEN
    RAISE EXCEPTION 'INMUTABLE: las predicciones no se borran.';
  END IF;
  RAISE EXCEPTION 'INMUTABLE: una predicción no se modifica. Los estados posteriores van en prediction_events.';
END $$;

CREATE OR REPLACE TRIGGER trg_predictions_immutable
  BEFORE UPDATE ON predictions
  FOR EACH ROW EXECUTE FUNCTION tip_block_predictions();

CREATE OR REPLACE TRIGGER trg_predictions_no_delete
  BEFORE DELETE ON predictions
  FOR EACH ROW EXECUTE FUNCTION tip_block_predictions();

-- TRUNCATE no dispara los triggers por fila: se bloquea aparte.
CREATE OR REPLACE FUNCTION tip_block_truncate() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'INMUTABLE: el histórico no se vacía.';
END $$;

CREATE OR REPLACE TRIGGER trg_predictions_no_truncate
  BEFORE TRUNCATE ON predictions
  FOR EACH STATEMENT EXECUTE FUNCTION tip_block_truncate();

-- ------------------------------------------------------------
-- 3. EVENTOS — append-only
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS prediction_events (
  event_seq     BIGSERIAL PRIMARY KEY,
  prediction_id TEXT NOT NULL REFERENCES predictions(prediction_id),
  event_type    TEXT NOT NULL,
  ts_event      TEXT NOT NULL,
  payload_json  TEXT NOT NULL,
  created_at    TEXT NOT NULL DEFAULT to_char(now() AT TIME ZONE 'UTC', 'YYYY-MM-DD HH24:MI:SS')
);
CREATE INDEX IF NOT EXISTS ix_ev_pred ON prediction_events(prediction_id, event_type);

CREATE OR REPLACE FUNCTION tip_block_events() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'INMUTABLE: los eventos son append-only.';
END $$;

CREATE OR REPLACE TRIGGER trg_events_immutable
  BEFORE UPDATE ON prediction_events
  FOR EACH ROW EXECUTE FUNCTION tip_block_events();

CREATE UNIQUE INDEX IF NOT EXISTS ux_ev_closing
  ON prediction_events(prediction_id) WHERE event_type = 'CLOSING_LINE';
CREATE UNIQUE INDEX IF NOT EXISTS ux_ev_result
  ON prediction_events(prediction_id) WHERE event_type = 'RESULT';

-- ------------------------------------------------------------
-- 4. CUOTAS — serie temporal del mercado
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS odds_snapshots (
  id          BIGSERIAL PRIMARY KEY,
  event_id    TEXT NOT NULL,
  market      TEXT NOT NULL,
  selection   TEXT NOT NULL,
  sportsbook  TEXT NOT NULL,
  odds        DOUBLE PRECISION NOT NULL,
  line        DOUBLE PRECISION,
  ts          TEXT NOT NULL,
  is_opening  INTEGER DEFAULT 0,
  is_closing  INTEGER DEFAULT 0,
  source      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_odds_ev ON odds_snapshots(event_id, market, ts);

-- ------------------------------------------------------------
-- 5. REGISTRO DE MODELOS — una versión nunca se reescribe
-- ------------------------------------------------------------
CREATE TABLE IF NOT EXISTS model_registry (
  model_version TEXT PRIMARY KEY,
  params_json   TEXT NOT NULL,
  features_json TEXT NOT NULL,
  created_at    TEXT NOT NULL,
  metrics_json  TEXT,
  is_champion   INTEGER DEFAULT 0,
  notes         TEXT
);

CREATE OR REPLACE FUNCTION tip_block_model() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
  RAISE EXCEPTION 'INMUTABLE: una versión de modelo no se reescribe. Crea una nueva.';
END $$;

CREATE OR REPLACE TRIGGER trg_model_no_overwrite
  BEFORE UPDATE OF params_json, features_json, created_at ON model_registry
  FOR EACH ROW EXECUTE FUNCTION tip_block_model();

-- ------------------------------------------------------------
-- 6. VISTA DE VALIDACIÓN — mismas columnas que la de SQLite
-- CLV solo sharp de entrada contra sharp de cierre; si falta, NULL.
-- ------------------------------------------------------------
CREATE OR REPLACE VIEW v_predictions_full AS
SELECT
  p.*,
  tip_num(c.payload_json::jsonb ->> 'closing_odds')        AS closing_odds,
  tip_num(c.payload_json::jsonb ->> 'closing_line')        AS closing_line,
  c.payload_json::jsonb ->> 'closing_sportsbook'           AS closing_sportsbook,
  c.ts_event                                               AS ts_closing,
  tip_num(p.market_snapshot_json::jsonb ->> 'odds')        AS odds_taken_best,
  tip_num(c.payload_json::jsonb ->> 'clv')                 AS clv,
  c.payload_json::jsonb ->> 'clv_basis'                    AS clv_basis,
  tip_num(p.market_snapshot_json::jsonb ->> 'sharp_odds')  AS sharp_entry,
  r.payload_json::jsonb ->> 'outcome'                      AS outcome,
  tip_num(r.payload_json::jsonb ->> 'score_home')          AS score_home,
  tip_num(r.payload_json::jsonb ->> 'score_away')          AS score_away,
  r.ts_event                                               AS ts_result
FROM predictions p
LEFT JOIN prediction_events c
  ON c.prediction_id = p.prediction_id AND c.event_type = 'CLOSING_LINE'
LEFT JOIN prediction_events r
  ON r.prediction_id = p.prediction_id AND r.event_type = 'RESULT';
