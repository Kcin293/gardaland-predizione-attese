-- Schema Gardaland Wait-Time Predictor
-- Ricostruisce lo schema del vecchio DB n8n_garda (stesse tabelle/view,
-- cosi' un dump storico si puo' reimportare senza modifiche) e aggiunge
-- model_runs per loggare i confronti tra modelli ML nel tempo.

-- === Tabelle esistenti (compatibili col dump storico) ===

CREATE TABLE IF NOT EXISTS public.giostre (
    id             numeric NOT NULL,
    nome           text,
    is_open        boolean,
    wait_time      numeric,
    last_updated   timestamptz NOT NULL,
    categoria      text
);

-- Una giostra puo' comparire piu' volte con lo stesso last_updated se il job
-- di ingestion gira due volte per errore: evitiamo duplicati esatti.
CREATE UNIQUE INDEX IF NOT EXISTS ux_giostre_id_last_updated
    ON public.giostre (id, last_updated);

CREATE INDEX IF NOT EXISTS ix_giostre_last_updated
    ON public.giostre (last_updated);

CREATE TABLE IF NOT EXISTS public.meteo (
    temperatura_percepita  double precision,
    precipitazioni         double precision,
    data_ora               timestamptz NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_meteo_data_ora
    ON public.meteo (data_ora);

CREATE TABLE IF NOT EXISTS public.previsioni (
    giorno       date NOT NULL PRIMARY KEY,
    previsione   double precision
);

-- === Nuova tabella: log dei training/confronti modelli ===

CREATE TABLE IF NOT EXISTS public.model_runs (
    id                integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    run_at            timestamptz NOT NULL DEFAULT now(),
    best_model        text,
    best_mae          double precision,
    best_rmse         double precision,
    best_r2           double precision,
    n_training_rows   integer,
    metrics_json      jsonb,
    livello           text
);

-- Su database creati prima dei tre livelli di previsione la colonna manca.
ALTER TABLE public.model_runs ADD COLUMN IF NOT EXISTS livello text;

-- === View di aggregazione giornaliera (identiche al dump originale) ===

CREATE OR REPLACE VIEW public.coda_giornaliera AS
 SELECT nome,
        categoria,
        (last_updated)::date AS giorno,
        avg(wait_time) AS media_coda
   FROM public.giostre
  WHERE (is_open IS TRUE)
  GROUP BY nome, categoria, ((last_updated)::date)
  ORDER BY nome, ((last_updated)::date);

CREATE OR REPLACE VIEW public.temp_giornaliera AS
 SELECT (data_ora)::date AS giorno,
        avg(temperatura_percepita) AS media_tmp,
        avg(precipitazioni) AS precip_tmp
   FROM public.meteo
  GROUP BY (data_ora)::date
  ORDER BY (data_ora)::date;

CREATE OR REPLACE VIEW public.totale_giornaliero AS
 SELECT (meteo.data_ora)::date AS giorno,
        avg(meteo.temperatura_percepita) AS media_tmp,
        avg(meteo.precipitazioni) AS precip_tmp,
        avg(giostre.wait_time) AS coda
   FROM public.meteo
   JOIN public.giostre ON ((meteo.data_ora)::date = (giostre.last_updated)::date)
  WHERE (giostre.categoria = 'Adrenaline' OR giostre.categoria = 'Adventure')
  GROUP BY (meteo.data_ora)::date
  ORDER BY (meteo.data_ora)::date;
