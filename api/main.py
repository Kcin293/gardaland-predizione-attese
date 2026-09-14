"""API e dashboard del Gardaland Wait-Time Predictor.

Le pagine HTML sono servite da api/static/, i dati dagli endpoint /api/*.
Fonti dei dati: API pubblica di queue-times.com per le code storiche e API
Open-Meteo per meteo osservato e previsto."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

import predictor
from ingestion import crud
from ingestion.meteo_client import get_previsione_domani

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Gardaland Wait-Time Predictor",
    description=(
        "Previsione delle code a Gardaland: totale parco, per reame e per "
        "singola giostra. Dati da queue-times.com e Open-Meteo."
    ),
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _record(df: pd.DataFrame) -> list[dict]:
    """DataFrame -> lista di dict serializzabili (i Decimal di Postgres e i
    tipi numpy non passano il JSON encoder di FastAPI)."""
    if df.empty:
        return []
    df = df.copy()
    for colonna in df.columns:
        if colonna == "giorno":
            df[colonna] = df[colonna].astype(str)
            continue
        try:
            df[colonna] = pd.to_numeric(df[colonna]).astype(float)
        except (TypeError, ValueError):
            df[colonna] = df[colonna].astype(str)
    return df.to_dict("records")


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health():
    return {"stato": "ok", "modelli": predictor.livelli_disponibili()}


@app.get("/api/previsione")
def api_previsione():
    """Previsione di domani ai tre livelli: totale, reame e giostra."""
    meteo = get_previsione_domani()
    if meteo is None:
        raise HTTPException(
            status_code=503,
            detail="Previsione meteo non disponibile (API Open-Meteo irraggiungibile).",
        )
    return predictor.previsione_completa(meteo)


@app.get("/api/storico")
def api_storico(giorni: int = 180):
    """Serie storica: coda media del parco e meteo osservato negli ultimi
    `giorni` giorni in cui il parco era aperto (i giorni di chiusura non
    compaiono nello storico)."""
    df = pd.read_sql(
        "SELECT giorno, coda, media_tmp, precip_tmp "
        "FROM totale_giornaliero ORDER BY giorno DESC LIMIT %(limite)s",
        crud.engine,
        params={"limite": max(1, min(giorni, 3650))},
    )
    return {"voci": list(reversed(_record(df)))}


@app.get("/api/storico/giostre")
def api_storico_giostre(giorni: int = 60):
    """Media della coda per giostra negli ultimi N giorni di rilevazioni."""
    df = pd.read_sql(
        """
        SELECT nome, categoria AS reame, avg(media_coda) AS coda_media,
               max(media_coda) AS coda_max, count(*) AS giorni_rilevati
          FROM coda_giornaliera
         WHERE giorno >= (SELECT max(giorno) FROM coda_giornaliera)
                          - (%(giorni)s * INTERVAL '1 day')
         GROUP BY nome, categoria
         ORDER BY coda_media DESC
        """,
        crud.engine,
        params={"giorni": max(1, min(giorni, 3650))},
    )
    return {"giorni": giorni, "voci": _record(df)}


@app.get("/api/modelli")
def api_modelli():
    """Ultimo confronto tra modelli per ciascun livello di previsione."""
    df = pd.read_sql(
        """
        SELECT DISTINCT ON (coalesce(livello, 'totale'))
               coalesce(livello, 'totale') AS livello,
               run_at, best_model, best_r2, best_mae, best_rmse,
               n_training_rows, metrics_json
          FROM model_runs
         ORDER BY coalesce(livello, 'totale'), run_at DESC
        """,
        crud.engine,
    )
    if df.empty:
        return {"livelli": []}

    livelli = []
    for _, riga in df.iterrows():
        livelli.append({
            "livello": riga["livello"],
            "run_at": str(riga["run_at"]),
            "best_model": riga["best_model"],
            "r2": float(riga["best_r2"]) if riga["best_r2"] is not None else None,
            "mae": float(riga["best_mae"]) if riga["best_mae"] is not None else None,
            "rmse": float(riga["best_rmse"]) if riga["best_rmse"] is not None else None,
            "n_training_rows": int(riga["n_training_rows"] or 0),
            "metriche": riga["metrics_json"] or [],
        })

    ordine = {"totale": 0, "reame": 1, "giostra": 2}
    livelli.sort(key=lambda v: ordine.get(v["livello"], 99))
    return {"livelli": livelli}


@app.get("/api/stato")
def api_stato():
    """Riepilogo di cosa c'e' in database: utile in testata alla dashboard."""
    df = pd.read_sql(
        """
        SELECT max(last_updated) AS ultimo_aggiornamento,
               count(DISTINCT nome) AS giostre_monitorate,
               count(DISTINCT last_updated::date) AS giorni_storico,
               count(*) AS rilevazioni
          FROM giostre
        """,
        crud.engine,
    )
    riga = df.iloc[0]
    return {
        "ultimo_aggiornamento": str(riga["ultimo_aggiornamento"]),
        "giostre_monitorate": int(riga["giostre_monitorate"] or 0),
        "giorni_storico": int(riga["giorni_storico"] or 0),
        "rilevazioni": int(riga["rilevazioni"] or 0),
        "modelli": predictor.livelli_disponibili(),
        "fonti": [
            {
                "nome": "Queue-Times API",
                "descrizione": "tempi di attesa delle giostre di Gardaland",
                "url": "https://queue-times.com/parks/12/queue_times.json",
            },
            {
                "nome": "Open-Meteo API",
                "descrizione": "temperatura percepita e precipitazioni, osservate e previste",
                "url": "https://open-meteo.com/",
            },
        ],
    }
