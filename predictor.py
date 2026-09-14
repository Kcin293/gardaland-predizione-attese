"""Caricamento dei modelli allenati da training.py e calcolo delle previsioni
ai tre livelli: totale parco, per reame e per singola giostra.

I dati di input arrivano dalle API pubbliche Open-Meteo (meteo previsto) e
queue-times.com (storico delle code, gia' in database)."""

from __future__ import annotations

import logging
from functools import lru_cache

import joblib
import pandas as pd

from training import LIVELLI, MODEL_PATHS, build_feature_row
from working_day import day_type, e_alta_stagione

logger = logging.getLogger(__name__)


class ModelloNonDisponibile(RuntimeError):
    """Il modello di un livello non e' ancora stato allenato."""


@lru_cache(maxsize=8)
def _carica_bundle_cached(livello: str, mtime: float) -> dict:
    return joblib.load(MODEL_PATHS[livello])


def carica_bundle(livello: str) -> dict:
    """Carica il bundle di un livello, ricaricandolo da disco solo quando il
    file cambia (cioe' dopo un nuovo training)."""
    if livello not in LIVELLI:
        raise ValueError(f"Livello sconosciuto: {livello}")

    path = MODEL_PATHS[livello]
    if not path.exists():
        raise ModelloNonDisponibile(
            f"Modello '{livello}' non allenato: esegui `python training.py`."
        )
    return _carica_bundle_cached(livello, path.stat().st_mtime)


def livelli_disponibili() -> list[str]:
    return [livello for livello in LIVELLI if MODEL_PATHS[livello].exists()]


def _pulisci(valore: float) -> float:
    """Una coda negativa non esiste: i modelli lineari possono predirla."""
    return round(max(float(valore), 0.0), 1)


def _numero(valore) -> float | None:
    """I numeri di sklearn sono numpy float: JSON non li digerisce."""
    return None if valore is None else float(valore)


def _info_modello(bundle: dict) -> dict:
    metriche = bundle.get("metriche", {})
    return {
        "modello": bundle.get("nome_modello"),
        "r2": _numero(metriche.get("r2")),
        "mae": _numero(metriche.get("mae")),
        "rmse": _numero(metriche.get("rmse")),
        "n_training_rows": bundle.get("n_training_rows"),
        "allenato_il": bundle.get("run_at"),
    }


def prevedi_totale(giorno, media_tmp: float, precip_tmp: float) -> dict:
    """Coda media prevista per l'intero parco (attrazioni principali)."""
    bundle = carica_bundle("totale")
    X = build_feature_row(giorno, media_tmp, precip_tmp, bundle["feature_columns"])
    return {
        "coda": _pulisci(bundle["pipeline"].predict(X)[0]),
        **_info_modello(bundle),
    }


def _prevedi_gruppo(bundle: dict, giorno, media_tmp: float, precip_tmp: float,
                    dummy_per_entita: list[tuple[str, ...]]) -> list[float]:
    """Una sola predict() per tutte le entita' di un livello."""
    righe = [
        build_feature_row(giorno, media_tmp, precip_tmp,
                          bundle["feature_columns"], dummy)
        for dummy in dummy_per_entita
    ]
    if not righe:
        return []
    X = pd.concat(righe, ignore_index=True)
    return [_pulisci(v) for v in bundle["pipeline"].predict(X)]


def prevedi_per_reame(giorno, media_tmp: float, precip_tmp: float) -> dict:
    """Coda media prevista per ogni reame del parco."""
    bundle = carica_bundle("reame")
    reami = bundle.get("reami", [])
    code = _prevedi_gruppo(
        bundle, giorno, media_tmp, precip_tmp,
        [(f"reame_{reame}",) for reame in reami],
    )
    voci = [{"nome": reame, "coda": coda} for reame, coda in zip(reami, code)]
    voci.sort(key=lambda v: v["coda"], reverse=True)
    return {"voci": voci, **_info_modello(bundle)}


def prevedi_per_giostra(giorno, media_tmp: float, precip_tmp: float) -> dict:
    """Coda prevista per ogni singola giostra."""
    bundle = carica_bundle("giostra")
    giostre = bundle.get("giostre", [])
    code = _prevedi_gruppo(
        bundle, giorno, media_tmp, precip_tmp,
        [(f"reame_{g['categoria']}", f"giostra_{g['nome']}") for g in giostre],
    )
    voci = [
        {"nome": g["nome"], "reame": g["categoria"], "coda": coda}
        for g, coda in zip(giostre, code)
    ]
    voci.sort(key=lambda v: v["coda"], reverse=True)
    return {"voci": voci, **_info_modello(bundle)}


def previsione_completa(meteo: dict) -> dict:
    """Previsione ai tre livelli per il giorno descritto da `meteo`
    (output di ingestion.meteo_client.get_previsione_domani()).

    Se un livello non e' ancora stato allenato viene riportato l'errore
    al posto dei numeri, senza far fallire gli altri livelli."""
    giorno = meteo["giorno"]
    media_tmp = meteo["media_tmp"]
    precip_tmp = meteo["precip_tmp"]

    risultato = {
        "giorno": str(giorno),
        "meteo": {
            "media_tmp": round(float(media_tmp), 1),
            "precip_tmp": round(float(precip_tmp), 2),
        },
        "calendario": {
            "tipo_giorno": day_type(giorno).value,
            "alta_stagione": bool(e_alta_stagione(giorno)),
        },
    }

    for chiave, funzione in (
        ("totale", prevedi_totale),
        ("reami", prevedi_per_reame),
        ("giostre", prevedi_per_giostra),
    ):
        try:
            risultato[chiave] = funzione(giorno, media_tmp, precip_tmp)
        except Exception as e:
            logger.exception("Previsione '%s' non disponibile.", chiave)
            risultato[chiave] = {"errore": str(e)}

    return risultato
