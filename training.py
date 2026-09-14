from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sqlalchemy
from sklearn.ensemble import (
    AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
    RandomForestRegressor, StackingRegressor, VotingRegressor,
)
from sklearn.linear_model import (
    BayesianRidge, ElasticNet, ElasticNetCV, HuberRegressor, Lasso, LassoCV,
    LinearRegression, PoissonRegressor, RANSACRegressor, Ridge, RidgeCV,
    TheilSenRegressor, TweedieRegressor,
)
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from ingestion import crud
from working_day import DayType, day_type, e_alta_stagione

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Tre livelli di previsione: la media dell'intero parco, la media di ogni
# reame (le "lands" di queue-times) e la coda della singola giostra.
LIVELLI = ("totale", "reame", "giostra")
MODEL_PATHS = {livello: MODELS_DIR / f"model_{livello}.joblib" for livello in LIVELLI}
MODEL_PATH = MODEL_PATHS["totale"]

# Feature comuni a tutti i livelli: meteo + calendario.
BASE_FEATURES = [
    "media_tmp",
    "precip_tmp",
    "alta_stagione",
    "giorno_num",
] + [f"tipo_{t.value}" for t in DayType]

# Il livello "totale" usa solo le feature di base.
FEATURE_COLUMNS = BASE_FEATURES

TARGET_COLUMN = "coda"

# Oltre questa soglia di righe i modelli piu' lenti (complessita' quadratica o
# peggio) vengono saltati: sui livelli di dettaglio farebbero durare il
# training decine di minuti senza dare risultati migliori.
SOGLIA_RIGHE_MODELLI_LENTI = 1000
MODELLI_LENTI = {"TheilSenRegressor", "SVR", "MLPRegressor"}


def define_day():
    lista = []
    giorni = crud.get_totale_giornaliero(crud.engine)['giorno'].unique()
    for giorno in giorni:
        tipo = day_type(giorno)
        alta_stagione = e_alta_stagione(giorno)
        lista.append({'giorno': giorno, 'tipo_giorno': tipo.value, 'alta_stagione': alta_stagione})
    df = pd.DataFrame(lista)
    return df


def merge_day_type():
    df = define_day()
    df_totale = crud.get_totale_giornaliero(crud.engine)
    df_totale = df_totale.merge(df, on='giorno', how='left')
    df_totale['tipo_giorno'] = pd.Categorical(
        df_totale['tipo_giorno'], categories=[t.value for t in DayType]
    )
    df_totale = pd.get_dummies(df_totale, columns=['tipo_giorno'], prefix='tipo')
    return df_totale


def _aggiungi_feature_calendario(df: pd.DataFrame) -> pd.DataFrame:
    """Aggiunge a un dataframe con colonna 'giorno' le feature di calendario:
    tipo di giorno one-hot, alta stagione e ordinale del giorno (trend)."""
    df = df.copy()
    df["alta_stagione"] = df["giorno"].map(e_alta_stagione)
    df["giorno_num"] = pd.to_datetime(df["giorno"]).map(lambda x: x.toordinal())
    df["tipo_giorno"] = pd.Categorical(
        df["giorno"].map(lambda g: day_type(g).value),
        categories=[t.value for t in DayType],
    )
    return pd.get_dummies(df, columns=["tipo_giorno"], prefix="tipo")


def _dummies_entita(df: pd.DataFrame, colonna: str, prefisso: str) -> tuple[pd.DataFrame, list[str]]:
    """One-hot di una colonna categoriale (reame o giostra): il modello impara
    cosi' il livello di coda tipico di ogni entita', oltre all'effetto meteo."""
    valori = sorted(df[colonna].dropna().unique())
    df[colonna] = pd.Categorical(df[colonna], categories=valori)
    df = pd.get_dummies(df, columns=[colonna], prefix=prefisso)
    return df, [f"{prefisso}_{v}" for v in valori]


def _build_models(escludi_lenti: bool = False) -> dict[str, Pipeline]:
    """Dizionario dei modelli da confrontare. Con escludi_lenti=True restano
    fuori quelli impraticabili sui dataset di dettaglio."""
    base_estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor()),
        ("gb", GradientBoostingRegressor()),
    ]

    def pipe(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])

    modelli = {
        "LinearRegression": pipe(LinearRegression()),
        "Ridge": pipe(Ridge()),
        "RidgeCV": pipe(RidgeCV()),
        "Lasso": pipe(Lasso()),
        "LassoCV": pipe(LassoCV()),
        "ElasticNet": pipe(ElasticNet()),
        "ElasticNetCV": pipe(ElasticNetCV()),
        "BayesianRidge": pipe(BayesianRidge()),
        "HuberRegressor": pipe(HuberRegressor()),
        "TheilSenRegressor": pipe(TheilSenRegressor()),
        "RANSACRegressor": pipe(RANSACRegressor()),
        "PoissonRegressor": pipe(PoissonRegressor(max_iter=1000)),
        "TweedieRegressor": pipe(TweedieRegressor(max_iter=1000)),
        "DecisionTreeRegressor": pipe(DecisionTreeRegressor()),
        "RandomForestRegressor": pipe(RandomForestRegressor()),
        "ExtraTreesRegressor": pipe(ExtraTreesRegressor()),
        "GradientBoostingRegressor": pipe(GradientBoostingRegressor()),
        "HistGradientBoostingRegressor": pipe(HistGradientBoostingRegressor()),
        "AdaBoostRegressor": pipe(AdaBoostRegressor()),
        "BaggingRegressor": pipe(BaggingRegressor()),
        "StackingRegressor": pipe(
            StackingRegressor(estimators=base_estimators, final_estimator=Ridge())
        ),
        "VotingRegressor": pipe(VotingRegressor(estimators=base_estimators)),
        "KNeighborsRegressor": pipe(KNeighborsRegressor()),
        "SVR": pipe(SVR()),
        "MLPRegressor": pipe(MLPRegressor(max_iter=1000)),
    }
    if escludi_lenti:
        modelli = {k: v for k, v in modelli.items() if k not in MODELLI_LENTI}
    return modelli


def _prepare_dataset() -> pd.DataFrame:
    """Dataset del livello 'totale': una riga per giorno, con la media del
    parco (view totale_giornaliero) piu' giorno_num come trend temporale."""

    df = merge_day_type()
    df["giorno_num"] = pd.to_datetime(df["giorno"]).map(lambda x: x.toordinal())
    return df


def _prepare_dataset_reame() -> tuple[pd.DataFrame, list[str], dict]:
    """Dataset del livello 'reame': una riga per (giorno, reame)."""
    df = crud.get_coda_per_reame(crud.engine)
    if df.empty:
        return df, BASE_FEATURES, {"reami": []}

    df = _aggiungi_feature_calendario(df)
    reami = sorted(df["categoria"].dropna().unique())
    df, colonne_dummy = _dummies_entita(df, "categoria", "reame")
    return df, BASE_FEATURES + colonne_dummy, {"reami": reami}


def _prepare_dataset_giostra() -> tuple[pd.DataFrame, list[str], dict]:
    """Dataset del livello 'giostra': una riga per (giorno, giostra). Il reame
    resta tra le feature, cosi' le giostre con poco storico ereditano comunque
    il comportamento medio della loro area."""
    df = crud.get_coda_per_giostra(crud.engine)
    if df.empty:
        return df, BASE_FEATURES, {"giostre": []}

    df = _aggiungi_feature_calendario(df)
    giostre = (
        df[["nome", "categoria"]].drop_duplicates()
        .sort_values(["categoria", "nome"]).to_dict("records")
    )
    df, dummy_reame = _dummies_entita(df, "categoria", "reame")
    df, dummy_giostra = _dummies_entita(df, "nome", "giostra")
    return df, BASE_FEATURES + dummy_reame + dummy_giostra, {"giostre": giostre}


def confronta_modelli(df: pd.DataFrame, feature_columns: list[str] | None = None) -> list[dict]:
    """Allena e valuta tutti i modelli, ordinati per R^2 decrescente.
    Ritorna una lista di dict pronta per essere serializzata in JSON."""
    feature_columns = feature_columns or FEATURE_COLUMNS
    X = df[feature_columns]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    escludi_lenti = len(df) > SOGLIA_RIGHE_MODELLI_LENTI
    if escludi_lenti:
        logger.info(
            "Dataset con %d righe: salto i modelli lenti (%s).",
            len(df), ", ".join(sorted(MODELLI_LENTI)),
        )

    risultati = []
    for nome, modello in _build_models(escludi_lenti).items():
        try:
            modello.fit(X_train, y_train)
            y_pred = modello.predict(X_test)
            risultati.append({
                "nome": nome,
                "r2": r2_score(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "rmse": root_mean_squared_error(y_test, y_pred),
                "pipeline": modello,
            })
        except Exception:
            logger.exception("Modello %s fallito durante il training, salto.", nome)

    risultati.sort(key=lambda r: r["r2"], reverse=True)
    return risultati


def _assicura_colonna_livello(engine) -> None:
    """model_runs nasceva con il solo livello 'totale': aggiunge la colonna
    anche ai database creati prima dei tre livelli."""
    with engine.connect() as conn:
        conn.execute(sqlalchemy.text(
            "ALTER TABLE public.model_runs ADD COLUMN IF NOT EXISTS livello text"
        ))
        conn.commit()


def salva_model_run(risultati: list[dict], n_training_rows: int, engine,
                    livello: str = "totale") -> None:
    migliore = risultati[0]

    metrics_json = json.dumps([
        {"nome": r["nome"], "r2": r["r2"], "mae": r["mae"], "rmse": r["rmse"]}
        for r in risultati
    ])

    _assicura_colonna_livello(engine)
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("""
                INSERT INTO model_runs
                    (run_at, livello, best_model, best_mae, best_rmse, best_r2,
                     n_training_rows, metrics_json)
                VALUES
                    (:run_at, :livello, :best_model, :best_mae, :best_rmse, :best_r2,
                     :n_training_rows, :metrics_json)
            """),
            {
                "run_at": datetime.now(timezone.utc),
                "livello": livello,
                "best_model": migliore["nome"],
                "best_mae": migliore["mae"],
                "best_rmse": migliore["rmse"],
                "best_r2": migliore["r2"],
                "n_training_rows": n_training_rows,
                "metrics_json": metrics_json,
            },
        )
        conn.commit()


def _train_livello(livello: str, df: pd.DataFrame, feature_columns: list[str],
                   meta: dict) -> dict | None:
    """Confronta i modelli su un singolo livello, salva il migliore su disco
    (insieme alle sue feature e ai suoi metadati) e logga le metriche."""
    logger.info("[%s] Dataset di training: %d righe.", livello, len(df))

    if len(df) < 10:
        logger.warning(
            "[%s] Solo %d righe disponibili: il training procede ma i risultati "
            "vanno presi con cautela finche' non c'e' piu' storico.", livello, len(df)
        )

    risultati = confronta_modelli(df, feature_columns)
    if not risultati:
        logger.error("[%s] Nessun modello allenato con successo.", livello)
        return None

    migliore = risultati[0]
    logger.info(
        "[%s] Modello migliore: %s (R2=%.3f, MAE=%.3f, RMSE=%.3f)",
        livello, migliore["nome"], migliore["r2"], migliore["mae"], migliore["rmse"],
    )

    bundle = {
        "livello": livello,
        "pipeline": migliore["pipeline"],
        "feature_columns": feature_columns,
        "nome_modello": migliore["nome"],
        "metriche": {k: migliore[k] for k in ("r2", "mae", "rmse")},
        "n_training_rows": len(df),
        "run_at": datetime.now(timezone.utc).isoformat(),
        **meta,
    }
    joblib.dump(bundle, MODEL_PATHS[livello])
    logger.info("[%s] Modello salvato in %s", livello, MODEL_PATHS[livello])

    salva_model_run(risultati, len(df), crud.engine, livello)
    logger.info("[%s] Metriche salvate in model_runs.", livello)
    return bundle


def train():
    """Allena i tre livelli di previsione: totale parco, per reame, per giostra."""
    datasets = {
        "totale": (_prepare_dataset(), BASE_FEATURES, {}),
        "reame": _prepare_dataset_reame(),
        "giostra": _prepare_dataset_giostra(),
    }

    for livello, (df, feature_columns, meta) in datasets.items():
        if df.empty:
            logger.warning("[%s] Nessun dato disponibile, livello saltato.", livello)
            continue
        _train_livello(livello, df, feature_columns, meta)


def build_feature_row(giorno, media_tmp: float, precip_tmp: float,
                      feature_columns: list[str] | None = None,
                      dummy_attive: tuple[str, ...] = ()) -> pd.DataFrame:
    """Costruisce una riga di feature per una singola data, nello stesso
    formato/ordine usato in training. dummy_attive elenca le colonne one-hot
    da accendere (es. "reame_Adrenaline", "giostra_Blue Tornado")."""

    feature_columns = feature_columns or FEATURE_COLUMNS
    tipo = day_type(giorno)
    row = {colonna: False for colonna in feature_columns}
    row.update({
        "media_tmp": media_tmp,
        "precip_tmp": precip_tmp,
        "alta_stagione": e_alta_stagione(giorno),
        "giorno_num": pd.Timestamp(giorno).toordinal(),
    })
    for t in DayType:
        row[f"tipo_{t.value}"] = (t == tipo)
    for colonna in dummy_attive:
        if colonna in row:
            row[colonna] = True
    return pd.DataFrame([row])[feature_columns]


if __name__ == "__main__":
    train()
