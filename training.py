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
 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
 
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "best_model.joblib"
from ingestion import crud
from working_day import DayType, day_type, e_alta_stagione 
import pandas as pd


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

FEATURE_COLUMNS = [
    "media_tmp",
    "precip_tmp",
    "alta_stagione",
    "giorno_num",
] + [f"tipo_{t.value}" for t in DayType]
 
TARGET_COLUMN = "coda"
 
 
def _build_models() -> dict[str, Pipeline]:
    """Stesso identico dizionario di modelli di check_modello() nel vecchio
    main.py, solo spostato qui in una funzione a se' stante."""
    base_estimators = [
        ("lr", LinearRegression()),
        ("rf", RandomForestRegressor()),
        ("gb", GradientBoostingRegressor()),
    ]
 
    def pipe(model):
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
 
    return {
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
 
 
def _prepare_dataset() -> pd.DataFrame:
    """Prende l'output di merge_day_type() (features.py) e aggiunge
    giorno_num (ordinale, per catturare un trend temporale)."""
 
    df = merge_day_type()
    df["giorno_num"] = pd.to_datetime(df["giorno"]).map(lambda x: x.toordinal())
    return df
 
 
def confronta_modelli(df: pd.DataFrame) -> list[dict]:
    """Allena e valuta tutti i modelli, ordinati per R^2 decrescente.
    Ritorna una lista di dict pronta per essere serializzata in JSON."""
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
 
    risultati = []
    for nome, modello in _build_models().items():
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
 
 
def salva_model_run(risultati: list[dict], n_training_rows: int, engine) -> None:
    migliore = risultati[0]
 
    metrics_json = json.dumps([
        {"nome": r["nome"], "r2": r["r2"], "mae": r["mae"], "rmse": r["rmse"]}
        for r in risultati
    ])
 
    with engine.connect() as conn:
        conn.execute(
            sqlalchemy.text("""
                INSERT INTO model_runs
                    (run_at, best_model, best_mae, best_rmse, best_r2,
                     n_training_rows, metrics_json)
                VALUES
                    (:run_at, :best_model, :best_mae, :best_rmse, :best_r2,
                     :n_training_rows, :metrics_json)
            """),
            {
                "run_at": datetime.now(timezone.utc),
                "best_model": migliore["nome"],
                "best_mae": migliore["mae"],
                "best_rmse": migliore["rmse"],
                "best_r2": migliore["r2"],
                "n_training_rows": n_training_rows,
                "metrics_json": metrics_json,
            },
        )
        conn.commit()
 
 
def train():
    df = _prepare_dataset()
    logger.info("Dataset di training: %d righe.", len(df))
 
    if len(df) < 10:
        logger.warning(
            "Solo %d righe disponibili: il training procede ma i risultati "
            "vanno presi con cautela finche' non c'e' piu' storico.", len(df)
        )
 
    risultati = confronta_modelli(df)
    migliore = risultati[0]
    logger.info(
        "Modello migliore: %s (R2=%.3f, MAE=%.3f, RMSE=%.3f)",
        migliore["nome"], migliore["r2"], migliore["mae"], migliore["rmse"],
    )
 
    joblib.dump(migliore["pipeline"], MODEL_PATH)
    logger.info("Modello salvato in %s", MODEL_PATH)
 
    salva_model_run(risultati, len(df), crud.engine)
    logger.info("Metriche salvate in model_runs.")
 
 
if __name__ == "__main__":
    train()
