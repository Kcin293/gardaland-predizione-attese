import enum

import pandas as pd
import holidays

italy_holidays = holidays.IT()


class DayType(enum.Enum):
    FESTIVO = "festivo"
    WEEKEND = "weekend"
    LAVORATIVO = "lavorativo"
    PONTE = "ponte"


def e_non_lavorativo(giorno: pd.Timestamp) -> bool:
    return giorno in italy_holidays or giorno.weekday() >= 5


def e_giorno_ponte_singolo(giorno: pd.Timestamp) -> bool:
    """Giorno lavorativo incastrato tra due giorni non lavorativi (es. venerdì
    tra un giovedì festivo e il weekend)."""
    if e_non_lavorativo(giorno):
        return False
    ieri = giorno - pd.Timedelta(days=1)
    domani = giorno + pd.Timedelta(days=1)
    return e_non_lavorativo(ieri) and e_non_lavorativo(domani)


def e_non_lavorativo_esteso(giorno: pd.Timestamp) -> bool:
    return e_non_lavorativo(giorno) or e_giorno_ponte_singolo(giorno)


def num_giorni_blocco(start_date: pd.Timestamp) -> int:
    start_date = pd.to_datetime(start_date)
    if not e_non_lavorativo_esteso(start_date):
        return 0

    giorni = 1
    prima = start_date - pd.Timedelta(days=1)
    while e_non_lavorativo_esteso(prima):
        giorni += 1
        prima -= pd.Timedelta(days=1)

    dopo = start_date + pd.Timedelta(days=1)
    while e_non_lavorativo_esteso(dopo):
        giorni += 1
        dopo += pd.Timedelta(days=1)

    return giorni


def day_type(date: pd.Timestamp) -> DayType:
    date = pd.to_datetime(date)
    is_festivo = date in italy_holidays
    is_weekend = date.weekday() >= 5

    if num_giorni_blocco(date) >= 3:
        return DayType.PONTE
    elif is_festivo:
        return DayType.FESTIVO
    elif is_weekend:
        return DayType.WEEKEND
    else:
        return DayType.LAVORATIVO


def e_alta_stagione(giorno: pd.Timestamp) -> bool:
    '''Alta stagione in estate o evento natalizio'''
    giorno = pd.to_datetime(giorno)
    return giorno.month in [1, 6, 7, 8, 12]