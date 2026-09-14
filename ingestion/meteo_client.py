
import requests
import logging
import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

logging.basicConfig(level=logging.INFO)   
logger = logging.getLogger(__name__)   

cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
openmeteo = openmeteo_requests.Client(session = retry_session)

url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 45.4544, "longitude": 10.7139, "timezone": "auto",
    "current": ["apparent_temperature", "precipitation"],
	"forecast_days": 1,
}


def get_meteo():
    try:
        responses = openmeteo.weather_api(url, params = params)
        current = responses[0].Current()
        current_apparent_temperature = current.Variables(0).Value()
        current_precipitation = current.Variables(1).Value()
        return {
            "temperatura_percepita": current_apparent_temperature,
            "precipitazioni": current_precipitation,
            "data_ora": pd.to_datetime(current.Time(), unit="s", utc=True)
        }   

    except openmeteo_requests.OpenMeteoRequestError as e:
        logger.error(f"Error fetching queue times: {e}")
        return None

def get_previsione_domani():
    """Media giornaliera prevista di temperatura percepita e precipitazioni
    per domani, calcolata dalle previsioni orarie di Open-Meteo — stessa
    logica di aggregazione (AVG) usata dalla vista totale_giornaliero."""
    params = {
        "latitude": 45.4544, "longitude": 10.7139, "timezone": "auto",
        "hourly": ["apparent_temperature", "precipitation"],
        "forecast_days": 2,  # oggi + domani
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        hourly = responses[0].Hourly()
        temp = hourly.Variables(0).ValuesAsNumpy()
        precip = hourly.Variables(1).ValuesAsNumpy()

        times = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
        df = pd.DataFrame({"data_ora": times, "temp": temp, "precip": precip})
        df["giorno"] = df["data_ora"].dt.date

        domani = (pd.Timestamp.now(tz="Europe/Rome") + pd.Timedelta(days=1)).date()
        riga = df[df["giorno"] == domani]

        return {
            "giorno": domani,
            "media_tmp": riga["temp"].mean(),
            "precip_tmp": riga["precip"].mean(),
        }
    except Exception as e:
        logger.error(f"Errore nel recupero previsione: {e}")
        return None