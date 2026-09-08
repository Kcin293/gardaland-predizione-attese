
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
