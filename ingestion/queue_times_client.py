import requests
import logging

logging.basicConfig(level=logging.INFO)   
logger = logging.getLogger(__name__)   


url = "https://queue-times.com/parks/12/queue_times.json"



def get_queue_times():
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            risposta = response.json()
            return get_giostre(risposta)
    except requests.RequestException as e:
        logger.error(f"Error fetching queue times: {e}")
    return None


def get_giostre(data):
    giostre = get_rides(data)
    for land in data['lands']:
        giostre += get_rides(land, land['name'])
    return giostre

def get_rides(data, categoria=None):
    rides = []
    for ride in data['rides']:
        rides.append({
            'id': ride['id'],
            'nome': ride['name'],
            'is_open': ride['is_open'],
            'wait_time': ride['wait_time'],
            'last_updated': ride['last_updated'],
            'categoria': categoria if categoria else "Sconosciuta"  # Placeholder
        })
    return rides
