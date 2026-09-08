import logging

from ingestion import crud, meteo_client, queue_times_client


logging.basicConfig(level=logging.INFO)   
logger = logging.getLogger(__name__)   

def main():
    giostre = queue_times_client.get_queue_times()
    if giostre:
        logger.info(f"Recuperate {len(giostre)} giostre dal sito queue-times.com.")
        chiuse = sum(1 for g in giostre if not g['is_open'])
        if chiuse == len(giostre):
            logger.warning("Tutte le giostre risultano chiuse. Parco probabilmente chiuso.")
        else:
            logger.info(f"Giostre aperte: {len(giostre) - chiuse}, Giostre chiuse: {chiuse}")
            crud.salva_giostre(giostre, crud.engine)
            logger.info("Dati delle giostre salvati con successo nel database.")
            crud.print_giostre_count(crud.engine)
            meteo = meteo_client.get_meteo()
            if meteo:
                logger.info(f"Dati meteo: {meteo}")
                crud.salva_meteo(meteo, crud.engine)
                logger.info("Dati meteo salvati con successo nel database.")
                crud.print_meteo_count(crud.engine)
            else:
                logger.error("Errore nel recupero dei dati meteo.")
    else:
        logger.error("Errore nel recupero dei dati delle giostre.")
   

if __name__ == "__main__":
    main()