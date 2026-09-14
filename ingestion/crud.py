from ast import stmt
import os

import pandas as pd
import sqlalchemy
from sqlalchemy.dialects.postgresql import insert as pg_insert
from dotenv import load_dotenv
load_dotenv()

engine = sqlalchemy.create_engine(f'postgresql://{os.getenv("POSTGRES_USER")}:{os.getenv("POSTGRES_PASSWORD")}@{os.getenv("POSTGRES_HOST")}:{os.getenv("POSTGRES_PORT")}/{os.getenv("POSTGRES_DB")}')

    
def salva_giostre(giostre: list[dict], engine):
    df = pd.DataFrame(giostre)
    df.to_sql('giostre', engine, if_exists='append', index=False)

def print_giostre_count(engine):
    risultato = pd.read_sql("SELECT count(*) FROM giostre", engine)
    print(f"Numero di giostre nel database: {risultato.iloc[0, 0]}")


def insert_on_conflict_nothing(pd_table, conn, keys, data_iter):
  data = [dict(zip(keys, row)) for row in data_iter]
  sql = sqlalchemy.text(
      f"INSERT INTO {pd_table.name} ({', '.join(keys)}) "
      f"VALUES ({', '.join([f':{k}' for k in keys])}) "
      "ON CONFLICT DO NOTHING"
  )
  conn.execute(sql, data)

def salva_meteo(meteo: dict, engine):
    df = pd.DataFrame([meteo])
    df.to_sql('meteo', engine, if_exists='append', index=False, method=insert_on_conflict_nothing)

def print_meteo_count(engine):
    risultato = pd.read_sql("SELECT count(*) FROM meteo", engine)
    print(f"Numero di dati meteorologici nel database: {risultato.iloc[0, 0]}")


def get_totale_giornaliero(engine):
    risultato = pd.read_sql("SELECT * FROM totale_giornaliero", engine)
    return risultato

def get_coda_per_giostra(engine):
    """Media giornaliera della coda per singola giostra, gia' unita al meteo
    del giorno. Una riga = (giorno, giostra)."""
    query = """
        SELECT c.giorno, c.nome, c.categoria,
               c.media_coda AS coda,
               t.media_tmp, t.precip_tmp
          FROM coda_giornaliera c
          JOIN temp_giornaliera t ON t.giorno = c.giorno
         ORDER BY c.giorno, c.nome
    """
    return pd.read_sql(query, engine)


def get_coda_per_reame(engine):
    """Media giornaliera della coda per reame (la 'land' di queue-times),
    calcolata su tutte le rilevazioni delle giostre aperte di quel reame."""
    query = """
        SELECT g.last_updated::date AS giorno,
               g.categoria,
               avg(g.wait_time) AS coda,
               t.media_tmp, t.precip_tmp
          FROM giostre g
          JOIN temp_giornaliera t ON t.giorno = g.last_updated::date
         WHERE g.is_open IS TRUE
         GROUP BY g.last_updated::date, g.categoria, t.media_tmp, t.precip_tmp
         ORDER BY 1, 2
    """
    return pd.read_sql(query, engine)


def get_elenco_giostre(engine):
    """Giostre viste almeno una volta, con il loro reame."""
    query = """
        SELECT DISTINCT nome, categoria
          FROM giostre
         ORDER BY categoria, nome
    """
    return pd.read_sql(query, engine)
