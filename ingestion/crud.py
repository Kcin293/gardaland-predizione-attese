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