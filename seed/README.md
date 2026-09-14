# Seed del database

`demo_data.sql.gz` contiene uno storico reale di **87 giorni** (460.522 rilevazioni delle giostre e
280 rilevazioni meteo, maggio 2025 → settembre 2026): serve a far partire il progetto con dati veri
senza aspettare settimane di raccolta.

Viene caricato **automaticamente** da `docker compose up`, ma solo alla creazione del volume
PostgreSQL — cioè al primo avvio su una macchina nuova. Chi ha già un database con i propri dati non
viene toccato: l'entrypoint di Postgres esegue `/docker-entrypoint-initdb.d/` solo su un data
directory vuoto.

Contiene solo le tabelle `giostre` e `meteo`. Le view di aggregazione arrivano da `db/schema.sql`
(caricato prima, come `01_schema.sql`) e `model_runs` resta vuota finché non lanci il training.

## Ripartire da zero

```bash
docker compose down -v      # elimina il volume: i dati vanno persi
docker compose up -d --build
```

## Usare un dump tuo

Sostituisci il file, oppure mettine un altro accanto e aggiungi il mount in `docker-compose.yml`:
l'entrypoint esegue i file in ordine alfabetico e accetta `.sql`, `.sql.gz` e `.sh`.

Per rigenerare il dump dalla tua istanza:

```bash
docker exec gardaland-db pg_dump -U postgres -d gardaland --data-only \
  --table=public.giostre --table=public.meteo | gzip -9 > seed/demo_data.sql.gz
```

I file `*.sql` non compressi in questa cartella sono ignorati da git.
