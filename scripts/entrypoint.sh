#!/usr/bin/env bash
# scripts/entrypoint.sh
# =======================
# Entrypoint chung cho api, worker, beat containers.
#
# Nhiem vu:
#   1. Cho postgres san sang (chi api va worker can, beat khong query DB truc tiep)
#   2. Chay init_db.py tao tables neu chua co (chi api service set RUN_INIT_DB=true)
#   3. Chuyen qua CMD duoc truyen vao (uvicorn / celery worker / celery beat)
#
# Bien moi truong:
#   WAIT_FOR_DB : true | false  (default: true)
#   RUN_INIT_DB : true | false  (default: false, chi api service set true)
#   DB_HOST     : hostname cua postgres (default: postgres)
#   DB_PORT     : port cua postgres    (default: 5432)

set -e

DB_HOST=${DB_HOST:-postgres}
DB_PORT=${DB_PORT:-5432}
WAIT_FOR_DB=${WAIT_FOR_DB:-true}
RUN_INIT_DB=${RUN_INIT_DB:-false}

# Cho PostgreSQL san sang
if [ "$WAIT_FOR_DB" = "true" ]; then
    echo "[entrypoint] Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT}..."
    max_retries=30
    count=0
    until pg_isready -h "$DB_HOST" -p "$DB_PORT" -q 2>/dev/null; do
        count=$((count + 1))
        if [ "$count" -ge "$max_retries" ]; then
            echo "[entrypoint] ERROR: PostgreSQL not ready after ${max_retries} attempts."
            exit 1
        fi
        echo "[entrypoint] Attempt ${count}/${max_retries} - retrying in 2s..."
        sleep 2
    done
    echo "[entrypoint] PostgreSQL is ready."
fi

# Tao tables neu chua co (chi api service)
if [ "$RUN_INIT_DB" = "true" ]; then
    echo "[entrypoint] Running init_db.py..."
    python3 init_db.py
    echo "[entrypoint] init_db.py done."
fi

# Chuyen qua main process
echo "[entrypoint] Starting: $*"
exec "$@"
