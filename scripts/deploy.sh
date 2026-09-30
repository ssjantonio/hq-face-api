#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/var/www/hq-face-api"
HEALTH_URL="http://127.0.0.1:8090/health"

echo "========================================"
echo " Deploy hq-face-api"
echo "========================================"

cd "$APP_DIR"

echo
echo "==> Construyendo y levantando contenedores..."
docker compose up -d --build --remove-orphans

echo
echo "==> Estado de contenedores..."
docker compose ps

echo
echo "==> Esperando health check..."

for i in {1..15}; do
    if curl -fsS "$HEALTH_URL" >/dev/null; then
        echo "==> Health check OK"
        curl -fsS "$HEALTH_URL"
        echo
        echo "==> Deploy finalizado correctamente"
        exit 0
    fi

    echo "Intento $i/15..."
    sleep 2
done

echo
echo "ERROR: El servicio no respondió correctamente al health check."
echo "Últimos logs:"
docker compose logs --tail=100

exit 1