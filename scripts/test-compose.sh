#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"

COMPOSE_FILE="${REPO_ROOT}/docker-compose.test.yml"
NETWORK_NAME="person_detection"

REBUILD=false
KEEP_RUNNING=false

usage() {
  echo "Usage: $(basename "$0") [--rebuild|-r] [--detach|-d] [--keep-running|-k]" >&2
  echo "  --rebuild, -r      Rebuild app_test image before starting" >&2
  echo "  --detach, -d       Run docker-compose detached (-d)" >&2
  echo "  --keep-running, -k Keep test stack running after tests complete" >&2
}

DETACH_FLAG=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild|-r)
      REBUILD=true
      shift
      ;;
    --detach|-d)
      DETACH_FLAG="-d"
      shift
      ;;
    --keep-running|-k)
      KEEP_RUNNING=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "Compose file not found: $COMPOSE_FILE" >&2
  exit 1
fi

echo "Bringing down existing test stack..."
docker-compose -f "$COMPOSE_FILE" down -v || true

echo "Ensuring network '$NETWORK_NAME' exists..."
docker network create "$NETWORK_NAME" >/dev/null 2>&1 || true

if [[ "$REBUILD" == "true" ]]; then
  echo "Rebuilding app_test image..."
  docker-compose -f "$COMPOSE_FILE" build app_test
fi

echo "Starting dependencies (postgres, minio)..."
docker-compose -f "$COMPOSE_FILE" up -d postgres minio

echo "Running database migrations..."
docker-compose -f "$COMPOSE_FILE" run --rm db_migrate

echo "Running tests..."
docker-compose -f "$COMPOSE_FILE" up --exit-code-from app_test app_test

if [[ "$KEEP_RUNNING" == "true" ]]; then
  echo "Keeping test stack running. Use 'docker-compose -f $COMPOSE_FILE down -v' to clean up manually."
else
  echo "Cleaning up test stack..."
  docker-compose -f "$COMPOSE_FILE" down -v
fi

echo "Done."
