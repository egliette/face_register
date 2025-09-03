#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR%/scripts}"

COMPOSE_FILE="${REPO_ROOT}/docker-compose.dev.yml"
NETWORK_NAME="person_detection"

REBUILD=false
ATTACH_SHELL=false
NO_COMMAND=false

usage() {
  echo "Usage: $(basename "$0") [--rebuild|-r] [--shell|-s] [--no-command|-n] [--help|-h]" >&2
  echo "  --rebuild, -r     Rebuild app image before starting" >&2
  echo "  --shell, -s       Attach to app container shell instead of running the app" >&2
  echo "  --no-command, -n  Don't execute the default command (useful with --shell)" >&2
  echo "  --help, -h        Show this help message" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  $(basename "$0")                    # Start dev stack normally" >&2
  echo "  $(basename "$0") --shell            # Start stack and attach to app shell" >&2
  echo "  $(basename "$0") --rebuild --shell  # Rebuild app and attach to shell" >&2
  echo "  $(basename "$0") --no-command       # Start stack without running app command" >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rebuild|-r)
      REBUILD=true
      shift
      ;;
    --shell|-s)
      ATTACH_SHELL=true
      shift
      ;;
    --no-command|-n)
      NO_COMMAND=true
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

echo "Bringing down existing dev stack..."
docker-compose -f "$COMPOSE_FILE" down -v || true

echo "Ensuring network '$NETWORK_NAME' exists..."
docker network create "$NETWORK_NAME" >/dev/null 2>&1 || true

if [[ "$REBUILD" == "true" ]]; then
  echo "Rebuilding app image..."
  docker-compose -f "$COMPOSE_FILE" build app
fi

echo "Starting dependencies (postgres, minio, qdrant)..."
docker-compose -f "$COMPOSE_FILE" up -d postgres minio qdrant

echo "Running database migrations..."
docker-compose -f "$COMPOSE_FILE" run --rm db_migrate

if [[ "$ATTACH_SHELL" == "true" ]]; then
  echo "Starting app container and attaching to shell..."
  # Ensure app is running and stays alive, then attach a shell
  docker-compose -f "$COMPOSE_FILE" up -d app
  docker exec -it face_register_app_dev /bin/bash
else
  if [[ "$NO_COMMAND" == "true" ]]; then
    echo "Starting app container without command..."
    docker-compose -f "$COMPOSE_FILE" up -d app
    echo "App container started. Use 'docker exec -it face_register_app_dev /bin/bash' to attach to shell"
  else
    echo "Starting dev stack with app..."
    docker-compose -f "$COMPOSE_FILE" up app
  fi
fi

echo "Done."
