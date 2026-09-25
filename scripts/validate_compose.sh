#!/bin/bash
# Render every Compose file to catch invalid paths and interpolation errors.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
export HOME="${HOME:-/tmp}"
fail=0
for f in docker/docker-compose.yml docker/docker-compose.*.yml; do
    echo "docker compose -f $f config"
    if ! docker compose -f "$f" config >/dev/null; then
        echo "FAIL: $f" >&2
        fail=1
    fi
done
exit "$fail"
