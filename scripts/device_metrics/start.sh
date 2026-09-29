#!/usr/bin/env bash
# Keep ROCm Device Metrics Exporter up and let Prometheus store the series.
# https://github.com/ROCm/device-metrics-exporter
# The retired amd/amd_smi_exporter is not used.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Published install: https://instinct.docs.amd.com/projects/device-metrics-exporter/en/latest/installation/docker.html
EXPORTER_IMAGE="${EXPORTER_IMAGE:-rocm/device-metrics-exporter:v1.5.2}"
PROMETHEUS_IMAGE="${PROMETHEUS_IMAGE:-prom/prometheus:v3.5.0}"
DATA_DIR="${DATA_DIR:-${ROOT}/../../_results/device_metrics/prometheus}"

mkdir -p "${DATA_DIR}"

docker pull "${EXPORTER_IMAGE}"
docker pull "${PROMETHEUS_IMAGE}"

docker rm -f device-metrics-exporter >/dev/null 2>&1 || true
docker run -d \
  --name device-metrics-exporter \
  --restart unless-stopped \
  --privileged \
  --device=/dev/dri \
  --device=/dev/kfd \
  -v /sys:/sys:ro \
  -p 5000:5000 \
  "${EXPORTER_IMAGE}"

docker rm -f gpu-metrics-prometheus >/dev/null 2>&1 || true
docker run -d \
  --name gpu-metrics-prometheus \
  --restart unless-stopped \
  --network host \
  -v "${ROOT}/prometheus.yml:/etc/prometheus/prometheus.yml:ro" \
  -v "${DATA_DIR}:/prometheus" \
  "${PROMETHEUS_IMAGE}" \
  --config.file=/etc/prometheus/prometheus.yml \
  --storage.tsdb.path=/prometheus \
  --storage.tsdb.retention.time=15d \
  --web.listen-address=127.0.0.1:9090

echo "exporter  http://127.0.0.1:5000/metrics"
echo "prometheus http://127.0.0.1:9090"
echo "tsdb      ${DATA_DIR}"
