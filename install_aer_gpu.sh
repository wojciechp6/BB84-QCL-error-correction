#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="aer-gpu-main"
DOCKERFILE="Dockerfile.aer-gpu"
OUT_DIR="${OUT_DIR:-$PWD/out_wheels}"
AER_REF="${AER_REF:-main}"

mkdir -p "$OUT_DIR"

echo "=== Building Docker image: ${IMAGE_NAME} (AER_REF=${AER_REF}) ==="
docker build \
  --build-arg AER_REF="${AER_REF}" \
  -f "${DOCKERFILE}" \
  -t "${IMAGE_NAME}" .

echo "=== Extracting wheel(s) to: ${OUT_DIR} ==="
docker run --rm \
  -v "${OUT_DIR}:/host_out" \
  "${IMAGE_NAME}" \
  bash -lc "cp -v /out/* /host_out/ && ls -lh /host_out"

echo "=== DONE ==="
echo "Wheels saved in: ${OUT_DIR}"
