#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"
LOCAL_DIR="/home/ubuntu/models/Qwen2.5-VL-3B-Instruct"
MAX_RETRIES=10
SLEEP_SEC=20
USE_MIRROR=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_id)
      MODEL_ID="$2"
      shift 2
      ;;
    --local_dir)
      LOCAL_DIR="$2"
      shift 2
      ;;
    --max_retries)
      MAX_RETRIES="$2"
      shift 2
      ;;
    --sleep_sec)
      SLEEP_SEC="$2"
      shift 2
      ;;
    --use_mirror)
      USE_MIRROR=true
      shift
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

echo "[vla_world_grasp] Python / Hugging Face CLI environment:"
which python || true
python --version || true
which huggingface-cli || true
which hf || true

export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=60
export HF_HUB_ETAG_TIMEOUT=60

if [[ "${USE_MIRROR}" == "true" ]]; then
  export HF_ENDPOINT="https://hf-mirror.com"
  echo "[vla_world_grasp] Using HF_ENDPOINT=${HF_ENDPOINT}"
fi

mkdir -p "${LOCAL_DIR}"

if command -v huggingface-cli >/dev/null 2>&1; then
  DOWNLOAD_CMD=(huggingface-cli download "${MODEL_ID}" --local-dir "${LOCAL_DIR}" --max-workers 1)
elif command -v hf >/dev/null 2>&1; then
  DOWNLOAD_CMD=(hf download "${MODEL_ID}" --local-dir "${LOCAL_DIR}" --max-workers 1)
else
  echo "ERROR: neither huggingface-cli nor hf was found in PATH."
  echo "Install huggingface_hub first, for example: pip install -U huggingface_hub"
  exit 1
fi

success=false
for ((i = 1; i <= MAX_RETRIES; i++)); do
  echo "[vla_world_grasp] Download attempt ${i}/${MAX_RETRIES}"
  echo "[vla_world_grasp] Command: ${DOWNLOAD_CMD[*]}"
  if "${DOWNLOAD_CMD[@]}"; then
    success=true
    break
  fi
  echo "[vla_world_grasp] Download failed, will retry."
  echo "[vla_world_grasp] Existing files:"
  ls -lh "${LOCAL_DIR}" || true
  if [[ "${i}" -lt "${MAX_RETRIES}" ]]; then
    sleep "${SLEEP_SEC}"
  fi
done

if [[ "${success}" != "true" ]]; then
  echo "[vla_world_grasp] Qwen model download failed after retries."
  echo "[vla_world_grasp] Try again with --use_mirror or check network."
  exit 1
fi

echo "[vla_world_grasp] Qwen model download completed."
bash "${ROOT_DIR}/scripts/check_qwen_vl_model.sh" --model_dir "${LOCAL_DIR}"
