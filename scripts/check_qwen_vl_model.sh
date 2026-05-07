#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="/home/ubuntu/models/Qwen2.5-VL-3B-Instruct"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model_dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1"
      exit 1
      ;;
  esac
done

if [[ ! -d "${MODEL_DIR}" ]]; then
  echo "ERROR: model directory does not exist: ${MODEL_DIR}"
  exit 1
fi

echo "[vla_world_grasp] Model directory: ${MODEL_DIR}"
du -sh "${MODEL_DIR}" || true
ls -lh "${MODEL_DIR}" || true

missing_core=0
for file in \
  config.json \
  model.safetensors.index.json \
  model-00001-of-00002.safetensors \
  model-00002-of-00002.safetensors; do
  if [[ ! -f "${MODEL_DIR}/${file}" ]]; then
    echo "ERROR: missing core model file: ${file}"
    missing_core=1
  else
    echo "[vla_world_grasp] found: ${file}"
  fi
done

if [[ "${missing_core}" != "0" ]]; then
  exit 1
fi

if [[ ! -f "${MODEL_DIR}/tokenizer_config.json" ]]; then
  echo "[vla_world_grasp] WARNING: tokenizer_config.json is missing."
else
  echo "[vla_world_grasp] found: tokenizer_config.json"
fi

if [[ ! -f "${MODEL_DIR}/tokenizer.json" && ! -f "${MODEL_DIR}/vocab.json" ]]; then
  echo "[vla_world_grasp] WARNING: tokenizer.json/vocab.json is missing."
else
  echo "[vla_world_grasp] found tokenizer vocabulary file."
fi

if [[ ! -f "${MODEL_DIR}/preprocessor_config.json" && ! -f "${MODEL_DIR}/processor_config.json" ]]; then
  echo "[vla_world_grasp] WARNING: preprocessor_config.json/processor_config.json is missing."
else
  echo "[vla_world_grasp] found processor config file."
fi

MODEL_DIR_FOR_PY="${MODEL_DIR}" python - <<'PY'
from pathlib import Path
import os
import sys

try:
    import torch
    from transformers import AutoProcessor

    model_dir = Path(os.environ["MODEL_DIR_FOR_PY"])

    processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
    print("[vla_world_grasp] AutoProcessor loaded.")

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration
        cls = Qwen2_5_VLForConditionalGeneration
    except Exception:
        from transformers import AutoModelForVision2Seq
        cls = AutoModelForVision2Seq

    model = cls.from_pretrained(
        str(model_dir),
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print("[vla_world_grasp] Qwen2.5-VL model loaded successfully.")
except Exception as exc:
    print(f"[vla_world_grasp] Python model load failed: {exc}", file=sys.stderr)
    print(
        "[vla_world_grasp] You may need: pip install -U transformers accelerate safetensors pillow qwen-vl-utils",
        file=sys.stderr,
    )
    raise
PY
