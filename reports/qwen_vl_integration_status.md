# Qwen2.5-VL Integration Status

## Scope

This report verifies the minimal local image inference path for `Qwen/Qwen2.5-VL-3B-Instruct`.

No Isaac Sim startup logic, Isaac scene code, robot control, renderer, baseline grasp logic, data collection, `execute_grasp`, or `scripts/run_demo.sh` was modified.

## Environment

- Python: `/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python`
- Torch: `2.7.0+cu128`
- `torch.cuda.is_available()`: `True`
- CUDA device: `NVIDIA GeForce RTX 4090`
- Transformers: `4.55.4`
- Accelerate: `1.13.0`
- Model directory: `/home/ubuntu/models/Qwen2.5-VL-3B-Instruct`
- Model directory size: `7.1G`
- Test image: `outputs/debug/qwen_vl_blue_cylinder_demo_qwen_input.png`
- Test image size: `640x480 RGB`
- Chinese instruction: `抓起红色方块`

## Model Files Observed

The local model directory contains the expected core files:

- `config.json`
- `model.safetensors.index.json`
- `model-00001-of-00002.safetensors`
- `model-00002-of-00002.safetensors`
- `tokenizer_config.json`
- `tokenizer.json`
- `vocab.json`
- `preprocessor_config.json`

## CUDA Check Command

```bash
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python - <<'PY'
import torch
import transformers
import accelerate
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
print(transformers.__version__)
print(accelerate.__version__)
PY
```

Key output:

```text
torch 2.7.0+cu128
cuda_available True
cuda_device_name NVIDIA GeForce RTX 4090
transformers 4.55.4
accelerate 1.13.0
```

## Minimal Image Inference Command

The first run loaded the model successfully but the model wrapped JSON in a Markdown code fence. A stricter Chinese prompt was then used requiring the first character to be `{` and forbidding Markdown.

```bash
/home/ubuntu/miniconda3/envs/env_isaaclab/bin/python - <<'PY'
from pathlib import Path
import json
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from PIL import Image

model_dir = Path('/home/ubuntu/models/Qwen2.5-VL-3B-Instruct')
image_path = Path('outputs/debug/qwen_vl_blue_cylinder_demo_qwen_input.png').resolve()
instruction = '抓起红色方块'

print(f'[vla_world_grasp] torch.cuda.is_available={torch.cuda.is_available()}')
processor = AutoProcessor.from_pretrained(str(model_dir), trust_remote_code=True)
print('[vla_world_grasp] AutoProcessor loaded.')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    str(model_dir),
    torch_dtype=torch.float16,
    device_map='auto',
    trust_remote_code=True,
)
print('[vla_world_grasp] Qwen2.5-VL model loaded.')

prompt = f'''你是机器人抓取系统中的视觉语言目标定位模块。
根据图像和中文指令找出要抓取的目标。

用户指令：{instruction}

只允许输出一个 JSON 对象。禁止 Markdown，禁止 ```，禁止解释。
输出的第一个字符必须是 {{，最后一个字符必须是 }}。
必须包含这四个键：target_object、color、shape、grasp_hint。
示例格式：{{"target_object":"red cube","color":"red","shape":"cube","grasp_hint":"top-down grasp at object center"}}'''
messages = [{'role': 'user', 'content': [{'type': 'image', 'image': str(image_path)}, {'type': 'text', 'text': prompt}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image = Image.open(image_path).convert('RGB')
inputs = processor(text=[text], images=[image], padding=True, return_tensors='pt').to(model.device)
with torch.inference_mode():
    generated_ids = model.generate(**inputs, max_new_tokens=96, do_sample=False)
trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(output)
parsed = json.loads(output.strip())
print(json.dumps(parsed, ensure_ascii=False, sort_keys=True))
PY
```

## Model Output

```json
{"target_object":"red cube","color":"red","shape":"cube","grasp_hint":"top-down grasp at object center"}
```

Parsed JSON:

```json
{"color": "red", "grasp_hint": "top-down grasp at object center", "shape": "cube", "target_object": "red cube"}
```

## Status

Success.

- `torch.cuda.is_available()` returned `True`.
- `AutoProcessor.from_pretrained(...)` loaded successfully.
- `Qwen2_5_VLForConditionalGeneration.from_pretrained(...)` loaded successfully.
- The model accepted an existing Isaac scene RGB image and the Chinese instruction `抓起红色方块`.
- The final strict-prompt run produced valid JSON with `target_object`, `color`, `shape`, and `grasp_hint`.

## Notes

- `qwen_vl_utils` was not installed in the environment during the first trial, so the test used the PIL image path through `AutoProcessor` directly.
- Transformers printed warnings about fast image processor defaults and deprecated video processor config. These warnings did not block image inference.
