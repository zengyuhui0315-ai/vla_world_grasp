# vla_world_model_grasp

基于 Isaac Sim 4.5、IsaacLab、Franka Panda 和视觉语言模型的桌面物体抓取 MVP 项目。

本项目把中文抓取指令、VLA 目标定位、抓取候选生成、候选评分和 Franka 顶部抓取执行串成一个可运行流程。当前支持 `mock` 后端做稳定演示，也预留并集成了 Qwen2.5-VL 后端用于图像目标定位。

## 项目功能

- 支持中文指令，例如 `抓起红色方块`、`抓起蓝色圆柱`、`抓起绿色球`。
- 支持两类 VLA 后端：
  - `mock`：规则/状态驱动的本地后端，适合快速调试和演示。
  - `qwen_vl`：使用 Qwen2.5-VL 对 Isaac Sim 相机图像做目标 grounding。
- 支持两类仿真后端：
  - `isaac`：Isaac Sim / IsaacLab 中的 Franka Panda 桌面抓取场景。
  - `mock`：不启动 Isaac 的轻量后端，适合检查流程、接口和 CI。
- 将 VLA 输出统一封装为 `VLAResult`，再转换为 `GraspCandidate`。
- 支持 `baseline`、`model`、`hybrid` 三种候选评分模式。
- 支持保存候选 JSON、中文解释、评分图、结果 JSON 和演示视频。
- 支持数据采集、评分模型训练、评估和可视化脚本。

## 核心流程

```text
中文抓取指令 + RGB-D / 物体状态
  -> VLA 后端定位目标
  -> VLAResult
  -> GraspCandidate 候选抓取动作
  -> baseline_score / world_model_score / hybrid score
  -> best_candidate
  -> Franka Panda 顶部抓取执行
```

设计边界：VLA 后端只负责目标定位，不直接控制机器人；机器人只执行经过候选生成和评分后选出的 `best_candidate`。

## 目录结构

```text
vla_world_model_grasp/
  configs/                    默认配置
  scripts/                    常用运行、测试、下载、训练脚本
  src/vla_world_grasp/
    grasp/                    抓取动作空间、候选生成、几何工具
    pipeline/                 单次抓取、数据采集、可视化主流程
    sim/                      Isaac/mock 场景、Franka 控制、传感器、录制
    utils/                    数据、投影、视频、可视化工具
    vla/                      mock、Qwen-VL、OpenVLA/SmolVLA 适配接口
    world_model/              抓取评分数据集、模型、训练、评估
  reports/                    实验说明、结果材料和对比报告
  outputs/                    本地运行输出，上传 GitHub 前可按需忽略或清理
```

## 环境依赖

### 推荐系统环境

- 操作系统：Ubuntu 20.04 / 22.04，推荐 Linux GPU 工作站或服务器。
- Python：`>=3.10`。
- GPU：运行 Isaac Sim 和 Qwen2.5-VL 时推荐 NVIDIA GPU。
- CUDA：以 Isaac Sim、PyTorch 和显卡驱动兼容版本为准。
- Git、bash、常见编译/系统工具。

### Isaac 仿真环境

运行 `--sim_backend isaac` 需要提前安装：

- NVIDIA Isaac Sim 4.5
- IsaacLab
- IsaacLab assets / `isaaclab_assets`
- IsaacLab 自带或兼容的 PyTorch、Omniverse / `omni` Python 模块

脚本会自动查找以下位置的 IsaacLab：

```text
./external/IsaacLab
../external/IsaacLab
~/IsaacLab
```

如果 IsaacLab 不在这些路径，请手动设置：

```bash
export ISAACLAB_DIR=/path/to/IsaacLab
```

服务器或无显示环境推荐使用默认 headless 模式：

```bash
export ENABLE_CAMERAS=1
export LIVESTREAM=0
```

### Python 基础依赖

基础依赖写在 `requirements.txt` 和 `pyproject.toml` 中：

```text
PyYAML>=6.0
numpy>=1.23
matplotlib>=3.7
Pillow>=10.0
```

这些依赖用于配置读取、候选计算、评分模型、图像/图表输出和轻量 mock 流程。

### Qwen-VL 可选依赖

只有运行 `--vla_backend qwen_vl` 时才需要安装：

```bash
pip install -U transformers accelerate qwen-vl-utils huggingface_hub
```

根据本机 CUDA 环境安装合适版本的 PyTorch。例如：

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Qwen2.5-VL 默认模型路径：

```text
/home/ubuntu/models/Qwen2.5-VL-3B-Instruct
```

也可以运行时通过 `--qwen_model_path` 指定其他路径。

## 安装方式

克隆仓库后进入项目目录：

```bash
git clone <your-repo-url>
cd vla_world_model_grasp
```

创建 Python 虚拟环境：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
```

安装基础依赖和本项目：

```bash
pip install -r requirements.txt
pip install -e .
```

如果要运行 Isaac 后端，请确认 `ISAACLAB_DIR` 指向可用 IsaacLab；如果只运行 mock 后端，不需要 Isaac Sim。

## 快速开始

### 1. 轻量 mock 流程

不启动 Isaac，适合快速验证代码路径：

```bash
bash scripts/run_demo.sh \
  --sim_backend mock \
  --vla_backend mock \
  --instruction "抓起红色方块"
```

### 2. Isaac 仿真演示

```bash
bash scripts/run_demo.sh \
  --instruction "抓起红色方块" \
  --vla_backend mock \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name red_cube_demo \
  --no_visualize_candidates \
  --debug_attach_on_grasp
```

### 3. 中文交互演示

```bash
bash scripts/run_interactive_demo.sh
```

终端会提示输入：

```text
请输入中文抓取指令，例如：抓起红色方块 / 抓起蓝色圆柱 / 抓起绿色球
```

直接回车会默认使用 `抓起红色方块`。

示例输出：

- `outputs/videos/interactive_red_cube_demo.mp4`
- `outputs/videos/interactive_blue_cylinder_demo.mp4`
- `outputs/videos/interactive_green_sphere_demo.mp4`
- `outputs/logs/*_candidates.json`
- `outputs/logs/*_explanation.txt`
- `outputs/figures/*_candidate_scores.png`

## Qwen2.5-VL 使用方法

下载 Qwen2.5-VL-3B-Instruct：

```bash
bash scripts/download_qwen_vl.sh \
  --local_dir /home/ubuntu/models/Qwen2.5-VL-3B-Instruct
```

如果 Hugging Face 网络不稳定，可以使用镜像：

```bash
bash scripts/download_qwen_vl.sh --use_mirror
```

检查本地模型目录：

```bash
bash scripts/check_qwen_vl_model.sh \
  --model_dir /home/ubuntu/models/Qwen2.5-VL-3B-Instruct
```

运行 Qwen-VL grounding 演示：

```bash
timeout 1200 bash scripts/run_qwen_vl_demo.sh
```

或直接指定参数：

```bash
timeout 900 bash scripts/run_demo.sh \
  --instruction "抓起蓝色圆柱" \
  --vla_backend qwen_vl \
  --qwen_model_path /home/ubuntu/models/Qwen2.5-VL-3B-Instruct \
  --qwen_device cuda \
  --sim_backend isaac \
  --headless \
  --save_video \
  --video_name qwen_vl_blue_cylinder_demo \
  --no_visualize_candidates \
  --debug_attach_on_grasp \
  --qwen_fallback_to_mock true
```

如果 Qwen-VL 依赖、权重或推理失败，系统会按设计回退到 `mock` 后端，并在结果中记录 `fallback_used=true`。

## 常用脚本

接口和单元式检查：

```bash
bash scripts/test_vla_interfaces.sh
bash scripts/test_chinese_instruction_parser.sh
bash scripts/test_chinese_vla.sh
bash scripts/test_grasp_candidates.sh
bash scripts/test_qwen_vl_grounding.sh
```

仿真 smoke test：

```bash
bash scripts/run_sim_smoke_test.sh
```

录制演示：

```bash
bash scripts/record_demo.sh
```

采集抓取评分数据：

```bash
bash scripts/collect_dataset.sh \
  --num_episodes 100 \
  --candidates_per_episode 16 \
  --output_dir data/grasp_success
```

训练评分模型：

```bash
bash scripts/train_score_model.sh \
  --dataset_csv data/grasp_success/processed/grasp_success_dataset.csv \
  --epochs 50 \
  --output_dir outputs
```

评估评分模型：

```bash
bash scripts/eval_score_model.sh \
  --dataset_csv data/grasp_success/processed/grasp_success_dataset.csv \
  --checkpoint outputs/checkpoints/score_model_best.pt \
  --output_dir outputs
```

优化可解释 baseline 权重：

```bash
bash scripts/optimize_score_weights.sh \
  --dataset_csv data/grasp_score_quick/processed/grasp_success_dataset.csv \
  --num_trials 5000 \
  --output_dir outputs
```

绘制结果：

```bash
bash scripts/plot_results.sh
bash scripts/plot_score_model_results.sh
```

## 主要命令参数

`scripts/run_demo.sh` 会转发参数给 `vla_world_grasp.pipeline.run_episode`。常用参数如下：

| 参数 | 说明 |
| --- | --- |
| `--instruction` | 中文抓取指令，默认 `抓起红色方块` |
| `--vla_backend` | VLA 后端，常用 `mock` 或 `qwen_vl` |
| `--sim_backend` | 仿真后端，`isaac` 或 `mock` |
| `--headless` / `--no-headless` | 是否无头运行，默认 headless |
| `--save_video` | 保存演示视频 |
| `--video_name` | 输出视频名称 |
| `--num_candidates` | 候选抓取数量，默认 16 |
| `--score_mode` | 评分模式，`baseline`、`model` 或 `hybrid` |
| `--score_model_checkpoint` | 评分模型 checkpoint 路径 |
| `--qwen_model_path` | Qwen2.5-VL 本地模型目录 |
| `--qwen_device` | Qwen 推理设备，默认 `cuda` |
| `--qwen_fallback_to_mock` | Qwen 失败时是否回退到 mock |
| `--debug_attach_on_grasp` | 调试模式下接近抓取后附着目标，便于稳定演示 |

查看完整参数：

```bash
bash scripts/run_demo.sh --sim_backend mock --help
```

## 输出文件说明

默认输出目录为 `outputs/`：

```text
outputs/demo/result.json                 单次运行结果
outputs/logs/*_candidates.json           抓取候选及评分
outputs/logs/*_explanation.txt           中文评分解释
outputs/logs/*_vla_result.json           VLA grounding 结果
outputs/figures/*_candidate_scores.png   候选评分图
outputs/videos/*.mp4                     演示视频
outputs/checkpoints/*.pt                 评分模型 checkpoint
outputs/debug/*                          Qwen 输入图、深度等调试文件
```

数据集默认输出到 `data/`：

```text
data/<dataset_name>/raw/                 每个 episode 的原始 RGB-D、状态和结果
data/<dataset_name>/processed/           训练用 CSV
```

上传 GitHub 时，建议不要提交体积较大的 `outputs/`、`data/`、模型权重和视频帧目录；如果要展示效果，可以只保留少量报告图片、压缩视频或在 Release 中附加大文件。

## GitHub 上传建议

推荐提交：

- `README.md`
- `pyproject.toml`
- `requirements.txt`
- `configs/`
- `scripts/`
- `src/`
- `reports/` 中需要展示的说明文档
- 少量关键 demo 图片或结果摘要

建议忽略：

- `.venv/`
- `__pycache__/`
- `.pytest_cache/`
- `outputs/videos/*_frames/`
- 大体积 `.mp4`、`.npy`、`.pt`、`.tar.gz`
- 本地模型目录，例如 `/home/ubuntu/models/`
- 大规模采集数据 `data/`

可参考 `.gitignore`：

```gitignore
.venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
outputs/videos/*_frames/
*.npy
*.pt
*.tar.gz
data/
models/
```

如果需要保留模型 checkpoint 或实验数据，请优先使用 Git LFS 或 GitHub Release。

## 注意事项

- `mock` 后端适合快速验证流程，但不代表真实视觉 grounding 能力。
- Qwen-VL 只输出目标定位结果，不输出机器人动作，也不直接控制 Franka。
- Isaac 相关脚本默认 headless，适合服务器运行。
- 所有模型后端都应支持失败回退到 `mock`，保证演示流程不中断。
- `--debug_attach_on_grasp` 是演示稳定性选项，不应作为真实物理抓取成功率的唯一依据。
- 当前项目是 MVP/实验性质，重点是打通端到端流程和生成可解释中间结果。
