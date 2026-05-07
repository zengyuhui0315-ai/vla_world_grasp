#!/usr/bin/env python3
"""Collect defense figures and build 16:9 PPT panels.

This script only copies or renders files under reports/defense_assets/.
It does not modify the project baseline, simulator, renderer, scripts, or
original experiment artifacts.
"""

from __future__ import annotations

import csv
import json
import shutil
import textwrap
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PROJECT = Path(__file__).resolve().parents[3]
ASSETS = PROJECT / "reports" / "defense_assets"
RAW = ASSETS / "raw_selected"
FIGURES = ASSETS / "figures"
PANELS = ASSETS / "panels"
QWEN = ASSETS / "qwen_outputs"
LOGS = ASSETS / "logs"
SCRIPTS = ASSETS / "scripts"
MANIFEST_JSON = ASSETS / "figure_manifest.json"
MANIFEST_CSV = ASSETS / "figure_manifest.csv"
README = ASSETS / "README.md"

PAGE = {
    "env": "03_仿真环境与任务设置",
    "data": "04_数据采集流程",
    "qwen": "05_Qwen视觉语言理解效果",
    "grounding": "06_目标定位与结构化输出",
    "candidate": "07_候选抓取动作生成",
    "success": "08_典型抓取成功样例",
    "failure": "09_失败案例与问题分析",
    "summary": "10_阶段性成果总结",
}

WIDE = (1920, 1080)
RESAMPLE = getattr(getattr(Image, "Resampling", Image), "LANCZOS", Image.LANCZOS)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT))
    except ValueError:
        return str(path)


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc" if bold else "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


FONT_TITLE = font(44, True)
FONT_SUBTITLE = font(30, True)
FONT_BODY = font(26)
FONT_SMALL = font(21)
FONT_MONO = font(22)


def ensure_dirs() -> None:
    for directory in (RAW, FIGURES, PANELS, QWEN, LOGS, SCRIPTS):
        directory.mkdir(parents=True, exist_ok=True)


def copy_file(src: str | Path, dst_dir: Path, dst_name: str | None = None) -> Path:
    src_path = PROJECT / src if not isinstance(src, Path) else src
    dst = dst_dir / (dst_name or src_path.name)
    shutil.copy2(src_path, dst)
    return dst


def load_image(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def fit_image(img: Image.Image, size: tuple[int, int], fill=(248, 249, 251)) -> Image.Image:
    canvas = Image.new("RGB", size, fill)
    fitted = ImageOps.contain(img, size)
    x = (size[0] - fitted.width) // 2
    y = (size[1] - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def cover_image(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    return ImageOps.fit(img, size, method=RESAMPLE, centering=(0.5, 0.5))


def draw_wrapped(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font_obj, fill, width: int, line_gap: int = 8) -> int:
    x, y = xy
    lines: list[str] = []
    for para in text.split("\n"):
        if not para:
            lines.append("")
            continue
        line = ""
        for ch in para:
            test = line + ch
            if draw.textbbox((0, 0), test, font=font_obj)[2] <= width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = ch
        if line:
            lines.append(line)
    for line in lines:
        draw.text((x, y), line, font=font_obj, fill=fill)
        y += font_obj.size + line_gap
    return y


def card(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> None:
    draw.rounded_rectangle(box, radius=8, fill=(255, 255, 255), outline=(218, 224, 232), width=2)
    draw.text((box[0] + 24, box[1] + 18), title, font=FONT_SUBTITLE, fill=(22, 33, 48))


def title_bar(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.rectangle((0, 0, WIDE[0], 92), fill=(25, 34, 47))
    draw.text((48, 22), title, font=FONT_TITLE, fill=(255, 255, 255))
    if subtitle:
        draw.text((1040, 33), subtitle, font=FONT_SMALL, fill=(210, 220, 232))


def make_image_with_text(image_path: Path, out: Path, title: str, text: str) -> Path:
    canvas = Image.new("RGB", WIDE, (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, title, "real experiment image + instruction")
    img = cover_image(load_image(image_path), (1180, 820))
    canvas.paste(img, (48, 170))
    card(draw, (1270, 170, 1872, 990), "中文指令")
    draw_wrapped(draw, (1310, 250), text, FONT_TITLE, (20, 48, 90), 500, 18)
    draw_wrapped(
        draw,
        (1310, 420),
        "输入包括桌面 RGB 图像和自然语言任务。该图仅展示 VLM 输入组织方式，不表示端到端控制已由 Qwen 完成。",
        FONT_BODY,
        (74, 86, 103),
        500,
    )
    canvas.save(out)
    return out


def make_standard_image(image_path: Path, out: Path, title: str, subtitle: str, caption: str) -> Path:
    canvas = Image.new("RGB", WIDE, (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, title, subtitle)
    img = fit_image(load_image(image_path), (1680, 820), fill=(255, 255, 255))
    draw.rounded_rectangle((120, 140, 1800, 960), radius=8, fill=(255, 255, 255), outline=(218, 224, 232), width=2)
    canvas.paste(img, (120, 140))
    draw.rectangle((120, 960, 1800, 1030), fill=(255, 255, 255))
    draw_wrapped(draw, (150, 978), caption, FONT_SMALL, (74, 86, 103), 1600)
    canvas.save(out)
    return out


def render_json(json_path: Path, out: Path, title: str, subtitle: str = "Qwen2.5-VL structured output") -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    fields = {
        "target_object": data.get("target_label_zh") or data.get("target_label_en") or data.get("matched_target_name"),
        "color": data.get("color"),
        "shape": data.get("shape"),
        "task": "grasp",
        "bbox_2d": data.get("bbox_2d"),
        "point_2d": data.get("point_2d"),
        "confidence": data.get("confidence"),
        "grasp_hint": "从上方中心区域接近",
    }
    text = json.dumps(fields, ensure_ascii=False, indent=2)
    canvas = Image.new("RGB", WIDE, (245, 248, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, title, subtitle)
    draw.rounded_rectangle((160, 170, 1760, 960), radius=8, fill=(17, 24, 39))
    y = 220
    for line in text.splitlines():
        draw.text((220, y), line, font=FONT_MONO, fill=(226, 232, 240))
        y += 34
    draw.text((220, 890), "Rendered from real Qwen result JSON; not a hand-written experiment result.", font=FONT_SMALL, fill=(148, 163, 184))
    canvas.save(out)
    return out


def render_failure_json(json_path: Path, out: Path) -> Path:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    execution = data.get("execution") or {}
    fields = {
        "episode_id": data.get("episode_id"),
        "instruction": data.get("instruction_zh") or data.get("instruction"),
        "target_name": data.get("target_name"),
        "success": data.get("success"),
        "attached": data.get("attached"),
        "target_initial_z": data.get("target_initial_z"),
        "target_final_z": data.get("target_final_z"),
        "success_threshold_z": execution.get("success_threshold_z"),
        "failure_note": "attached=false; target_final_z did not pass success threshold",
    }
    text = json.dumps(fields, ensure_ascii=False, indent=2)
    canvas = Image.new("RGB", WIDE, (245, 248, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, "失败样例 result.json", "real failed episode metadata")
    draw.rounded_rectangle((160, 170, 1760, 960), radius=8, fill=(17, 24, 39))
    y = 220
    for line in text.splitlines():
        color = (248, 113, 113) if '"success": false' in line or '"attached": false' in line else (226, 232, 240)
        draw.text((220, y), line, font=FONT_MONO, fill=color)
        y += 34
    draw.text((220, 890), "Rendered from real failed episode result.json; not a fabricated result.", font=FONT_SMALL, fill=(148, 163, 184))
    canvas.save(out)
    return out


def make_grid(images: list[tuple[Path, str]], out: Path, title: str, cols: int = 3) -> Path:
    rows = (len(images) + cols - 1) // cols
    canvas = Image.new("RGB", WIDE, (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, title, "real collected RGB samples")
    margin, gap = 48, 24
    top = 140
    cell_w = (WIDE[0] - 2 * margin - (cols - 1) * gap) // cols
    cell_h = (WIDE[1] - top - 56 - (rows - 1) * gap) // rows
    for idx, (path, label) in enumerate(images):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + gap)
        y = top + row * (cell_h + gap)
        tile = cover_image(load_image(path), (cell_w, cell_h - 52))
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8, fill=(255, 255, 255), outline=(218, 224, 232))
        canvas.paste(tile, (x, y))
        draw.rectangle((x, y + cell_h - 52, x + cell_w, y + cell_h), fill=(255, 255, 255))
        draw.text((x + 16, y + cell_h - 40), label, font=FONT_SMALL, fill=(44, 56, 74))
    canvas.save(out)
    return out


def make_four_panel(items: list[dict], out: Path, title: str, subtitle: str) -> Path:
    canvas = Image.new("RGB", WIDE, (247, 249, 252))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, title, subtitle)
    margin, gap = 48, 24
    cell_w = (WIDE[0] - 2 * margin - gap) // 2
    cell_h = (WIDE[1] - 140 - 48 - gap) // 2
    for idx, item in enumerate(items[:4]):
        row, col = divmod(idx, 2)
        x = margin + col * (cell_w + gap)
        y = 140 + row * (cell_h + gap)
        draw.rounded_rectangle((x, y, x + cell_w, y + cell_h), radius=8, fill=(255, 255, 255), outline=(218, 224, 232), width=2)
        draw.text((x + 22, y + 18), item["title"], font=FONT_SUBTITLE, fill=(22, 33, 48))
        if item.get("path"):
            img = fit_image(load_image(item["path"]), (cell_w - 44, cell_h - 96), fill=(255, 255, 255))
            canvas.paste(img, (x + 22, y + 72))
        else:
            draw_wrapped(draw, (x + 36, y + 110), item.get("text", ""), FONT_BODY, (60, 72, 88), cell_w - 72, 10)
        if item.get("caption"):
            draw.rectangle((x + 22, y + cell_h - 54, x + cell_w - 22, y + cell_h - 18), fill=(246, 248, 251))
            draw.text((x + 34, y + cell_h - 48), item["caption"], font=FONT_SMALL, fill=(74, 86, 103))
    canvas.save(out)
    return out


def draw_pipeline(out: Path) -> Path:
    canvas = Image.new("RGB", WIDE, (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, "VLA-style 抓取原型流程示意图", "schematic, not an experimental result")
    nodes = [
        ("中文指令", "如：抓起蓝色圆柱"),
        ("图像输入", "桌面 RGB / Depth"),
        ("Qwen2.5-VL", "目标物体与属性理解"),
        ("目标定位", "bbox / center point"),
        ("候选动作", "Top-K grasp candidates"),
        ("评分排序", "几何与学习评分"),
        ("执行记录", "抓取结果与日志"),
    ]
    x0, y0, w, h, gap = 70, 360, 230, 170, 28
    for i, (name, desc) in enumerate(nodes):
        x = x0 + i * (w + gap)
        draw.rounded_rectangle((x, y0, x + w, y0 + h), radius=8, fill=(255, 255, 255), outline=(90, 119, 150), width=3)
        draw.text((x + 24, y0 + 34), name, font=FONT_SUBTITLE, fill=(23, 37, 54))
        draw_wrapped(draw, (x + 24, y0 + 88), desc, FONT_SMALL, (82, 96, 112), w - 48)
        if i < len(nodes) - 1:
            ax = x + w + 6
            ay = y0 + h // 2
            draw.line((ax, ay, ax + gap - 12, ay), fill=(58, 84, 112), width=4)
            draw.polygon([(ax + gap - 12, ay - 10), (ax + gap + 4, ay), (ax + gap - 12, ay + 10)], fill=(58, 84, 112))
    draw_wrapped(
        draw,
        (120, 690),
        "展示口径：当前系统展示的是 VLA-style 抓取原型闭环，包括语言指令、视觉理解、目标定位、候选抓取动作生成和执行结果记录。",
        FONT_BODY,
        (54, 67, 83),
        1680,
    )
    draw.text((120, 850), "SCHEMATIC / 示意图：该图不代表新的实验结果。", font=FONT_BODY, fill=(150, 55, 45))
    canvas.save(out)
    return out


def draw_stage_summary(out: Path) -> Path:
    canvas = Image.new("RGB", WIDE, (246, 248, 251))
    draw = ImageDraw.Draw(canvas)
    title_bar(draw, "阶段性成果总结示意图", "schematic, not an experimental result")
    left = [
        "仿真桌面抓取任务与多物体场景",
        "中文指令解析与目标物体选择",
        "Qwen2.5-VL grounding 结果记录",
        "候选抓取动作生成与评分排序",
        "成功 / 失败 episode 与日志留存",
    ]
    right = [
        "端到端机器人控制仍未声称完成",
        "失败过程视频样例仍需补采",
        "更大规模真实 VLM 评估待扩展",
        "评分模型与执行策略仍需联合优化",
    ]
    card(draw, (140, 190, 900, 910), "已完成模块")
    card(draw, (1020, 190, 1780, 910), "后续工作")
    y = 285
    for item in left:
        draw.text((190, y), "✓", font=FONT_SUBTITLE, fill=(30, 125, 75))
        draw_wrapped(draw, (245, y), item, FONT_BODY, (38, 52, 69), 560)
        y += 95
    y = 285
    for item in right:
        draw.text((1070, y), "•", font=FONT_SUBTITLE, fill=(169, 98, 38))
        draw_wrapped(draw, (1125, y), item, FONT_BODY, (38, 52, 69), 560)
        y += 105
    draw.text((140, 980), "SCHEMATIC / 示意图：用于答辩总结，不是实验截图。", font=FONT_SMALL, fill=(150, 55, 45))
    canvas.save(out)
    return out


def write_placeholder(out: Path, title: str, message: str) -> Path:
    canvas = Image.new("RGB", WIDE, (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, 0, WIDE[0], WIDE[1]), fill=(250, 250, 250), outline=(204, 210, 218), width=8)
    draw.text((110, 110), title, font=FONT_TITLE, fill=(80, 88, 100))
    draw_wrapped(draw, (110, 230), message, FONT_BODY, (90, 99, 112), 1550, 12)
    draw.text((110, 930), "PLACEHOLDER ONLY / 非实验结果", font=FONT_SUBTITLE, fill=(160, 60, 48))
    canvas.save(out)
    return out


def add_record(records: list[dict], figure_id: str, original: Path | str | None, copied: Path, category: str, page: str, title: str, caption: str, real: bool, schematic: bool, notes: str) -> None:
    records.append(
        {
            "figure_id": figure_id,
            "original_path": rel(original) if isinstance(original, Path) else (original or ""),
            "copied_path": rel(copied),
            "category": category,
            "suggested_ppt_page": page,
            "title_zh": title,
            "caption_zh": caption,
            "is_real_experiment": real,
            "is_schematic": schematic,
            "notes": notes,
        }
    )


def collect_assets() -> list[dict]:
    ensure_dirs()
    log_lines = [f"Defense asset collection started at {datetime.now().isoformat(timespec='seconds')}"]
    records: list[dict] = []

    selected_sources = {
        "environment": PROJECT / "outputs" / "recordings" / "2_blue_cylinder" / "frames" / "frame_0000.png",
        "qwen_input": PROJECT / "outputs" / "debug" / "qwen_vl_real_blue_cylinder_demo_qwen_input.png",
        "grounding": PROJECT / "outputs" / "figures" / "qwen_overlay_integration_smoke_grounding_overlay.png",
        "qwen_scores": PROJECT / "outputs" / "figures" / "qwen_vl_real_blue_cylinder_demo_candidate_scores.png",
        "success_pre": PROJECT / "outputs" / "videos" / "qwen_vl_real_blue_cylinder_demo_frames" / "frame_000000.png",
        "success_mid": PROJECT / "outputs" / "videos" / "qwen_vl_real_blue_cylinder_demo_frames" / "frame_000084.png",
        "success_end": PROJECT / "outputs" / "videos" / "qwen_vl_real_blue_cylinder_demo_frames" / "frame_000168.png",
        "failure_rgb": PROJECT / "data" / "grasp_success" / "raw" / "episode_000073" / "rgb.png",
        "failure_result": PROJECT / "data" / "grasp_success" / "raw" / "episode_000073" / "result.json",
        "baseline_distribution": PROJECT / "outputs" / "figures" / "baseline_score_success_failure_distribution.png",
    }
    data_samples = [
        PROJECT / "data" / "grasp_success" / "raw" / f"episode_{i:06d}" / "rgb.png"
        for i in (1, 2, 3, 4, 5, 73)
    ]

    copied = {}
    for key, src in selected_sources.items():
        if src.exists() and src.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            copied[key] = copy_file(src, RAW, f"raw_{key}{src.suffix.lower()}")
            log_lines.append(f"copied {rel(src)} -> {rel(copied[key])}")
    sample_copies = []
    for idx, src in enumerate(data_samples, start=1):
        if src.exists():
            dst = copy_file(src, RAW, f"raw_data_sample_{idx:02d}.png")
            sample_copies.append(dst)
            log_lines.append(f"copied {rel(src)} -> {rel(dst)}")

    for src in sorted((PROJECT / "outputs" / "logs").glob("qwen_*")):
        dst = copy_file(src, QWEN)
        log_lines.append(f"copied {rel(src)} -> {rel(dst)}")

    qwen_json = QWEN / "qwen_vl_real_blue_cylinder_demo_vla_result.json"
    figure_01 = make_standard_image(copied["environment"], FIGURES / "figure_01_environment_overview.png", "Isaac 桌面多物体抓取场景总览", "real recording frame", "真实录制帧，展示桌面、机械臂和多物体抓取任务环境。")
    add_record(records, "figure_01_environment_overview", selected_sources["environment"], figure_01, "environment_setup", PAGE["env"], "Isaac 桌面多物体抓取场景总览", "真实录制帧，展示桌面、机械臂和多物体抓取任务环境。", True, False, "copied from existing recording frame")

    figure_02 = make_grid([(p, f"episode sample {i}") for i, p in enumerate(sample_copies, 1)], FIGURES / "figure_02_data_collection_samples.png", "数据采集 episode 样例", cols=3)
    add_record(records, "figure_02_data_collection_samples", "multiple copied data rgb.png files", figure_02, "data_collection", PAGE["data"], "数据采集样例拼图", "真实数据集 RGB 样例拼图，展示不同 episode 的桌面物体组合。", True, False, "composed from copied real dataset images")

    figure_03 = make_image_with_text(copied["qwen_input"], FIGURES / "figure_03_qwen_input_instruction.png", "Qwen 输入：图像 + 中文指令", "抓起蓝色圆柱")
    add_record(records, "figure_03_qwen_input_instruction", selected_sources["qwen_input"], figure_03, "qwen_vl_grounding", PAGE["qwen"], "Qwen2.5-VL 输入示例", "真实 Qwen 输入图与中文指令排版，用于展示视觉语言理解输入。", True, False, "layout generated from real Qwen input image")

    figure_04 = make_standard_image(copied["grounding"], FIGURES / "figure_04_qwen_grounding_result.png", "Qwen2.5-VL 目标定位结果", "real grounding overlay", "真实 grounding 可视化，展示目标物体 bbox / center point。")
    add_record(records, "figure_04_qwen_grounding_result", selected_sources["grounding"], figure_04, "qwen_vl_grounding", PAGE["grounding"], "Qwen2.5-VL 目标定位结果", "真实 grounding 可视化，展示目标物体 bbox / center point。", True, False, "copied existing grounding overlay")

    figure_05 = render_json(qwen_json, FIGURES / "figure_05_qwen_json_output.png", "Qwen 输出 JSON 渲染图")
    add_record(records, "figure_05_qwen_json_output", qwen_json, figure_05, "qwen_vl_json_output", PAGE["grounding"], "Qwen2.5-VL 结构化输出", "由真实 Qwen 结果 JSON 渲染，展示目标、属性、位置和抓取提示字段。", True, False, "rendered from copied real JSON output")

    figure_06 = make_four_panel(
        [
            {"title": "输入图像", "path": copied["qwen_input"], "caption": "real Qwen input"},
            {"title": "候选评分", "path": copied["qwen_scores"], "caption": "candidate score chart"},
            {"title": "目标定位", "path": copied["grounding"], "caption": "grounding overlay"},
            {"title": "候选动作说明", "text": "Top-K grasp candidates are generated around the localized target and ranked by score terms. This tile summarizes existing outputs, not a new execution result."},
        ],
        FIGURES / "figure_06_grasp_candidates.png",
        "候选抓取动作可视化",
        "real figures + explanatory tile",
    )
    add_record(records, "figure_06_grasp_candidates", "qwen input, grounding overlay, candidate score chart", figure_06, "grasp_candidate", PAGE["candidate"], "候选抓取动作生成", "基于真实输入、grounding 和候选评分图组成，说明从视觉理解到候选动作的连接。", True, False, "one explanatory tile included; no fabricated experimental result")

    figure_07 = make_four_panel(
        [
            {"title": "抓取前", "path": copied["success_pre"], "caption": "frame_000000"},
            {"title": "目标识别", "path": copied["grounding"], "caption": "Qwen grounding"},
            {"title": "执行中", "path": copied["success_mid"], "caption": "frame_000084"},
            {"title": "抓取成功", "path": copied["success_end"], "caption": "frame_000168"},
        ],
        FIGURES / "figure_07_success_case_panel.png",
        "典型抓取成功样例",
        "real Qwen-guided recording frames",
    )
    add_record(records, "figure_07_success_case_panel", "qwen real blue cylinder frames + grounding overlay", figure_07, "qwen_guided_grasp", PAGE["success"], "抓取成功样例面板", "真实录制帧展示抓取前、目标定位、执行中和成功结果。", True, False, "composed from existing real frames")

    failure_placeholder = write_placeholder(
        FIGURES / "figure_08_failure_process_placeholder.png",
        "失败过程帧缺失",
        "当前项目存在 success=false 的真实失败 episode 和 result.json，但该 episode 没有保存执行过程视频帧。本占位图仅用于提醒补采，不能作为实验截图。",
    )
    add_record(records, "figure_08_failure_process_placeholder", None, failure_placeholder, "baseline_failure", PAGE["failure"], "失败过程帧占位图", "失败过程视频帧当前缺失。该图是 placeholder，不是实验结果。", False, False, "placeholder only, not an experimental result")

    failure_text = render_failure_json(selected_sources["failure_result"], FIGURES / "figure_08_failure_result_json.png")
    add_record(records, "figure_08_failure_result_json", selected_sources["failure_result"], failure_text, "baseline_failure", PAGE["failure"], "失败样例结构化记录", "由真实 success=false episode 的 result.json 渲染。", True, False, "rendered from real failed episode result")

    figure_08 = make_four_panel(
        [
            {"title": "失败前输入", "path": copied["failure_rgb"], "caption": "episode_000073 rgb"},
            {"title": "失败过程", "path": failure_placeholder, "caption": "placeholder only"},
            {"title": "失败结果记录", "path": failure_text, "caption": "success=false JSON"},
            {"title": "原因分析辅助", "path": copied["baseline_distribution"], "caption": "success/failure distribution"},
        ],
        FIGURES / "figure_08_failure_case_panel.png",
        "失败案例与问题分析",
        "mixed: real failure artifacts + labeled placeholder",
    )
    add_record(records, "figure_08_failure_case_panel", "real failed episode + placeholder + real distribution figure", figure_08, "problem_analysis", PAGE["failure"], "失败案例分析面板", "包含真实失败 episode 输入与记录；失败过程帧缺失位置已明确标为 placeholder。", False, False, "mixed panel; contains placeholder, not a pure experimental result")

    figure_09 = draw_pipeline(FIGURES / "figure_09_pipeline_overview_schematic.png")
    add_record(records, "figure_09_pipeline_overview_schematic", None, figure_09, "pipeline_overview", PAGE["summary"], "系统总流程示意图", "中文指令、图像输入、Qwen2.5-VL、目标定位、候选动作、评分和执行记录流程。", False, True, "schematic, not an experimental result")

    figure_10 = draw_stage_summary(FIGURES / "figure_10_stage_summary_schematic.png")
    add_record(records, "figure_10_stage_summary_schematic", None, figure_10, "scoring_model_pending", PAGE["summary"], "阶段性成果总结示意图", "总结已完成模块与后续工作。", False, True, "schematic, not an experimental result")

    panels = [
        (
            PANELS / "panel_qwen_effect.png",
            [
                {"title": "输入图像", "path": copied["qwen_input"], "caption": "real input"},
                {"title": "中文指令", "text": "抓起蓝色圆柱\n\nQwen2.5-VL-3B-Instruct 作为视觉语言理解模块，根据中文指令和桌面图像输出目标物体、属性、位置或抓取提示。"},
                {"title": "Qwen JSON", "path": figure_05, "caption": "rendered from real JSON"},
                {"title": "目标框可视化", "path": copied["grounding"], "caption": "real overlay"},
            ],
            "Qwen2.5-VL 视觉语言理解效果",
            "real outputs",
        ),
        (
            PANELS / "panel_grasp_success.png",
            [
                {"title": "抓取前", "path": copied["success_pre"], "caption": "frame_000000"},
                {"title": "目标识别", "path": copied["grounding"], "caption": "grounding overlay"},
                {"title": "执行中", "path": copied["success_mid"], "caption": "frame_000084"},
                {"title": "抓取成功", "path": copied["success_end"], "caption": "frame_000168"},
            ],
            "抓取成功闭环样例",
            "real recording frames",
        ),
        (
            PANELS / "panel_failure_analysis.png",
            [
                {"title": "失败前", "path": copied["failure_rgb"], "caption": "real failed episode"},
                {"title": "失败过程", "path": failure_placeholder, "caption": "placeholder only"},
                {"title": "失败结果", "path": failure_text, "caption": "success=false"},
                {"title": "原因说明", "text": "该失败样例 result.json 中 attached=false，target_final_z 未超过成功阈值。当前没有对应执行过程帧，因此过程位置使用明确标注的 placeholder。"},
            ],
            "失败案例与问题分析",
            "real failure record + labeled placeholder",
        ),
    ]
    for out, items, title, subtitle in panels:
        make_four_panel(items, out, title, subtitle)

    make_grid([(p, f"episode sample {i}") for i, p in enumerate(sample_copies, 1)], PANELS / "panel_data_collection.png", "数据采集样例", cols=3)
    draw_pipeline(PANELS / "panel_system_pipeline.png")

    for panel_name, page, title, real, schematic, notes in [
        ("panel_qwen_effect.png", PAGE["qwen"], "Qwen 效果四宫格", True, False, "panel generated from real Qwen artifacts"),
        ("panel_grasp_success.png", PAGE["success"], "成功样例四宫格", True, False, "panel generated from real frames"),
        ("panel_failure_analysis.png", PAGE["failure"], "失败分析四宫格", False, False, "mixed panel includes explicit placeholder"),
        ("panel_data_collection.png", PAGE["data"], "数据采集多图网格", True, False, "panel generated from real dataset images"),
        ("panel_system_pipeline.png", PAGE["summary"], "系统流程图", False, True, "schematic, not an experimental result"),
    ]:
        add_record(records, panel_name.removesuffix(".png"), None, PANELS / panel_name, "pipeline_overview" if schematic else "problem_analysis" if "failure" in panel_name else "qwen_vl_grounding", page, title, title, real, schematic, notes)

    all_images = sorted(
        p for p in PROJECT.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        and "external" not in p.parts
        and "reports/defense_assets" not in str(p)
    )
    log_lines.append(f"scanned image files: {len(all_images)}")
    log_lines.append(f"manifest records: {len(records)}")
    (LOGS / "asset_collection.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return records


def write_manifest(records: list[dict]) -> None:
    MANIFEST_JSON.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    fieldnames = [
        "figure_id",
        "original_path",
        "copied_path",
        "category",
        "suggested_ppt_page",
        "title_zh",
        "caption_zh",
        "is_real_experiment",
        "is_schematic",
        "notes",
    ]
    with MANIFEST_CSV.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def write_readme(records: list[dict]) -> None:
    real = [r for r in records if r["is_real_experiment"]]
    schematics = [r for r in records if r["is_schematic"]]
    placeholders = [r for r in records if "placeholder" in r["copied_path"] or "placeholder" in r["notes"]]
    lines = [
        "# 结题答辩实验图包",
        "",
        "本图包用于整理 `vla_world_model_grasp / vla_world_grasp` 项目中已经产生的实验截图、Qwen2.5-VL 推理结果、抓取执行截图、成功/失败样例、数据采集样例和系统流程图，方便制作结题答辩 PPT。",
        "",
        "整理原则：所有真实实验素材均来自项目已有文件；本目录内的整理操作只复制或渲染副本，不移动、不删除原始实验文件，不修改 baseline 抓取逻辑、执行脚本、Isaac scene、robot、renderer 或 external 目录。",
        "",
        "Qwen2.5-VL 当前展示口径：",
        "",
        "> Qwen2.5-VL-3B-Instruct 作为视觉语言理解模块，用于根据中文指令和桌面图像输出目标物体、属性、位置或抓取提示。当前阶段不声称其已经完成端到端机器人控制。",
        "",
        "当前系统展示口径：",
        "",
        "> 当前系统展示的是 VLA-style 抓取原型闭环，包括语言指令、视觉理解、目标定位、候选抓取动作生成和执行结果记录。",
        "",
        "## 目录说明",
        "",
        "- `raw_selected/`：从项目中筛选出的原始截图副本。",
        "- `figures/`：单张标准化实验图、示意图和占位图。",
        "- `panels/`：适合 PPT 的 16:9 拼图。",
        "- `qwen_outputs/`：Qwen2.5-VL 结构化输出 JSON 或文本副本。",
        "- `logs/`：整理过程日志。",
        "- `scripts/`：本次新增的整理和可视化脚本。",
        "",
        "## 推荐 PPT 页面",
        "",
    ]
    for r in records:
        if r["figure_id"].startswith("figure_") or r["figure_id"].startswith("panel_"):
            flag = "真实实验图" if r["is_real_experiment"] else "示意图" if r["is_schematic"] else "占位/混合说明图"
            lines.append(f"- `{r['copied_path']}`：{r['suggested_ppt_page']}，{r['title_zh']}，{flag}。")
    lines += [
        "",
        "## 真实实验截图",
        "",
    ]
    lines += [f"- `{r['copied_path']}`：{r['title_zh']}" for r in real]
    lines += [
        "",
        "## 示意图",
        "",
    ]
    lines += [f"- `{r['copied_path']}`：{r['title_zh']}。文件名或说明中已标注 schematic / 示意图。" for r in schematics]
    lines += [
        "",
        "## 占位图",
        "",
    ]
    if placeholders:
        lines += [f"- `{r['copied_path']}`：{r['notes']}" for r in placeholders]
    else:
        lines.append("- 无。")
    lines += [
        "",
        "详细字段见 `figure_manifest.json` 和 `figure_manifest.csv`。",
    ]
    README.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    records = collect_assets()
    write_manifest(records)
    write_readme(records)
    print(json.dumps(
        {
            "total": len(records),
            "real_experiment": sum(bool(r["is_real_experiment"]) for r in records),
            "schematic": sum(bool(r["is_schematic"]) for r in records),
            "placeholder": sum("placeholder" in r["copied_path"] or "placeholder" in r["notes"] for r in records),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
