"""Lightweight visualization helpers for VLA grounding outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def save_grounding_overlay(
    image_path: str | Path,
    output_path: str | Path,
    bbox_2d: Iterable[float] | None = None,
    point_2d: Iterable[float] | None = None,
    target_label_zh: str = "",
    confidence: float | None = None,
    reason_zh: str = "",
    instruction: str = "",
    matched_target_name: str = "",
    matched_object_projected_point: Iterable[float] | None = None,
    qwen_bbox_valid: bool | None = None,
    warning: str = "",
) -> Path | None:
    """Draw Qwen-VL bbox/point grounding on a saved Isaac RGB image."""

    if bbox_2d is None and point_2d is None:
        return None

    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception as exc:
        print(f"[vla_world_grasp] WARNING: grounding overlay skipped because PIL is unavailable: {exc}", flush=True)
        return None

    src = Path(image_path)
    if not src.exists():
        print(f"[vla_world_grasp] WARNING: grounding overlay skipped because image is missing: {src}", flush=True)
        return None

    image = Image.open(src).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    red = (255, 48, 48)
    green = (0, 190, 80)

    if bbox_2d is not None:
        x1, y1, x2, y2 = _clamp_bbox(bbox_2d, width, height)
        _draw_dashed_rectangle(draw, (x1, y1, x2, y2), red, width=3)
        _draw_label(draw, (x1, max(0, y1 - 18)), "Qwen raw bbox", red)
        if point_2d is None:
            point_2d = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    if point_2d is not None:
        u, v = _clamp_point(point_2d, width, height)
        radius = 6
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=(255, 230, 0), outline=(0, 0, 0), width=2)
        draw.line((u - 12, v, u + 12, v), fill=(0, 0, 0), width=1)
        draw.line((u, v - 12, u, v + 12), fill=(0, 0, 0), width=1)

    if matched_object_projected_point is not None:
        u, v = _clamp_point(matched_object_projected_point, width, height)
        radius = 7
        draw.ellipse((u - radius, v - radius, u + radius, v + radius), outline=green, width=4)
        draw.line((u - 14, v, u + 14, v), fill=green, width=3)
        draw.line((u, v - 14, u, v + 14), fill=green, width=3)
        _draw_label(draw, (u + 10, max(0, v - 10)), f"Matched object: {matched_target_name}", green)
    elif bbox_2d is not None or point_2d is not None:
        warning = warning or "Matched object projection unavailable."

    lines = []
    if instruction:
        lines.append(f"中文指令: {instruction}")
    if target_label_zh:
        lines.append(f"Qwen 语义识别: {target_label_zh}")
    if matched_target_name:
        lines.append(f"匹配 Isaac 物体: {matched_target_name}")
    if qwen_bbox_valid is not None:
        lines.append(f"Qwen bbox valid: {str(bool(qwen_bbox_valid)).lower()}")
    if confidence is not None:
        lines.append(f"confidence={float(confidence):.2f}")
    if reason_zh:
        lines.extend(_wrap_text(str(reason_zh), max_chars=28))
    if warning:
        lines.extend(_wrap_text(str(warning), max_chars=28))
    if lines:
        font = _load_font(ImageFont)
        line_height = 14
        box_width = max(180, min(width - 8, max(len(line) * 8 for line in lines) + 12))
        box_height = min(height - 8, line_height * len(lines) + 12)
        draw.rectangle((4, 4, 4 + box_width, 4 + box_height), fill=(255, 255, 255), outline=red)
        y = 10
        for line in lines:
            draw.text((10, y), _safe_text(line, font), fill=(0, 0, 0), font=font)
            y += line_height

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst)
    return dst


def _clamp_bbox(values: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = [float(value) for value in values]
    x1, x2 = sorted((max(0.0, min(float(width - 1), x1)), max(0.0, min(float(width - 1), x2))))
    y1, y2 = sorted((max(0.0, min(float(height - 1), y1)), max(0.0, min(float(height - 1), y2))))
    return x1, y1, x2, y2


def _clamp_point(values: Iterable[float], width: int, height: int) -> tuple[float, float]:
    u, v = [float(value) for value in values]
    return max(0.0, min(float(width - 1), u)), max(0.0, min(float(height - 1), v))


def _wrap_text(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]
    return [text[index : index + max_chars] for index in range(0, len(text), max_chars)]


def _draw_dashed_rectangle(draw, box, color, width: int = 2, dash: int = 10, gap: int = 6) -> None:
    x1, y1, x2, y2 = box
    for offset in range(width):
        _draw_dashed_line(draw, (x1, y1 - offset, x2, y1 - offset), color, dash, gap)
        _draw_dashed_line(draw, (x1, y2 + offset, x2, y2 + offset), color, dash, gap)
        _draw_dashed_line(draw, (x1 - offset, y1, x1 - offset, y2), color, dash, gap)
        _draw_dashed_line(draw, (x2 + offset, y1, x2 + offset, y2), color, dash, gap)


def _draw_dashed_line(draw, line, color, dash: int, gap: int) -> None:
    x1, y1, x2, y2 = [float(value) for value in line]
    length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
    if length <= 0:
        return
    step_x = (x2 - x1) / length
    step_y = (y2 - y1) / length
    progress = 0.0
    while progress < length:
        end = min(progress + dash, length)
        draw.line(
            (
                x1 + step_x * progress,
                y1 + step_y * progress,
                x1 + step_x * end,
                y1 + step_y * end,
            ),
            fill=color,
            width=1,
        )
        progress += dash + gap


def _draw_label(draw, position, text: str, color) -> None:
    x, y = position
    safe = text.encode("unicode_escape").decode("ascii") if any(ord(ch) > 127 for ch in text) else text
    draw.rectangle((x, y, x + max(90, len(safe) * 7), y + 15), fill=(255, 255, 255), outline=color)
    draw.text((x + 3, y + 2), safe, fill=(0, 0, 0))


def _load_font(image_font_module):
    for path in (
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        candidate = Path(path)
        if candidate.exists():
            try:
                return image_font_module.truetype(str(candidate), 14)
            except Exception:
                pass
    return image_font_module.load_default()


def _safe_text(text: str, font) -> str:
    try:
        font.getmask(text)
        return text
    except Exception:
        return text.encode("unicode_escape").decode("ascii")
