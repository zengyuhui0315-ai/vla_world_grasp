"""Video helpers.

The default demo path records Isaac camera frames. The 2D renderer in this file
is kept only for explicit ``--sim_backend mock`` debug runs.
"""

from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
import zlib
from pathlib import Path
from typing import Any


Color = tuple[int, int, int]


WHITE: Color = (250, 250, 247)
INK: Color = (28, 32, 38)
TABLE: Color = (210, 205, 194)
ROBOT: Color = (48, 52, 60)
SUCCESS: Color = (42, 143, 72)
MUTED: Color = (130, 136, 145)


def create_debug_mock_video(
    output_dir: str | Path,
    result: dict[str, Any],
    fps: int = 8,
    size: tuple[int, int] = (640, 480),
) -> dict[str, Any]:
    """Mock video generation is disabled for final demos."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "mock_renderer_disabled",
        "frames_dir": None,
        "frame_count": 0,
        "fps": fps,
        "mp4_path": None,
        "message": "2D mock renderer video output is disabled. Use --sim_backend isaac for final demos.",
    }
    manifest_path = output_path / "video_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def render_episode_frames(
    result: dict[str, Any],
    frames_dir: str | Path,
    size: tuple[int, int] = (640, 480),
) -> list[Path]:
    """Render a simple top-down scene animation from an episode result."""

    frames_path = Path(frames_dir)
    frames_path.mkdir(parents=True, exist_ok=True)

    objects = result.get("object_states", [])
    selected = result.get("selected_candidate", {})
    execution = result.get("execution", {})
    target_name = execution.get("target_object") or selected.get("target_object")
    trajectory = execution.get("trajectory", [])
    success = bool(result.get("success"))

    timeline = _expanded_timeline(trajectory)
    if not timeline:
        timeline = [{"name": "scene", "position": selected.get("position", (0.0, 0.0, 0.15))}]

    frame_paths = []
    lifted_z = float(execution.get("final_z", 0.2))
    for idx, command in enumerate(timeline):
        progress = idx / max(1, len(timeline) - 1)
        canvas = new_canvas(size[0], size[1], WHITE)
        _draw_scene(
            canvas,
            objects=objects,
            target_name=target_name,
            command=command,
            selected=selected,
            progress=progress,
            lifted_z=lifted_z,
            success=success,
            result=result,
        )
        frame_path = frames_path / f"frame_{idx:04d}.png"
        write_png_rgb(frame_path, canvas)
        frame_paths.append(frame_path)
    return frame_paths


def frames_to_video(
    frames_dir: str | Path,
    output_path: str | Path,
    fps: int = 8,
    timeout_seconds: int = 120,
) -> dict[str, Any]:
    """Encode PNG frames to mp4 with ffmpeg and a hard timeout."""

    frames_path = Path(frames_dir)
    output = Path(output_path)
    ffmpeg = _find_ffmpeg()
    if ffmpeg is None:
        print("[vla_world_grasp] ffmpeg not found.", flush=True)
        print("Install with:", flush=True)
        print("  sudo apt install -y ffmpeg", flush=True)
        print("or:", flush=True)
        print("  conda install -c conda-forge ffmpeg -y", flush=True)
        print(f"Frames kept in: {frames_path}", flush=True)
        return {
            "available": False,
            "encoded": False,
            "message": "ffmpeg not found; frames were saved instead",
            "frames_dir": str(frames_path),
        }

    primary_pattern = frames_path / "frame_%06d.png"
    primary_first_frame = frames_path / "frame_000000.png"
    glob_pattern = frames_path / "*.png"
    fallback_frames = sorted(frames_path.glob("*.png"))

    if primary_first_frame.exists():
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(primary_pattern),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output),
        ]
        input_mode = "sequence"
    elif fallback_frames:
        cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            str(glob_pattern),
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
            str(output),
        ]
        input_mode = "glob"
    else:
        message = f"no frames found in {frames_path}"
        print(f"[vla_world_grasp] {message}", flush=True)
        return {
            "available": True,
            "encoded": False,
            "message": message,
            "frames_dir": str(frames_path),
        }

    print(f"[vla_world_grasp] ffmpeg command: {' '.join(cmd)}", flush=True)
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        stderr = exc.stderr or ""
        print(f"[vla_world_grasp] ffmpeg timed out after {timeout_seconds}s", flush=True)
        if stderr:
            print(f"[vla_world_grasp] ffmpeg stderr:\n{stderr}", flush=True)
        return {
            "available": True,
            "encoded": False,
            "timed_out": True,
            "timeout_seconds": timeout_seconds,
            "stderr": stderr,
            "frames_dir": str(frames_path),
            "command": cmd,
            "input_mode": input_mode,
        }

    if completed.returncode != 0:
        print(f"[vla_world_grasp] ffmpeg failed with code {completed.returncode}", flush=True)
        if completed.stdout:
            print(f"[vla_world_grasp] ffmpeg stdout:\n{completed.stdout}", flush=True)
        if completed.stderr:
            print(f"[vla_world_grasp] ffmpeg stderr:\n{completed.stderr}", flush=True)
        return {
            "available": True,
            "encoded": False,
            "returncode": completed.returncode,
            "stderr": completed.stderr,
            "stdout": completed.stdout,
            "frames_dir": str(frames_path),
            "command": cmd,
            "input_mode": input_mode,
        }

    print(f"[vla_world_grasp] Saved video: {output}", flush=True)
    return {
        "available": True,
        "encoded": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "frames_dir": str(frames_path),
        "mp4_path": str(output),
        "command": cmd,
        "input_mode": input_mode,
    }


def encode_frames_to_mp4(frames_dir: str | Path, output_path: str | Path, fps: int = 8) -> dict[str, Any]:
    """Backward-compatible wrapper for frame encoding."""

    timeout_seconds = int(os.environ.get("VLA_WORLD_GRASP_FFMPEG_TIMEOUT", "120"))
    return frames_to_video(frames_dir, output_path, fps=fps, timeout_seconds=timeout_seconds)


def _find_ffmpeg() -> str | None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        return ffmpeg
    python_bin_dir = Path(sys.executable).resolve().parent
    candidate = python_bin_dir / "ffmpeg"
    if candidate.exists():
        return str(candidate)
    return None


def write_png_rgb(path: str | Path, pixels: list[list[Color]]) -> Path:
    """Write an RGB PNG with no third-party dependencies."""

    output_path = Path(path)
    height = len(pixels)
    width = len(pixels[0]) if height else 0
    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for red, green, blue in row:
            raw.extend((red, green, blue))

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), level=6))
    png += chunk(b"IEND", b"")
    output_path.write_bytes(png)
    return output_path


def write_png_array_rgb(path: str | Path, rgb: Any) -> Path:
    """Write an HxWx3 uint8-like array as PNG without converting to Python pixels."""

    if hasattr(rgb, "detach"):
        rgb = rgb.detach().cpu().numpy()
    if rgb.shape[-1] > 3:
        rgb = rgb[:, :, :3]
    if str(rgb.dtype) != "uint8":
        rgb = (rgb.clip(0.0, 1.0) * 255).astype("uint8") if float(rgb.max()) <= 1.0 else rgb.astype("uint8")

    output_path = Path(path)
    height, width = int(rgb.shape[0]), int(rgb.shape[1])
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw.extend(rgb[y].tobytes())

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(bytes(raw), level=3))
    png += chunk(b"IEND", b"")
    output_path.write_bytes(png)
    return output_path


def new_canvas(width: int, height: int, color: Color = WHITE) -> list[list[Color]]:
    return [[color for _ in range(width)] for _ in range(height)]


def draw_rect(canvas: list[list[Color]], x0: int, y0: int, x1: int, y1: int, color: Color) -> None:
    width, height = len(canvas[0]), len(canvas)
    for y in range(max(0, y0), min(height, y1)):
        row = canvas[y]
        for x in range(max(0, x0), min(width, x1)):
            row[x] = color


def draw_circle(canvas: list[list[Color]], cx: int, cy: int, radius: int, color: Color) -> None:
    width, height = len(canvas[0]), len(canvas)
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                canvas[y][x] = color


def draw_line(canvas: list[list[Color]], start: tuple[int, int], end: tuple[int, int], color: Color) -> None:
    x0, y0 = start
    x1, y1 = end
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    for idx in range(steps + 1):
        t = idx / steps
        x = round(x0 + (x1 - x0) * t)
        y = round(y0 + (y1 - y0) * t)
        draw_circle(canvas, x, y, 2, color)


def draw_text(canvas: list[list[Color]], x: int, y: int, text: str, color: Color = INK, scale: int = 2) -> None:
    cursor = x
    for char in text:
        glyph = FONT.get(char.upper(), FONT.get("?", ()))
        for row_idx, row in enumerate(glyph):
            for col_idx, bit in enumerate(row):
                if bit == "1":
                    draw_rect(
                        canvas,
                        cursor + col_idx * scale,
                        y + row_idx * scale,
                        cursor + (col_idx + 1) * scale,
                        y + (row_idx + 1) * scale,
                        color,
                    )
        cursor += 6 * scale


def _draw_scene(
    canvas: list[list[Color]],
    objects: list[dict[str, Any]],
    target_name: str | None,
    command: dict[str, Any],
    selected: dict[str, Any],
    progress: float,
    lifted_z: float,
    success: bool,
    result: dict[str, Any],
) -> None:
    width, height = len(canvas[0]), len(canvas)
    table = (90, 95, width - 90, height - 85)
    draw_rect(canvas, *table, TABLE)
    draw_rect(canvas, table[0], table[1], table[2], table[1] + 4, MUTED)
    draw_rect(canvas, table[0], table[3] - 4, table[2], table[3], MUTED)

    command_position = command.get("position") or selected.get("position") or (0.0, 0.0, 0.2)
    hand_xy = _world_to_pixel(command_position, table)
    draw_line(canvas, (width // 2, 72), hand_xy, ROBOT)
    draw_circle(canvas, width // 2, 72, 18, ROBOT)

    for obj in objects:
        position = tuple(obj.get("position", (0.0, 0.0, 0.0)))
        if obj.get("name") == target_name and progress > 0.72:
            position = (position[0], position[1], lifted_z)
        px, py = _world_to_pixel(position, table)
        color = _rgb255(obj.get("rgb"), obj.get("color"))
        radius = max(10, int(max(obj.get("size", (0.05, 0.05, 0.05))) * 360))
        if obj.get("name") == target_name:
            draw_circle(canvas, px, py, radius + 6, INK if not success else SUCCESS)
        if obj.get("type") == "cube":
            draw_rect(canvas, px - radius, py - radius, px + radius, py + radius, color)
        elif obj.get("type") == "cylinder":
            draw_circle(canvas, px, py, radius, color)
            draw_line(canvas, (px - radius, py), (px + radius, py), WHITE)
        else:
            draw_circle(canvas, px, py, radius, color)

    selected_xy = _world_to_pixel(selected.get("position", (0.0, 0.0, 0.0)), table)
    draw_circle(canvas, selected_xy[0], selected_xy[1], 5, INK)
    draw_circle(canvas, hand_xy[0], hand_xy[1], 10, ROBOT)

    draw_text(canvas, 24, 22, "VLA WORLD GRASP", INK, scale=2)
    draw_text(canvas, 24, 52, f"TARGET {target_name or 'UNKNOWN'}", INK, scale=2)
    draw_text(canvas, 24, height - 42, f"STEP {command.get('name', 'scene')}", MUTED, scale=2)
    status = "SUCCESS" if success else "FAILED"
    draw_text(canvas, width - 170, height - 42, status, SUCCESS if success else (170, 55, 55), scale=2)
    draw_text(canvas, width - 170, 26, str(result.get("vla_backend", "mock")), MUTED, scale=2)


def _expanded_timeline(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frames = []
    previous = None
    for command in trajectory:
        position = command.get("position")
        if position is None:
            repeat = 4
            frames.extend([command] * repeat)
            continue
        if previous is None:
            frames.extend([command] * 4)
        else:
            for idx in range(1, 7):
                t = idx / 6.0
                interp = tuple(previous[i] + (position[i] - previous[i]) * t for i in range(3))
                item = dict(command)
                item["position"] = interp
                frames.append(item)
        previous = tuple(position)
    return frames


def _world_to_pixel(position: Any, table: tuple[int, int, int, int]) -> tuple[int, int]:
    x, y = float(position[0]), float(position[1])
    x_min, x_max = -0.35, 0.35
    y_min, y_max = -0.25, 0.25
    left, top, right, bottom = table
    px = left + int((x - x_min) / (x_max - x_min) * (right - left))
    py = bottom - int((y - y_min) / (y_max - y_min) * (bottom - top))
    return px, py


def _rgb255(value: Any, fallback: str | None = None) -> Color:
    if value is not None:
        return tuple(max(0, min(255, int(float(channel) * 255))) for channel in value[:3])  # type: ignore[return-value]
    if fallback == "red":
        return (220, 42, 36)
    if fallback == "blue":
        return (45, 90, 220)
    if fallback == "green":
        return (45, 165, 80)
    return (160, 160, 160)


FONT = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    "?": ("11110", "00001", "00001", "00110", "00100", "00000", "00100"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "A": ("01110", "10001", "10001", "11111", "10001", "10001", "10001"),
    "B": ("11110", "10001", "10001", "11110", "10001", "10001", "11110"),
    "C": ("01111", "10000", "10000", "10000", "10000", "10000", "01111"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "E": ("11111", "10000", "10000", "11110", "10000", "10000", "11111"),
    "F": ("11111", "10000", "10000", "11110", "10000", "10000", "10000"),
    "G": ("01111", "10000", "10000", "10111", "10001", "10001", "01110"),
    "H": ("10001", "10001", "10001", "11111", "10001", "10001", "10001"),
    "I": ("11111", "00100", "00100", "00100", "00100", "00100", "11111"),
    "J": ("00111", "00010", "00010", "00010", "00010", "10010", "01100"),
    "K": ("10001", "10010", "10100", "11000", "10100", "10010", "10001"),
    "L": ("10000", "10000", "10000", "10000", "10000", "10000", "11111"),
    "M": ("10001", "11011", "10101", "10101", "10001", "10001", "10001"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "O": ("01110", "10001", "10001", "10001", "10001", "10001", "01110"),
    "P": ("11110", "10001", "10001", "11110", "10000", "10000", "10000"),
    "Q": ("01110", "10001", "10001", "10001", "10101", "10010", "01101"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "S": ("01111", "10000", "10000", "01110", "00001", "00001", "11110"),
    "T": ("11111", "00100", "00100", "00100", "00100", "00100", "00100"),
    "U": ("10001", "10001", "10001", "10001", "10001", "10001", "01110"),
    "V": ("10001", "10001", "10001", "10001", "10001", "01010", "00100"),
    "W": ("10001", "10001", "10001", "10101", "10101", "10101", "01010"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
    "Z": ("11111", "00001", "00010", "00100", "01000", "10000", "11111"),
}
