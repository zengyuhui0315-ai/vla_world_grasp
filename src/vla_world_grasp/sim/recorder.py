"""Episode result and Isaac camera recording utilities."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from vla_world_grasp.utils.video import frames_to_video, write_png_array_rgb


class IsaacCameraRecorder:
    """Save frames from the IsaacLab camera sensor and encode them to mp4."""

    def __init__(
        self,
        video_name: str,
        videos_dir: str | Path = "outputs/videos",
        fps: int = 20,
        capture_every: int | None = None,
    ) -> None:
        self.video_name = _safe_video_name(video_name)
        self.videos_dir = Path(videos_dir)
        self.frames_dir = self.videos_dir / f"{self.video_name}_frames"
        self.mp4_path = self.videos_dir / f"{self.video_name}.mp4"
        self.fps = fps
        env_interval = os.environ.get("VLA_WORLD_GRASP_FRAME_INTERVAL")
        if capture_every is None and env_interval:
            capture_every = int(env_interval)
        self.capture_every = max(1, capture_every or 4)
        self._step_count = 0
        self._frame_count = 0
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir.mkdir(parents=True, exist_ok=True)

    def capture_from_scene(self, scene: Any, force: bool = False) -> Path | None:
        self._step_count += 1
        if not force and self._step_count % self.capture_every != 0:
            return None

        camera = scene.isaac_context.get("camera")
        if camera is None:
            raise RuntimeError("Isaac camera is not available; cannot record Isaac video")
        output = camera.data.output
        if "rgb" not in output:
            raise RuntimeError("Isaac camera produced no rgb output; cannot record Isaac video")
        rgb = output["rgb"][0]
        frame_path = self.frames_dir / f"frame_{self._frame_count:06d}.png"
        write_png_array_rgb(frame_path, rgb)
        self._frame_count += 1
        return frame_path

    def finalize(self) -> dict[str, Any]:
        if self._frame_count == 0:
            manifest = {
                "source": "isaac_camera",
                "frames_dir": str(self.frames_dir),
                "frame_count": 0,
                "fps": self.fps,
                "mp4_path": None,
                "ffmpeg": {
                    "available": None,
                    "encoded": False,
                    "message": "no Isaac camera frames were captured",
                },
            }
            manifest_path = self.videos_dir / f"{self.video_name}_manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
            manifest["manifest_path"] = str(manifest_path)
            return manifest
        ffmpeg_status = frames_to_video(self.frames_dir, self.mp4_path, fps=self.fps, timeout_seconds=120)
        manifest = {
            "source": "isaac_camera",
            "frames_dir": str(self.frames_dir),
            "frame_count": self._frame_count,
            "fps": self.fps,
            "mp4_path": str(self.mp4_path) if ffmpeg_status.get("encoded") else None,
            "ffmpeg": ffmpeg_status,
        }
        manifest_path = self.videos_dir / f"{self.video_name}_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        manifest["manifest_path"] = str(manifest_path)
        return manifest


def save_result(output_dir: str | Path, result: dict[str, Any]) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    result_path = path / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return result_path


def _safe_video_name(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in name.strip())
    return safe or "isaac_demo"


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dict__"):
        return value.__dict__
    return str(value)
