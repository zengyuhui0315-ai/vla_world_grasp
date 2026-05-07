"""Qwen2.5-VL grounding adapter for Chinese target localization."""

from __future__ import annotations

import json
import os
import re
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from vla_world_grasp.utils.projection import (
    bbox_center,
    bbox_contains_point,
    nearest_object_by_projected_center,
    project_point,
)
from vla_world_grasp.vla.base import BaseVLABackend
from vla_world_grasp.vla.schemas import VLAObservation, VLAResult, VLATarget


QWEN_GROUNDING_PROMPT = """你是机器人抓取系统中的视觉语言定位模块。
请根据图像和中文指令，找出机器人应该抓取的目标物体。

用户指令：
{instruction}

请只输出 JSON，不要输出 Markdown，不要输出解释性文本。
输出的第一个字符必须是 {{，最后一个字符必须是 }}。

JSON 格式：
{{
  "target_label_zh": "蓝色圆柱",
  "target_label_en": "blue cylinder",
  "bbox_2d": [x1, y1, x2, y2],
  "point_2d": [u, v],
  "confidence": 0.0,
  "reason_zh": "该物体是图中唯一的蓝色圆柱，符合用户指令。"
}}

如果无法可靠给出 bbox_2d 或 point_2d，也可以输出：
{{
  "target_object": "red cube",
  "color": "red",
  "shape": "cube",
  "grasp_hint": "top-down grasp at object center"
}}"""


COLOR_ZH = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    "white": "白色",
    "black": "黑色",
}
SHAPE_ZH = {
    "cube": "方块",
    "cylinder": "圆柱",
    "sphere": "球",
}


class QwenVLAdapter(BaseVLABackend):
    """Optional Qwen2.5-VL backend that only performs visual grounding."""

    name = "qwen_vl"

    def __init__(self, model_path: str | None = None, device: str = "cuda") -> None:
        self.model_path = model_path or os.environ.get("QWEN_MODEL_PATH") or os.environ.get("QWEN_VL_MODEL_PATH")
        self.device = device
        self._model = None
        self._processor = None

    def infer(self, observation: VLAObservation) -> VLAResult:
        model_path = str(observation.metadata.get("qwen_model_path") or self.model_path or "")
        if not model_path or not Path(model_path).exists():
            return self._failed(
                observation,
                f"qwen model path does not exist: {model_path or '<empty>'}",
            )
        if not observation.rgb_path:
            return self._failed(observation, "rgb_path is required for qwen_vl grounding")
        if find_spec("transformers") is None or find_spec("PIL") is None:
            return self._failed(observation, "optional dependencies transformers/PIL are unavailable")

        try:
            raw_output = self._run_model(observation, model_path)
            payload = _parse_json_object(raw_output)
        except Exception as exc:
            return self._failed(observation, f"qwen_vl inference failed: {exc}", raw_output=locals().get("raw_output", ""))

        try:
            parsed = _normalize_qwen_payload(payload)
            target_label_zh = parsed["target_label_zh"]
            target_label_en = parsed["target_label_en"]
            color = parsed["color"]
            shape = parsed["shape"]
            bbox_2d = parsed["bbox_2d"]
            point_2d = parsed["point_2d"]
            confidence = float(parsed["confidence"])
            reason_zh = parsed["reason_zh"]
            object_states = _object_state_dicts(observation.object_states)
            matched_distance = None
            semantic_match_used = False
            bbox_point_used_for_target = False
            matched_obj = _match_object_by_color_shape(
                object_states,
                color=color,
                shape=shape,
                label=target_label_en or target_label_zh,
            )
            if matched_obj is not None:
                semantic_match_used = True
            else:
                visual_point = point_2d
                if visual_point is None:
                    center = bbox_center(bbox_2d)
                    visual_point = None if center is None else [float(center[0]), float(center[1])]
                matched_obj = None
                if visual_point is not None:
                    matched_obj, matched_distance = nearest_object_by_projected_center(
                        visual_point,
                        object_states,
                        intrinsics=observation.camera_intrinsics,
                        camera_pose=observation.camera_pose,
                    )
                    bbox_point_used_for_target = matched_obj is not None
            if matched_obj is None:
                raise ValueError(f"could not match Qwen-VL output to object_states: color={color}, shape={shape}, label={target_label_en or target_label_zh}")
            matched_projected = project_point(
                matched_obj.get("position") or matched_obj.get("point_3d"),
                intrinsics=observation.camera_intrinsics,
                camera_pose=observation.camera_pose,
            )
            validation = _validate_qwen_visual_result(
                bbox_2d=bbox_2d,
                point_2d=point_2d,
                matched_projected_point=matched_projected,
            )
            target_name = str(matched_obj.get("name") or matched_obj.get("object_id"))
            target_position = _vector3(matched_obj.get("position") or matched_obj.get("point_3d"))
            metadata_common = {
                "model_path": model_path,
                "target_label_zh": target_label_zh,
                "target_label_en": target_label_en,
                "color": color,
                "shape": shape,
                "bbox_2d": bbox_2d,
                "point_2d": point_2d,
                "grasp_hint": parsed["grasp_hint"],
                "reason_zh": reason_zh,
                "matched_target_name": target_name,
                "matched_projected_distance_px": matched_distance,
                "semantic_match_used": semantic_match_used,
                "bbox_point_used_for_target": bbox_point_used_for_target,
                **validation,
                "raw_model_output": raw_output,
                "parsed_qwen_result": parsed,
                "fallback_used": False,
            }
            target = VLATarget(
                object_id=target_name,
                label=target_label_en or target_label_zh or target_name,
                point_3d=target_position,
                confidence=confidence,
                bbox_xyxy=None if bbox_2d is None else tuple(int(round(value)) for value in bbox_2d),
                point_2d=None if point_2d is None else (float(point_2d[0]), float(point_2d[1])),
                metadata={
                    "color": matched_obj.get("color", ""),
                    "shape": matched_obj.get("shape") or matched_obj.get("type", ""),
                    "size": matched_obj.get("size", (0.05, 0.05, 0.05)),
                    "source": "qwen_vl_grounding",
                    **metadata_common,
                },
            )
            return VLAResult(
                mode="grounding",
                backend=self.name,
                status="ok",
                target=target,
                message="target localized by Qwen2.5-VL grounding",
                metadata={
                    "instruction": observation.text,
                    "confidence": confidence,
                    **metadata_common,
                },
            )
        except Exception as exc:
            return self._failed(observation, f"invalid qwen_vl grounding output: {exc}", raw_output=raw_output)

    def _run_model(self, observation: VLAObservation, model_path: str) -> str:
        from PIL import Image
        from transformers import AutoProcessor

        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelCls
        except Exception:
            from transformers import AutoModelForVision2Seq as ModelCls

        if self._model is None or self._processor is None:
            self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
            self._model = ModelCls.from_pretrained(
                model_path,
                torch_dtype="auto",
                device_map=self.device if self.device else "auto",
                trust_remote_code=True,
            )

        image = Image.open(str(observation.rgb_path)).convert("RGB")
        prompt = QWEN_GROUNDING_PROMPT.format(instruction=observation.text)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(observation.rgb_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if find_spec("qwen_vl_utils") is not None:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to(self._model.device)
        generated_ids = self._model.generate(**inputs, max_new_tokens=256)
        trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        decoded = self._processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decoded[0] if decoded else ""

    def _failed(self, observation: VLAObservation, message: str, raw_output: str = "") -> VLAResult:
        return VLAResult(
            mode="grounding",
            backend=self.name,
            status="not_enabled",
            message=message,
            metadata={
                "instruction": observation.text,
                "model_path": str(observation.metadata.get("qwen_model_path") or self.model_path or ""),
                "fallback_used": True,
                "raw_model_output": raw_output,
            },
        )


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?", "", stripped, flags=re.IGNORECASE).strip()
        stripped = re.sub(r"```$", "", stripped).strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("model output JSON is not an object")
    return value


def _float_list(value: Any, expected: int) -> list[float] | None:
    if value is None or value == "":
        return None
    if not isinstance(value, (list, tuple)) or len(value) != expected:
        return None
    return [float(item) for item in value]


def _vector3(value: Any) -> tuple[float, float, float]:
    return (float(value[0]), float(value[1]), float(value[2]))


def _normalize_qwen_payload(payload: dict[str, Any]) -> dict[str, Any]:
    target_object = _clean_text(payload.get("target_object", ""))
    target_label_en = _clean_text(payload.get("target_label_en", "")) or target_object
    target_label_zh = _clean_text(payload.get("target_label_zh", ""))
    color = _clean_text(payload.get("color", "")).lower()
    shape = _clean_text(payload.get("shape", "")).lower()
    if (not color or not shape) and (target_label_en or target_label_zh):
        inferred_color, inferred_shape = _infer_color_shape(f"{target_label_en} {target_label_zh}")
        color = color or inferred_color
        shape = shape or inferred_shape
    if not target_label_zh and color and shape:
        target_label_zh = _target_label_zh(color, shape)
    if not target_label_en and color and shape:
        target_label_en = f"{color} {shape}"
    reason_zh = _clean_text(payload.get("reason_zh", ""))
    if not reason_zh:
        reason_zh = f"千问根据图像和中文指令识别目标为 {target_label_zh or target_label_en}。"
    return {
        "target_label_zh": target_label_zh,
        "target_label_en": target_label_en,
        "color": color,
        "shape": shape,
        "bbox_2d": _float_list(payload.get("bbox_2d"), expected=4),
        "point_2d": _float_list(payload.get("point_2d"), expected=2),
        "confidence": float(payload.get("confidence", 0.9 if target_object else 0.0) or 0.0),
        "reason_zh": reason_zh,
        "grasp_hint": _clean_text(payload.get("grasp_hint", "")),
    }


def _object_state_dicts(object_states: Any) -> list[dict[str, Any]]:
    rows = []
    for obj in object_states:
        if hasattr(obj, "__dict__"):
            rows.append(dict(obj.__dict__))
        elif isinstance(obj, dict):
            rows.append(dict(obj))
    return rows


def _match_object_by_color_shape(
    object_states: list[dict[str, Any]],
    color: str,
    shape: str,
    label: str = "",
) -> dict[str, Any] | None:
    if (not color or not shape) and label:
        inferred_color, inferred_shape = _infer_color_shape(label)
        color = color or inferred_color
        shape = shape or inferred_shape
    color = color.lower()
    shape = shape.lower()
    for obj in object_states:
        obj_color = _clean_text(obj.get("color", "")).lower()
        obj_shape = _clean_text(obj.get("type") or obj.get("shape") or "").lower()
        obj_name = _clean_text(obj.get("name") or obj.get("object_id") or "").lower()
        if obj_color == color and obj_shape == shape:
            return obj
        if _name_matches_color_shape(obj_name, color, shape):
            return obj
    return None


def _name_matches_color_shape(name: str, color: str, shape: str) -> bool:
    if not name or not color or not shape:
        return False
    tokens = [token for token in name.replace("-", "_").split("_") if token]
    return color in tokens and shape in tokens


def _infer_color_shape(text: str) -> tuple[str, str]:
    lowered = text.strip().lower().replace("-", " ")
    color = ""
    shape = ""
    for candidate, zh in COLOR_ZH.items():
        if candidate in lowered or zh in lowered:
            color = candidate
            break
    for candidate, zh in SHAPE_ZH.items():
        if candidate in lowered or zh in lowered:
            shape = candidate
            break
    return color, shape


def _target_label_zh(color: str, shape: str) -> str:
    return f"{COLOR_ZH.get(color, color)}{SHAPE_ZH.get(shape, shape)}"


def _clean_text(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _validate_qwen_visual_result(
    bbox_2d: list[float] | None,
    point_2d: list[float] | None,
    matched_projected_point: tuple[float, float] | None,
    max_distance_px: float = 80.0,
) -> dict[str, Any]:
    projected = None if matched_projected_point is None else [float(matched_projected_point[0]), float(matched_projected_point[1])]
    pixel_distance = None
    bbox_contains = bbox_contains_point(bbox_2d, projected)
    point_valid = None
    bbox_valid = None
    warning = ""
    if point_2d is not None and projected is not None:
        pixel_distance = (
            (float(point_2d[0]) - projected[0]) ** 2
            + (float(point_2d[1]) - projected[1]) ** 2
        ) ** 0.5
        point_valid = bool(pixel_distance <= max_distance_px)
    elif point_2d is not None:
        point_valid = False
    if bbox_2d is not None and projected is not None:
        bbox_valid = bool(bbox_contains)
    elif bbox_2d is not None:
        bbox_valid = False
    invalid = point_valid is False or bbox_valid is False
    if invalid:
        warning = "Qwen bbox/point does not align with matched object projection."
        point_valid = False
        bbox_valid = False
    return {
        "qwen_bbox_valid": bbox_valid,
        "qwen_point_valid": point_valid,
        "qwen_point_to_matched_object_px": None if pixel_distance is None else float(pixel_distance),
        "matched_object_projected_point": projected,
        "bbox_validation_warning": warning,
    }
