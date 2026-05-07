"""Run one Franka tabletop grasp episode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from vla_world_grasp.grasp.candidate_generator import (
    candidate_to_json,
    score_and_sort_candidates,
)
from vla_world_grasp.grasp.vla_to_candidates import vla_result_to_candidates
from vla_world_grasp.pipeline.visualize import plot_candidate_scores
from vla_world_grasp.sim.controller import execute_grasp
from vla_world_grasp.sim.recorder import IsaacCameraRecorder, save_result
from vla_world_grasp.sim.scene import GraspScene
from vla_world_grasp.sim.sensors import capture_observation
from vla_world_grasp.utils.projection import (
    backproject_pixel,
    bbox_center,
    default_camera_intrinsics,
    default_camera_pose,
    depth_at_pixel,
    nearest_object_by_3d_point,
    nearest_object_by_projected_center,
)
from vla_world_grasp.utils.visualization import save_grounding_overlay
from vla_world_grasp.utils.video import write_png_array_rgb
from vla_world_grasp.vla.manager import VLABackendManager
from vla_world_grasp.vla.schemas import CandidateScore, GraspCandidate, VLAResult, VLATarget


def run_episode(
    instruction: str,
    vla_backend: str = "mock",
    num_candidates: int = 16,
    seed: int | None = None,
    headless: bool = True,
    save_video: bool = False,
    output_dir: str | Path = "outputs/demo",
    sim_backend: str = "isaac",
    video_name: str = "isaac_demo",
    videos_dir: str | Path = "outputs/videos",
    visualize_candidates: bool = False,
    candidate_vis_duration: int = 80,
    clear_candidate_markers_before_grasp: bool = True,
    max_steps: int | None = None,
    frame_interval: int | None = None,
    debug_attach_on_grasp: bool = False,
    explain_scores: bool = True,
    save_candidate_scores_plot: bool = True,
    candidate_top_k: int = 5,
    target_highlight: bool = False,
    candidate_marker_visual_only: bool = True,
    qwen_model_path: str | None = None,
    qwen_device: str = "cuda",
    qwen_fallback_to_mock: bool = True,
    score_mode: str = "baseline",
    score_model_checkpoint: str | None = None,
) -> dict[str, Any]:
    print(f"[vla_world_grasp] Starting episode with sim_backend={sim_backend}, headless={headless}", flush=True)
    scene = GraspScene(headless=headless, seed=seed, sim_backend=sim_backend)
    try:
        print("[vla_world_grasp] Resetting scene...", flush=True)
        object_states = scene.reset_scene()
        normalized_backend = _normalize_backend_name(vla_backend)
        print(f"[vla_world_grasp] VLA backend: {normalized_backend}", flush=True)
        qwen_capture = {}
        if normalized_backend in {"qwen_vl", "qwen"}:
            qwen_capture = _save_qwen_camera_inputs(scene, video_name)
        print("[vla_world_grasp] Running VLA backend...", flush=True)
        observation = capture_observation(
            scene,
            instruction,
            image=qwen_capture.get("image"),
            depth=qwen_capture.get("depth"),
            rgb_path=qwen_capture.get("rgb_path"),
            depth_path=qwen_capture.get("depth_path"),
            camera_intrinsics=qwen_capture.get("camera_intrinsics") or default_camera_intrinsics(),
            camera_pose=qwen_capture.get("camera_pose") or default_camera_pose(),
        )
        observation.metadata["qwen_model_path"] = qwen_model_path
        print("[vla_world_grasp] Calling Qwen-VL grounding..." if normalized_backend in {"qwen_vl", "qwen"} else "[vla_world_grasp] Calling VLA localization...", flush=True)
        manager = VLABackendManager.from_name(
            normalized_backend,
            qwen_model_path=qwen_model_path,
            qwen_device=qwen_device,
            qwen_fallback_to_mock=qwen_fallback_to_mock,
        )
        vla_result = manager.infer(observation)
        if normalized_backend in {"qwen_vl", "qwen"}:
            vla_result = _resolve_qwen_target_from_grounding(
                vla_result=vla_result,
                object_states=object_states,
                depth=qwen_capture.get("depth"),
                intrinsics=qwen_capture.get("camera_intrinsics") or default_camera_intrinsics(),
                camera_pose=qwen_capture.get("camera_pose") or default_camera_pose(),
            )
            _print_qwen_grounding_logs(vla_result)
            _save_vla_result_log(video_name, instruction, vla_result, requested_backend="qwen_vl")
            _save_qwen_grounding_overlay(video_name, qwen_capture.get("rgb_path"), vla_result, instruction)
            vla_result = _qwen_grounding_as_target_localization(vla_result)
        instruction_parse = vla_result.metadata.get("instruction_parse", {})
        selection_reason = str(vla_result.metadata.get("selection_reason", ""))
        print(f"[vla_world_grasp] 中文指令: {instruction}", flush=True)
        print(
            "[vla_world_grasp] 解析结果: "
            f"{json.dumps(instruction_parse, ensure_ascii=False, sort_keys=True)}",
            flush=True,
        )
        print(f"[vla_world_grasp] 目标物体: {vla_result.target_id}", flush=True)
        print(f"[vla_world_grasp] 选择原因: {selection_reason}", flush=True)
        print(f"[vla_world_grasp] VLA target={vla_result.target_id}", flush=True)
        target_state = _find_target_state(object_states, vla_result.target_id)
        candidates = vla_result_to_candidates(
            vla_result,
            candidate_count=_normalize_candidate_count(num_candidates),
        )
        scored_candidates = score_and_sort_candidates(candidates, target_state, object_states)
        best_score = scored_candidates[0]
        print(f"[vla_world_grasp] Generated {len(scored_candidates)} candidates.", flush=True)
        _print_top_candidates(scored_candidates, top_k=candidate_top_k)
        print(f"[vla_world_grasp] Selected best candidate id={best_score.candidate.candidate_id}", flush=True)
        print(f"[vla_world_grasp] Best reason: {best_score.candidate.metadata.get('explanation', '')}", flush=True)
        candidate_log_path = _save_candidate_log(
            video_name=video_name,
            instruction=instruction,
            instruction_parse=instruction_parse,
            selection_reason=selection_reason,
            target_state=target_state,
            scored_candidates=scored_candidates,
            selected_score=best_score,
        )
        explanation_path = None
        if explain_scores:
            explanation_path = _save_candidate_explanation(
                video_name=video_name,
                instruction=instruction,
                instruction_parse=instruction_parse,
                selection_reason=selection_reason,
                target_state=target_state,
                scored_candidates=scored_candidates,
                selected_score=best_score,
                top_k=candidate_top_k,
            )
        candidate_score_plot = None
        if save_candidate_scores_plot:
            candidate_score_plot = plot_candidate_scores(
                candidates=[candidate_to_json(item.candidate, item) for item in scored_candidates],
                selected_candidate_id=int(best_score.candidate.candidate_id),
                output_path=Path("outputs/figures") / f"{video_name}_candidate_scores.png",
            )
        recorder = None
        if save_video and sim_backend == "isaac":
            print("[vla_world_grasp] Attaching Isaac camera recorder...", flush=True)
            recorder = IsaacCameraRecorder(video_name=video_name, videos_dir=videos_dir, capture_every=frame_interval)
            scene.attach_recorder(recorder)
            recorder.capture_from_scene(scene, force=True)
        if visualize_candidates or target_highlight:
            print("[vla_world_grasp] Candidate visualization is disabled in final demo mode.", flush=True)
        if clear_candidate_markers_before_grasp:
            print("[vla_world_grasp] Clearing debug markers before grasp...", flush=True)
            scene.clear_debug_markers(settle_steps=3)
        print("[vla_world_grasp] Executing top-down grasp...", flush=True)
        execution = execute_grasp(
            scene,
            best_score.candidate,
            debug_attach_on_grasp=debug_attach_on_grasp,
        )

        result = {
            "instruction": instruction,
            "vla_backend": vla_backend,
            "headless": headless,
            "seed": seed,
            "sim_backend": scene.backend,
            "target_name": execution.target_object,
            "target_type": execution.target_type,
            "debug_attach_on_grasp": execution.debug_attach_on_grasp,
            "attached": execution.attached,
            "attach_xy_threshold": execution.attach_xy_threshold,
            "attach_z_threshold": execution.attach_z_threshold,
            "attach_offset": execution.attach_offset,
            "target_initial_z": execution.initial_z,
            "target_final_z": execution.final_z,
            "object_states": object_states,
            "vla_result": _vla_result_to_dict(vla_result),
            "candidates": [candidate_to_json(score.candidate, score) for score in scored_candidates],
            "selected_candidate": _candidate_to_dict(best_score.candidate, best_score),
            "candidate_log_path": str(candidate_log_path),
            "candidate_explanation_path": None if explanation_path is None else str(explanation_path),
            "candidate_score_plot": None if candidate_score_plot is None else str(candidate_score_plot),
            "execution": execution.to_dict(),
            "success": execution.success,
            "score_mode": score_mode,
            "score_model_checkpoint": score_model_checkpoint,
        }

        if save_video:
            if sim_backend == "isaac":
                if recorder is None:
                    raise RuntimeError("Isaac recorder was not initialized")
                print("[vla_world_grasp] Encoding Isaac camera video...", flush=True)
                video = recorder.finalize()
            elif sim_backend == "mock":
                from vla_world_grasp.utils.video import create_debug_mock_video

                video = create_debug_mock_video(output_dir, result)
            else:
                raise RuntimeError(f"unsupported sim_backend for video: {sim_backend}")
            result["video"] = video
            result["video_path"] = video.get("mp4_path") or video.get("frames_dir")

        result_path = save_result(output_dir, result)
        result["result_path"] = str(result_path)
        return result
    finally:
        scene.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one VLA-to-Franka grasp episode.")
    parser.add_argument("--instruction", default="抓起红色方块")
    parser.add_argument("--vla_backend", default="mock")
    parser.add_argument("--qwen_model_path", default=None)
    parser.add_argument("--qwen_device", default="cuda")
    parser.add_argument("--qwen_fallback_to_mock", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save_video", action="store_true")
    parser.add_argument("--output_dir", default="outputs/demo")
    parser.add_argument("--sim_backend", choices=["isaac", "mock"], default="isaac")
    parser.add_argument("--video_name", default="isaac_demo")
    parser.add_argument("--videos_dir", default="outputs/videos")
    parser.add_argument("--visualize_candidates", dest="visualize_candidates", action="store_true", default=False)
    parser.add_argument("--no_visualize_candidates", dest="visualize_candidates", action="store_false")
    parser.add_argument("--candidate_vis_duration", type=int, default=80)
    parser.add_argument("--target_highlight", action="store_true", default=False)
    parser.add_argument("--candidate_marker_visual_only", dest="candidate_marker_visual_only", action="store_true", default=True)
    parser.add_argument("--no_candidate_marker_visual_only", dest="candidate_marker_visual_only", action="store_false")
    parser.add_argument(
        "--clear_candidate_markers_before_grasp",
        dest="clear_candidate_markers_before_grasp",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no_clear_candidate_markers_before_grasp",
        dest="clear_candidate_markers_before_grasp",
        action="store_false",
    )
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--frame_interval", type=int, default=None)
    parser.add_argument("--explain_scores", dest="explain_scores", action="store_true", default=True)
    parser.add_argument("--no_explain_scores", dest="explain_scores", action="store_false")
    parser.add_argument("--save_candidate_scores_plot", dest="save_candidate_scores_plot", action="store_true", default=True)
    parser.add_argument("--no_save_candidate_scores_plot", dest="save_candidate_scores_plot", action="store_false")
    parser.add_argument("--candidate_top_k", type=int, default=5)
    parser.add_argument("--score_mode", choices=("baseline", "model", "hybrid"), default="baseline")
    parser.add_argument("--score_model_checkpoint", default=None)
    parser.add_argument(
        "--debug_attach_on_grasp",
        action="store_true",
        help="Enable debug-only target attachment after a close-gripper proximity check.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    result = run_episode(
        instruction=args.instruction,
        vla_backend=args.vla_backend,
        num_candidates=args.num_candidates,
        seed=args.seed,
        headless=args.headless,
        save_video=args.save_video,
        output_dir=args.output_dir,
        sim_backend=args.sim_backend,
        video_name=args.video_name,
        videos_dir=args.videos_dir,
        visualize_candidates=args.visualize_candidates,
        candidate_vis_duration=args.candidate_vis_duration,
        clear_candidate_markers_before_grasp=args.clear_candidate_markers_before_grasp,
        max_steps=args.max_steps,
        frame_interval=args.frame_interval,
        debug_attach_on_grasp=args.debug_attach_on_grasp,
        explain_scores=args.explain_scores,
        save_candidate_scores_plot=args.save_candidate_scores_plot,
        candidate_top_k=args.candidate_top_k,
        target_highlight=args.target_highlight,
        candidate_marker_visual_only=args.candidate_marker_visual_only,
        qwen_model_path=args.qwen_model_path,
        qwen_device=args.qwen_device,
        qwen_fallback_to_mock=args.qwen_fallback_to_mock,
        score_mode=args.score_mode,
        score_model_checkpoint=args.score_model_checkpoint,
    )
    print(f"result_path={result['result_path']}")
    print(f"success={result['success']}")
    print(f"target={result['execution']['target_object']}")
    print(f"sim_backend={result['sim_backend']}")
    if result.get("video_path"):
        print(f"video_path={result['video_path']}")


def _normalize_backend_name(name: str) -> str:
    if name in {"mock", "mock-vla"}:
        return "mock_vla"
    if name in {"qwen", "qwen-vl"}:
        return "qwen_vl"
    return name


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


def _normalize_candidate_count(count: int) -> int:
    return max(12, min(24, int(count)))


def _vla_result_to_dict(result: VLAResult) -> dict[str, Any]:
    return {
        "mode": result.mode,
        "backend": result.backend,
        "status": result.status,
        "message": result.message,
        "target": None
        if result.target is None
        else {
            "object_id": result.target.object_id,
            "label": result.target.label,
            "point_3d": result.target.point_3d,
            "confidence": result.target.confidence,
            "bbox_xyxy": result.target.bbox_xyxy,
            "point_2d": result.target.point_2d,
            "metadata": result.target.metadata,
        },
        "metadata": result.metadata,
    }


def _candidate_to_dict(candidate: GraspCandidate, score: CandidateScore | None = None) -> dict[str, Any]:
    return candidate_to_json(candidate, score)


def _find_target_state(object_states: list[dict[str, Any]], target_name: str) -> dict[str, Any]:
    for obj in object_states:
        if obj.get("name") == target_name or obj.get("object_id") == target_name:
            return obj
    raise KeyError(f"target object not found in object_states: {target_name}")


def _save_qwen_camera_inputs(scene: Any, video_name: str) -> dict[str, Any]:
    output_dir = Path("outputs/debug")
    output_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = output_dir / f"{video_name}_qwen_input.png"
    depth_path = output_dir / f"{video_name}_qwen_depth.npy"
    print(f"[vla_world_grasp] Saving Qwen-VL input image: {rgb_path}", flush=True)
    if scene.backend != "isaac":
        print("[vla_world_grasp] WARNING: Qwen-VL requires Isaac camera RGB; no mock renderer fallback will be used.", flush=True)
        return {
            "rgb_path": None,
            "depth_path": None,
            "image": None,
            "depth": None,
            "camera_intrinsics": default_camera_intrinsics(),
            "camera_pose": default_camera_pose(),
        }
    camera = scene.isaac_context.get("camera")
    if camera is None:
        print("[vla_world_grasp] WARNING: Isaac camera is unavailable for Qwen-VL input.", flush=True)
        return {
            "rgb_path": None,
            "depth_path": None,
            "image": None,
            "depth": None,
            "camera_intrinsics": default_camera_intrinsics(),
            "camera_pose": default_camera_pose(),
        }
    camera.update(dt=scene.isaac_context["sim"].get_physics_dt())
    output = camera.data.output
    if "rgb" not in output:
        print("[vla_world_grasp] WARNING: Isaac camera produced no rgb output for Qwen-VL.", flush=True)
        return {
            "rgb_path": None,
            "depth_path": None,
            "image": None,
            "depth": None,
            "camera_intrinsics": default_camera_intrinsics(),
            "camera_pose": default_camera_pose(),
        }
    rgb = output["rgb"][0]
    write_png_array_rgb(rgb_path, rgb)
    depth = None
    depth_key = _find_depth_key(output)
    if depth_key is not None:
        depth = output[depth_key][0]
        if hasattr(depth, "detach"):
            depth = depth.detach().cpu().numpy()
        np.save(depth_path, depth)
    return {
        "rgb_path": str(rgb_path),
        "depth_path": None if depth is None else str(depth_path),
        "image": rgb,
        "depth": depth,
        "camera_intrinsics": default_camera_intrinsics(),
        "camera_pose": default_camera_pose(),
    }


def _resolve_qwen_target_from_grounding(
    vla_result: VLAResult,
    object_states: list[dict[str, Any]],
    depth: Any | None,
    intrinsics: dict[str, Any],
    camera_pose: dict[str, Any],
) -> VLAResult:
    if vla_result.target is None:
        return vla_result
    if bool(vla_result.metadata.get("semantic_match_used", False)):
        return vla_result
    point_2d = vla_result.target.point_2d
    if point_2d is None:
        point_2d = bbox_center(vla_result.target.bbox_xyxy)
    if point_2d is None:
        return vla_result

    matched_obj = None
    matched_distance = None
    pixel_depth = depth_at_pixel(depth, point_2d)
    if pixel_depth is not None:
        point_3d = backproject_pixel(point_2d, pixel_depth, intrinsics=intrinsics, camera_pose=camera_pose)
        if point_3d is not None:
            matched_obj, matched_distance = nearest_object_by_3d_point(point_3d, object_states)
    if matched_obj is None:
        matched_obj, matched_distance = nearest_object_by_projected_center(
            point_2d,
            object_states,
            intrinsics=intrinsics,
            camera_pose=camera_pose,
        )
    if matched_obj is None:
        return vla_result

    metadata = dict(vla_result.target.metadata)
    metadata.update(
        {
            "matched_target_name": matched_obj.get("name") or matched_obj.get("object_id"),
            "matched_distance": matched_distance,
            "matched_with_depth": pixel_depth is not None,
        }
    )
    target = VLATarget(
        object_id=str(matched_obj.get("name") or matched_obj.get("object_id")),
        label=vla_result.target.label,
        point_3d=tuple(float(value) for value in matched_obj["position"]),
        confidence=vla_result.target.confidence,
        bbox_xyxy=vla_result.target.bbox_xyxy,
        point_2d=(float(point_2d[0]), float(point_2d[1])),
        metadata=metadata,
    )
    result_metadata = dict(vla_result.metadata)
    result_metadata["matched_target_name"] = target.object_id
    result_metadata["point_2d"] = [float(point_2d[0]), float(point_2d[1])]
    return VLAResult(
        mode=vla_result.mode,
        backend=vla_result.backend,
        status=vla_result.status,
        target=target,
        action_proposal=vla_result.action_proposal,
        message=vla_result.message,
        metadata=result_metadata,
    )


def _print_qwen_grounding_logs(vla_result: VLAResult) -> None:
    metadata = vla_result.metadata
    target_metadata = vla_result.target.metadata if vla_result.target is not None else {}
    parsed = metadata.get("parsed_qwen_result", target_metadata.get("parsed_qwen_result", {}))
    color = metadata.get("color", target_metadata.get("color", ""))
    shape = metadata.get("shape", target_metadata.get("shape", ""))
    label_zh = metadata.get("target_label_zh", target_metadata.get("target_label_zh", ""))
    print(f"[vla_world_grasp] Qwen raw output: {metadata.get('raw_model_output', target_metadata.get('raw_model_output', ''))}", flush=True)
    print(f"[vla_world_grasp] Parsed Qwen result: {json.dumps(parsed, ensure_ascii=False, sort_keys=True)}", flush=True)
    print(f"[vla_world_grasp] Qwen semantic result: color={color} shape={shape} label={label_zh}", flush=True)
    print(
        "[vla_world_grasp] "
        f"Qwen color={color} "
        f"shape={shape}",
        flush=True,
    )
    print(f"[vla_world_grasp] Semantic matched target object={metadata.get('matched_target_name', target_metadata.get('matched_target_name', vla_result.target_id))}", flush=True)
    print(f"[vla_world_grasp] Qwen raw point_2d={metadata.get('point_2d', target_metadata.get('point_2d'))}", flush=True)
    print(f"[vla_world_grasp] Matched object projected point={metadata.get('matched_object_projected_point', target_metadata.get('matched_object_projected_point'))}", flush=True)
    print(f"[vla_world_grasp] qwen_point_to_matched_object_px={metadata.get('qwen_point_to_matched_object_px', target_metadata.get('qwen_point_to_matched_object_px'))}", flush=True)
    print(f"[vla_world_grasp] qwen_bbox_valid={metadata.get('qwen_bbox_valid', target_metadata.get('qwen_bbox_valid'))}", flush=True)
    if bool(metadata.get("semantic_match_used", target_metadata.get("semantic_match_used", False))):
        print("[vla_world_grasp] Using semantic object_states match for target.", flush=True)
    print(f"[vla_world_grasp] Matched target object={metadata.get('matched_target_name', target_metadata.get('matched_target_name', vla_result.target_id))}", flush=True)
    print(f"[vla_world_grasp] fallback_used={bool(metadata.get('fallback_used', False))}", flush=True)


def _save_vla_result_log(video_name: str, instruction: str, vla_result: VLAResult, requested_backend: str) -> Path:
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{video_name}_vla_result.json"
    metadata = dict(vla_result.metadata)
    target_metadata = dict(vla_result.target.metadata) if vla_result.target is not None else {}
    payload = {
        "instruction": instruction,
        "backend": requested_backend,
        "model_path": metadata.get("model_path", target_metadata.get("model_path", "")),
        "actual_backend": vla_result.backend,
        "mode": "grounding" if requested_backend == "qwen_vl" else vla_result.mode,
        "actual_mode": vla_result.mode,
        "target_label_zh": metadata.get("target_label_zh", target_metadata.get("target_label_zh", "")),
        "target_label_en": metadata.get("target_label_en", target_metadata.get("target_label_en", "")),
        "color": metadata.get("color", target_metadata.get("color", "")),
        "shape": metadata.get("shape", target_metadata.get("shape", "")),
        "bbox_2d": metadata.get("bbox_2d", target_metadata.get("bbox_2d")),
        "point_2d": metadata.get("point_2d", target_metadata.get("point_2d")),
        "matched_target_name": metadata.get("matched_target_name", target_metadata.get("matched_target_name", vla_result.target_id)),
        "confidence": metadata.get("confidence", None if vla_result.target is None else vla_result.target.confidence),
        "reason_zh": metadata.get("reason_zh", target_metadata.get("reason_zh", "")),
        "semantic_match_used": bool(metadata.get("semantic_match_used", target_metadata.get("semantic_match_used", False))),
        "bbox_point_used_for_target": bool(metadata.get("bbox_point_used_for_target", target_metadata.get("bbox_point_used_for_target", False))),
        "qwen_bbox_valid": metadata.get("qwen_bbox_valid", target_metadata.get("qwen_bbox_valid")),
        "qwen_point_valid": metadata.get("qwen_point_valid", target_metadata.get("qwen_point_valid")),
        "qwen_point_to_matched_object_px": metadata.get("qwen_point_to_matched_object_px", target_metadata.get("qwen_point_to_matched_object_px")),
        "matched_object_projected_point": metadata.get("matched_object_projected_point", target_metadata.get("matched_object_projected_point")),
        "bbox_validation_warning": metadata.get("bbox_validation_warning", target_metadata.get("bbox_validation_warning", "")),
        "fallback_used": bool(metadata.get("fallback_used", False)),
        "fallback_reason": metadata.get("fallback_reason", ""),
        "raw_model_output": metadata.get("raw_model_output", target_metadata.get("raw_model_output", "")),
        "message": vla_result.message,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _save_qwen_grounding_overlay(video_name: str, image_path: str | None, vla_result: VLAResult, instruction: str) -> Path | None:
    if not image_path:
        return None
    metadata = dict(vla_result.metadata)
    target_metadata = dict(vla_result.target.metadata) if vla_result.target is not None else {}
    bbox_2d = metadata.get("bbox_2d", target_metadata.get("bbox_2d"))
    point_2d = metadata.get("point_2d", target_metadata.get("point_2d"))
    if bbox_2d is None and point_2d is None:
        return None
    output_path = Path("outputs/figures") / f"{video_name}_grounding_overlay.png"
    saved = save_grounding_overlay(
        image_path=image_path,
        output_path=output_path,
        bbox_2d=bbox_2d,
        point_2d=point_2d,
        target_label_zh=str(metadata.get("target_label_zh", target_metadata.get("target_label_zh", ""))),
        confidence=metadata.get("confidence", None if vla_result.target is None else vla_result.target.confidence),
        reason_zh=str(metadata.get("reason_zh", target_metadata.get("reason_zh", ""))),
        instruction=instruction,
        matched_target_name=str(metadata.get("matched_target_name", target_metadata.get("matched_target_name", ""))),
        matched_object_projected_point=metadata.get("matched_object_projected_point", target_metadata.get("matched_object_projected_point")),
        qwen_bbox_valid=metadata.get("qwen_bbox_valid", target_metadata.get("qwen_bbox_valid")),
        warning=str(metadata.get("bbox_validation_warning", target_metadata.get("bbox_validation_warning", ""))),
    )
    if saved is not None:
        print(f"[vla_world_grasp] Saved Qwen-VL grounding overlay: {saved}", flush=True)
    return saved


def _qwen_grounding_as_target_localization(vla_result: VLAResult) -> VLAResult:
    if vla_result.backend != "qwen_vl" or vla_result.mode != "grounding":
        return vla_result
    return VLAResult(
        mode="target_localization",
        backend=vla_result.backend,
        status=vla_result.status,
        target=vla_result.target,
        action_proposal=vla_result.action_proposal,
        message=vla_result.message,
        metadata=dict(vla_result.metadata),
    )


def _find_depth_key(output: dict[str, Any]) -> str | None:
    for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
        if key in output:
            return key
    return None


def _candidate_stability_key(candidate: GraspCandidate) -> tuple[int, tuple[float, float, float], float]:
    return (
        int(candidate.candidate_id),
        tuple(round(float(value), 6) for value in candidate.position),
        round(float(candidate.yaw), 6),
    )


def _position_delta(before: tuple[float, float, float], after: tuple[float, float, float]) -> float:
    return sum((float(a) - float(b)) ** 2 for a, b in zip(before, after)) ** 0.5


def _print_top_candidates(scored_candidates: list[CandidateScore], top_k: int = 5) -> None:
    for rank, item in enumerate(scored_candidates[:top_k], start=1):
        candidate = item.candidate
        terms = item.components
        print(
            "[vla_world_grasp] "
            f"Candidate rank {rank}: id={candidate.candidate_id}, "
            f"score={item.score:.4f}, pos={list(candidate.position)}, yaw={candidate.yaw:.3f}",
            flush=True,
        )
        print(
            "[vla_world_grasp]   "
            f"center={terms.get('center_score', 0.0):.3f}, "
            f"reachability={terms.get('reachability_score', 0.0):.3f}, "
            f"collision={terms.get('collision_score', 0.0):.3f}, "
            f"height={terms.get('height_score', 0.0):.3f}, "
            f"width={terms.get('gripper_width_score', 0.0):.3f}, "
            f"yaw={terms.get('yaw_score', 0.0):.3f}",
            flush=True,
        )
        print(f"[vla_world_grasp]   reason: {candidate.metadata.get('explanation', '')}", flush=True)


def _save_candidate_log(
    video_name: str,
    instruction: str,
    instruction_parse: Any,
    selection_reason: str,
    target_state: dict[str, Any],
    scored_candidates: list[CandidateScore],
    selected_score: CandidateScore,
) -> Path:
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{video_name}_candidates.json"
    payload = {
        "instruction": instruction,
        "instruction_parse": instruction_parse,
        "selection_reason": selection_reason,
        "target_name": target_state.get("name") or target_state.get("object_id"),
        "target_position": target_state["position"],
        "candidates": [candidate_to_json(item.candidate, item) for item in scored_candidates],
        "all_candidates": [candidate_to_json(item.candidate, item) for item in scored_candidates],
        "selected_candidate": candidate_to_json(selected_score.candidate, selected_score),
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _save_candidate_explanation(
    video_name: str,
    instruction: str,
    instruction_parse: Any,
    selection_reason: str,
    target_state: dict[str, Any],
    scored_candidates: list[CandidateScore],
    selected_score: CandidateScore,
    top_k: int,
) -> Path:
    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"{video_name}_explanation.txt"
    selected = selected_score.candidate
    lines = [
        f"中文指令：{instruction}",
        f"解析目标：{json.dumps(instruction_parse, ensure_ascii=False, sort_keys=True)}",
        f"目标物体：{target_state.get('name') or target_state.get('object_id')}",
        f"选择原因：{selection_reason}",
        "",
        "评分公式：",
        "score = 0.30 * center_score + 0.20 * reachability_score + 0.20 * collision_score + 0.15 * height_score + 0.10 * gripper_width_score + 0.05 * yaw_score",
        "",
        f"Top-{top_k} 候选动作得分：",
    ]
    for rank, item in enumerate(scored_candidates[:top_k], start=1):
        candidate = item.candidate
        terms = item.components
        lines.extend(
            [
                f"{rank}. candidate_id={candidate.candidate_id}, score={item.score:.4f}, position={list(candidate.position)}, yaw={candidate.yaw:.3f}",
                (
                    "   score_terms: "
                    f"center={terms.get('center_score', 0.0):.3f}, "
                    f"reachability={terms.get('reachability_score', 0.0):.3f}, "
                    f"collision={terms.get('collision_score', 0.0):.3f}, "
                    f"height={terms.get('height_score', 0.0):.3f}, "
                    f"width={terms.get('gripper_width_score', 0.0):.3f}, "
                    f"yaw={terms.get('yaw_score', 0.0):.3f}"
                ),
                f"   解释：{candidate.metadata.get('explanation', '')}",
            ]
        )
    lines.extend(
        [
            "",
            f"为什么选择最终动作：候选 {selected.candidate_id} 的 baseline_score 最高，为 {selected_score.score:.4f}。",
            f"最佳候选解释：{selected.metadata.get('explanation', '')}",
            (
                "最终执行动作：Franka 使用 top-down 抓取，移动到 "
                f"{list(selected.position)}，yaw={selected.yaw:.3f}，夹爪宽度={selected.gripper_width:.3f}。"
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
