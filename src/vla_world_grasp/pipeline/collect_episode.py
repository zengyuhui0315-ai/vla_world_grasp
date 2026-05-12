"""Collect one Isaac camera grasp-success episode for score-model training."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from random import Random
from typing import Any

from vla_world_grasp.grasp.candidate_generator import candidate_to_json, score_and_sort_candidates
from vla_world_grasp.grasp.vla_to_candidates import vla_result_to_candidates
from vla_world_grasp.sim.controller import execute_grasp
from vla_world_grasp.sim.objects import create_default_objects, create_randomized_objects
from vla_world_grasp.sim.recorder import IsaacCameraRecorder
from vla_world_grasp.sim.scene import GraspScene
from vla_world_grasp.sim.sensors import capture_observation
from vla_world_grasp.utils.dataset_io import episode_dir, write_json, write_text
from vla_world_grasp.utils.video import write_png_array_rgb
from vla_world_grasp.vla.manager import VLABackendManager
from vla_world_grasp.vla.schemas import CandidateScore


COLOR_ZH: dict[str, str] = {
    "red": "红色",
    "blue": "蓝色",
    "green": "绿色",
    "yellow": "黄色",
    "white": "白色",
    "black": "黑色",
}

TYPE_ZH: dict[str, str] = {
    "cube": "方块",
    "cylinder": "圆柱",
    "sphere": "球",
}

POSITION_INSTRUCTIONS: tuple[str, ...] = (
    "抓起中间的物体",
    "抓起最左边的物体",
)


def collect_episode(
    episode_index: int,
    output_raw_dir: str | Path,
    scene_seed: int,
    candidates_per_episode: int = 8,
    headless: bool = True,
    debug_attach_on_grasp: bool = False,
    save_video_per_episode: bool = False,
    attach_mode: str = "no_attach",
    randomize_color_shape: bool = True,
    randomize_gripper_width: bool = True,
    randomize_candidate_offsets: bool = True,
    min_object_distance: float = 0.10,
    candidate_eval_mode: str = "selected_only",
    candidate_eval_k: int = 4,
    reset_per_candidate: bool = True,
    scene: GraspScene | None = None,
    close_scene: bool = True,
) -> dict[str, Any]:
    """Run one Isaac episode and save raw training artifacts."""

    ep_dir = episode_dir(output_raw_dir, episode_index)
    log_prefix = f"[vla_world_grasp] Episode {episode_index:06d}:"
    if candidate_eval_mode != "selected_only":
        save_video_per_episode = False
    if scene is None:
        scene = GraspScene(headless=headless, seed=scene_seed, sim_backend="isaac")
    else:
        scene.seed = scene_seed
    print(f"{log_prefix} reset scene start", flush=True)
    scene.prepare_for_episode(clear_objects=False)
    print(f"{log_prefix} reset scene done", flush=True)
    print(f"{log_prefix} sampling randomized objects start", flush=True)
    scene_objects = (
        create_randomized_objects(
            scene_seed,
            min_object_distance=min_object_distance,
            max_position_attempts=100,
            max_scene_attempts=300,
            log_prefix=f"[vla_world_grasp] Episode {episode_index:06d}:",
        )
        if randomize_color_shape
        else create_default_objects(scene_seed)
    )
    print(f"{log_prefix} sampled objects = {_objects_summary(scene_objects)}", flush=True)
    print(f"{log_prefix} generating instruction", flush=True)
    instruction, requested_target_name = _sample_instruction(scene_seed, scene_objects)
    recorder = None
    try:
        scene.detach_object_from_gripper()
        print(f"{log_prefix} spawning objects start", flush=True)
        object_states = scene.reset_scene(objects=scene_objects)
        print(f"{log_prefix} spawning objects done", flush=True)
        scene.clear_debug_markers(settle_steps=0)
        scene.step(3)
        if save_video_per_episode:
            recorder = IsaacCameraRecorder(
                video_name=f"dataset_episode_{episode_index:06d}",
                videos_dir=ep_dir,
                capture_every=5,
            )
            scene.attach_recorder(recorder)
            recorder.capture_from_scene(scene, force=True)

        rgb_path, depth_path = _save_isaac_camera_observation(scene, ep_dir)
        print(f"{log_prefix} reading object states", flush=True)
        object_states_path = write_json(ep_dir / "object_states.json", object_states)
        instruction_path = write_text(ep_dir / "instruction.txt", instruction + "\n")

        observation = capture_observation(scene, instruction)
        print(f"{log_prefix} running mock_vla", flush=True)
        vla_result = VLABackendManager.from_name("mock_vla").infer(observation)
        vla_result.metadata["scene_seed"] = scene_seed
        vla_result.metadata["randomize_gripper_width"] = randomize_gripper_width
        vla_result.metadata["randomize_candidate_offsets"] = randomize_candidate_offsets
        parsed_instruction = vla_result.metadata.get("instruction_parse", {})
        parsed_instruction_path = write_json(ep_dir / "parsed_instruction.json", parsed_instruction)
        target_state = _find_target_state(object_states, vla_result.target_id)

        print(f"{log_prefix} generating candidates", flush=True)
        all_candidates = vla_result_to_candidates(vla_result, candidate_count=candidates_per_episode)
        scored_all = score_and_sort_candidates(all_candidates, target_state, object_states)
        scored_candidates = scored_all[: max(1, int(candidates_per_episode))]
        selected_score = scored_candidates[0]
        candidates_json = [candidate_to_json(item.candidate, item) for item in scored_candidates]
        candidate_json_path = write_json(
            ep_dir / "candidates.json",
            {
                "instruction": instruction,
                "parsed_instruction": parsed_instruction,
                "target_name": vla_result.target_id,
                "selection_reason": vla_result.metadata.get("selection_reason", ""),
                "candidates": candidates_json,
                "selected_candidate": candidate_to_json(selected_score.candidate, selected_score),
            },
        )

        print(f"{log_prefix} executing grasp", flush=True)
        eval_scores = _select_candidate_eval_scores(
            scored_candidates,
            mode=candidate_eval_mode,
            k=candidate_eval_k,
            seed=scene_seed,
        )
        candidate_eval_results: dict[int, dict[str, Any]] = {}
        for eval_score in eval_scores:
            candidate_id = int(eval_score.candidate.candidate_id)
            print(
                f"[vla_world_grasp] Candidate eval episode={episode_index:06d} candidate_id={candidate_id}",
                flush=True,
            )
            if reset_per_candidate:
                print("[vla_world_grasp] Resetting scene for candidate eval...", flush=True)
                scene.detach_object_from_gripper()
                scene.reset_scene(objects=deepcopy(scene_objects))
                scene.clear_debug_markers(settle_steps=0)
                scene.step(3)
            print("[vla_world_grasp] Executing candidate...", flush=True)
            try:
                eval_execution = execute_grasp(
                    scene,
                    eval_score.candidate,
                    debug_attach_on_grasp=debug_attach_on_grasp,
                )
                print(f"[vla_world_grasp] candidate_success={eval_execution.success}", flush=True)
                candidate_eval_results[candidate_id] = {
                    "candidate_id": candidate_id,
                    "success": bool(eval_execution.success),
                    "attached": bool(eval_execution.attached),
                    "target_initial_z": float(eval_execution.initial_z),
                    "target_final_z": float(eval_execution.final_z),
                    "execution": eval_execution.to_dict(),
                    "error_msg": "",
                }
            except Exception as exc:
                if isinstance(exc, TimeoutError):
                    raise
                error_msg = f"{type(exc).__name__}: {exc}"
                print(
                    f"[vla_world_grasp] WARNING: candidate eval failed "
                    f"episode={episode_index:06d} candidate_id={candidate_id}: {error_msg}",
                    flush=True,
                )
                failure_execution = _candidate_failure_execution(
                    target_state=target_state,
                    candidate_target_id=vla_result.target_id,
                    debug_attach_on_grasp=debug_attach_on_grasp,
                    error_msg=error_msg,
                )
                candidate_eval_results[candidate_id] = {
                    "candidate_id": candidate_id,
                    "success": False,
                    "attached": False,
                    "target_initial_z": failure_execution["initial_z"],
                    "target_final_z": failure_execution["final_z"],
                    "execution": failure_execution,
                    "error_msg": error_msg,
                }
        selected_candidate_id = int(selected_score.candidate.candidate_id)
        if selected_candidate_id in candidate_eval_results:
            primary_candidate_id = selected_candidate_id
        elif candidate_eval_results:
            primary_candidate_id = next(iter(candidate_eval_results))
        else:
            raise RuntimeError("no candidate was evaluated")
        execution_dict = candidate_eval_results[primary_candidate_id]["execution"]
        execution = _ExecutionView(execution_dict)
        print(f"[vla_world_grasp] Episode {episode_index:06d} finished.", flush=True)
        print(f"[vla_world_grasp] success={execution.success}", flush=True)
        print(f"{log_prefix} saving data", flush=True)
        print("[vla_world_grasp] Saving episode data...", flush=True)
        video_manifest = _finalize_recorder(recorder) if recorder is not None else None
        result = {
            "episode_id": episode_index,
            "scene_seed": scene_seed,
            "instruction_zh": instruction,
            "parsed_instruction": parsed_instruction,
            "target_name": execution.target_object,
            "target_type": execution.target_type,
            "target_color": target_state.get("color", ""),
            "requested_target_name": requested_target_name,
            "debug_attach_on_grasp": bool(debug_attach_on_grasp),
            "attach_mode": attach_mode,
            "selected_candidate": candidate_to_json(selected_score.candidate, selected_score),
            "success": execution.success,
            "attached": execution.attached,
            "natural_physics_success": bool(execution.success) if not debug_attach_on_grasp else None,
            "target_initial_z": execution.initial_z,
            "target_final_z": execution.final_z,
            "execution": execution.to_dict(),
            "candidate_eval_mode": candidate_eval_mode,
            "candidate_eval_k": int(candidate_eval_k),
            "reset_per_candidate": bool(reset_per_candidate),
            "candidate_eval_results": list(candidate_eval_results.values()),
            "video": video_manifest,
            "metadata": {
                "sim_backend": "isaac",
                "rgb_source": "isaac_camera",
                "depth_source": "isaac_camera",
                "candidate_success_labels": "candidate_eval rows have real candidate success; non-executed rows leave success empty",
                "label_policy": "candidate-level eval labels when candidate_eval_mode is top_k/random_k; selected-only otherwise",
                "candidate_markers": "disabled",
            },
        }
        result_path = write_json(ep_dir / "result.json", result)
        metadata_path = write_json(
            ep_dir / "metadata.json",
            {
                "episode_id": f"episode_{episode_index:06d}",
                "scene_seed": scene_seed,
                "instruction_path": str(instruction_path),
                "parsed_instruction_path": str(parsed_instruction_path),
                "rgb_path": str(rgb_path),
                "depth_path": str(depth_path),
                "object_states_path": str(object_states_path),
                "candidate_json_path": str(candidate_json_path),
                "result_path": str(result_path),
                "no_candidate_markers": True,
            },
        )
        print(f"[vla_world_grasp] Saved episode data: {ep_dir}", flush=True)
        return {
            "episode_id": episode_index,
            "scene_seed": scene_seed,
            "instruction_zh": instruction,
            "parsed_instruction": parsed_instruction,
            "vla_result": vla_result,
            "target_state": target_state,
            "target_name": execution.target_object,
            "target_type": execution.target_type,
            "target_color": target_state.get("color", ""),
            "debug_attach_on_grasp": bool(debug_attach_on_grasp),
            "attach_mode": attach_mode,
            "candidates": candidates_json,
            "scored_candidates": scored_candidates,
            "candidate_eval_results": candidate_eval_results,
            "candidate_eval_mode": candidate_eval_mode,
            "selected_candidate_id": selected_candidate_id,
            "success": bool(execution.success),
            "attached": bool(execution.attached),
            "natural_physics_success": bool(execution.success) if not debug_attach_on_grasp else "",
            "target_initial_z": float(execution.initial_z),
            "target_final_z": float(execution.final_z),
            "rgb_path": str(rgb_path),
            "depth_path": str(depth_path),
            "object_states_path": str(object_states_path),
            "candidate_json_path": str(candidate_json_path),
            "result_json_path": str(result_path),
            "metadata_path": str(metadata_path),
        }
    finally:
        if close_scene:
            print(f"[vla_world_grasp] Closing Isaac app for episode {episode_index:06d}...", flush=True)
            scene.close()
            print(f"[vla_world_grasp] Closed Isaac app for episode {episode_index:06d}.", flush=True)


def episode_to_csv_rows(episode: dict[str, Any]) -> list[dict[str, Any]]:
    target = episode["target_state"]
    parsed = episode["parsed_instruction"]
    rows = []
    eval_results = episode.get("candidate_eval_results", {})
    for rank, item in enumerate(episode["scored_candidates"], start=1):
        assert isinstance(item, CandidateScore)
        candidate = item.candidate
        terms = item.components
        selected = int(candidate.candidate_id) == episode["selected_candidate_id"]
        eval_result = eval_results.get(int(candidate.candidate_id), {})
        executed = bool(eval_result)
        label_source = "candidate_eval" if executed and episode.get("candidate_eval_mode") != "selected_only" else ("executed" if executed else "not_executed")
        rows.append(
            {
                "episode_id": f"episode_{int(episode['episode_id']):06d}",
                "scene_seed": episode["scene_seed"],
                "instruction_zh": episode["instruction_zh"],
                "parsed_instruction_json": json.dumps(parsed, ensure_ascii=False, sort_keys=True),
                "normalized_target_label_zh": parsed.get("normalized_target_label_zh", ""),
                "normalized_target_label_en": parsed.get("normalized_target_label_en", ""),
                "target_name": target.get("name") or target.get("object_id"),
                "target_type": target.get("type") or target.get("shape"),
                "target_color": target.get("color", ""),
                "target_position_x": target["position"][0],
                "target_position_y": target["position"][1],
                "target_position_z": target["position"][2],
                "candidate_id": int(candidate.candidate_id),
                "candidate_rank": rank,
                "action_x": candidate.position[0],
                "action_y": candidate.position[1],
                "action_z": candidate.position[2],
                "dx_to_target": candidate.metadata.get("dx_to_target", ""),
                "dy_to_target": candidate.metadata.get("dy_to_target", ""),
                "dz_to_target": candidate.metadata.get("dz_to_target", ""),
                "distance_to_target_center": candidate.metadata.get("distance_to_target_center", ""),
                "neighbor_clearance": candidate.metadata.get("neighbor_clearance", ""),
                "table_clearance": candidate.metadata.get("table_clearance", ""),
                "yaw": candidate.yaw,
                "gripper_width": candidate.gripper_width,
                "approach_height": candidate.pregrasp_height,
                "center_score": terms.get("center_score", 0.0),
                "reachability_score": terms.get("reachability_score", 0.0),
                "collision_score": terms.get("collision_score", 0.0),
                "height_score": terms.get("height_score", 0.0),
                "gripper_width_score": terms.get("gripper_width_score", 0.0),
                "yaw_score": terms.get("yaw_score", 0.0),
                "baseline_score": item.score,
                "selected": selected,
                "executed": executed,
                "label_available": executed,
                "label_source": label_source,
                "success": bool(eval_result["success"]) if executed else "",
                "debug_attach_on_grasp": bool(episode.get("debug_attach_on_grasp", False)),
                "attach_mode": episode.get("attach_mode", ""),
                "attached": bool(eval_result.get("attached", False)) if executed else "",
                "natural_physics_success": bool(eval_result["success"]) if executed and not episode.get("debug_attach_on_grasp", False) else "",
                "rgb_path": episode["rgb_path"],
                "depth_path": episode["depth_path"],
                "object_states_path": episode["object_states_path"],
                "candidate_json_path": episode["candidate_json_path"],
                "error_msg": eval_result.get("error_msg", "") if executed else "",
            }
        )
    return rows


def error_to_csv_row(
    episode_index: int,
    scene_seed: int,
    error_msg: str,
    attach_mode: str = "",
    debug_attach_on_grasp: bool | str = "",
) -> dict[str, Any]:
    return {
        "episode_id": f"episode_{episode_index:06d}",
        "scene_seed": scene_seed,
        "selected": "",
        "executed": "",
        "label_available": "",
        "label_source": "",
        "success": "",
        "debug_attach_on_grasp": debug_attach_on_grasp,
        "attach_mode": attach_mode,
        "error_msg": error_msg,
    }


def _finalize_recorder(recorder: IsaacCameraRecorder) -> dict[str, Any]:
    try:
        return recorder.finalize()
    except Exception as exc:
        print(f"[vla_world_grasp] WARNING: video finalize failed: {exc}", flush=True)
        return {
            "source": "isaac_camera",
            "frames_dir": str(recorder.frames_dir),
            "mp4_path": None,
            "ffmpeg": {
                "encoded": False,
                "message": str(exc),
            },
        }


def _sample_instruction(scene_seed: int, objects: list[Any]) -> tuple[str, str]:
    rng = Random(scene_seed)
    target = objects[rng.randrange(len(objects))]
    color_zh = COLOR_ZH.get(target.color, target.color)
    type_zh = TYPE_ZH.get(target.type, target.type)
    return f"抓起{color_zh}{type_zh}", target.name


class _ExecutionView:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.success = bool(payload["success"])
        self.target_object = str(payload["target_object"])
        self.target_type = str(payload["target_type"])
        self.debug_attach_on_grasp = bool(payload["debug_attach_on_grasp"])
        self.attached = bool(payload["attached"])
        self.initial_z = float(payload.get("initial_z", payload.get("target_initial_z", 0.0)))
        self.final_z = float(payload.get("final_z", payload.get("target_final_z", 0.0)))

    def to_dict(self) -> dict[str, Any]:
        return self._payload


def _select_candidate_eval_scores(
    scored_candidates: list[CandidateScore],
    mode: str,
    k: int,
    seed: int,
) -> list[CandidateScore]:
    if not scored_candidates:
        return []
    if mode == "selected_only":
        return [scored_candidates[0]]
    count = max(1, min(int(k), len(scored_candidates)))
    if mode == "top_k":
        return scored_candidates[:count]
    if mode == "random_k":
        rng = Random(seed + 7919)
        indices = list(range(len(scored_candidates)))
        selected_indices = sorted(rng.sample(indices, count))
        return [scored_candidates[index] for index in selected_indices[:count]]
    raise ValueError(f"unsupported candidate_eval_mode: {mode}")


def _objects_summary(objects: list[Any]) -> str:
    parts = []
    for obj in objects:
        position = ", ".join(f"{value:.3f}" for value in obj.position)
        parts.append(f"{obj.name}(color={obj.color}, type={obj.type}, position=[{position}])")
    return "; ".join(parts)


def _candidate_failure_execution(
    target_state: dict[str, Any],
    candidate_target_id: str,
    debug_attach_on_grasp: bool,
    error_msg: str,
) -> dict[str, Any]:
    initial_z = float(target_state.get("position", (0.0, 0.0, 0.0))[2])
    target_type = str(target_state.get("type") or target_state.get("shape") or "")
    target_name = str(target_state.get("name") or target_state.get("object_id") or candidate_target_id)
    return {
        "success": False,
        "target_object": target_name,
        "target_name": target_name,
        "target_type": target_type,
        "debug_attach_on_grasp": bool(debug_attach_on_grasp),
        "attached": False,
        "initial_z": initial_z,
        "final_z": initial_z,
        "target_initial_z": initial_z,
        "target_final_z": initial_z,
        "success_threshold_z": initial_z + 0.08,
        "attach_xy_threshold": "",
        "attach_z_threshold": "",
        "attach_offset": "",
        "trajectory": [],
        "error_msg": error_msg,
    }


def _save_isaac_camera_observation(scene: GraspScene, ep_dir: Path) -> tuple[Path, Path]:
    import numpy as np

    camera = scene.isaac_context.get("camera")
    if camera is None:
        raise RuntimeError("Isaac camera is required for dataset RGB/depth capture")
    camera.update(dt=scene.isaac_context["sim"].get_physics_dt())
    output = camera.data.output
    if "rgb" not in output:
        raise RuntimeError("Isaac camera produced no rgb output")
    depth_key = _find_depth_key(output)
    if depth_key is None:
        raise RuntimeError("Isaac camera produced no depth output")
    rgb_path = ep_dir / "rgb.png"
    depth_path = ep_dir / "depth.npy"
    write_png_array_rgb(rgb_path, output["rgb"][0])
    depth = output[depth_key][0]
    if hasattr(depth, "detach"):
        depth = depth.detach().cpu().numpy()
    np.save(depth_path, depth)
    return rgb_path, depth_path


def _find_depth_key(output: dict[str, Any]) -> str | None:
    for key in ("distance_to_image_plane", "distance_to_camera", "depth"):
        if key in output:
            return key
    return None


def _find_target_state(object_states: list[dict[str, Any]], target_name: str) -> dict[str, Any]:
    for obj in object_states:
        if obj.get("name") == target_name or obj.get("object_id") == target_name:
            return obj
    raise KeyError(f"target object not found: {target_name}")
