"""Batch collector for grasp success prediction training data."""

from __future__ import annotations

import argparse
import signal
from pathlib import Path
from random import Random
from types import FrameType

from vla_world_grasp.pipeline.collect_episode import collect_episode, episode_to_csv_rows, error_to_csv_row
from vla_world_grasp.sim.scene import GraspScene
from vla_world_grasp.utils.dataset_io import append_dataset_rows, ensure_dataset_dirs, initialize_dataset_csv, write_json


class EpisodeTimeoutError(TimeoutError):
    pass


def collect_dataset(
    num_episodes: int = 100,
    candidates_per_episode: int = 16,
    seed: int = 0,
    headless: bool = True,
    output_dir: str | Path = "data/grasp_success",
    debug_attach_on_grasp: bool = False,
    save_video_per_episode: bool = False,
    episode_timeout_sec: int = 180,
    attach_mode: str = "mixed",
    randomize_color_shape: bool = True,
    randomize_gripper_width: bool = True,
    randomize_candidate_offsets: bool = True,
    min_object_distance: float = 0.10,
    candidate_eval_mode: str = "selected_only",
    candidate_eval_k: int = 4,
    reset_per_candidate: bool = True,
) -> Path:
    if candidate_eval_mode != "selected_only" and save_video_per_episode:
        print(
            "[vla_world_grasp] WARNING: candidate-level evaluation disables per-episode video recording.",
            flush=True,
        )
        save_video_per_episode = False
    dirs = ensure_dataset_dirs(output_dir)
    csv_path = dirs["processed"] / "grasp_success_dataset.csv"
    initialize_dataset_csv(csv_path)
    scene = GraspScene(headless=headless, seed=seed, sim_backend="isaac")
    mode_rng = Random(seed)
    try:
        for idx in range(1, int(num_episodes) + 1):
            scene_seed = int(seed) + idx - 1
            episode_debug_attach = _resolve_debug_attach(attach_mode, debug_attach_on_grasp, mode_rng)
            print(f"[vla_world_grasp] Collecting episode {idx}/{num_episodes}", flush=True)
            print(f"[vla_world_grasp] scene_seed={scene_seed}", flush=True)
            print(f"[vla_world_grasp] attach_mode={attach_mode} debug_attach_on_grasp={episode_debug_attach}", flush=True)
            try:
                with _episode_timeout(episode_timeout_sec):
                    episode = collect_episode(
                        episode_index=idx,
                        output_raw_dir=dirs["raw"],
                        scene_seed=scene_seed,
                        candidates_per_episode=candidates_per_episode,
                        headless=headless,
                        debug_attach_on_grasp=episode_debug_attach,
                        save_video_per_episode=save_video_per_episode,
                        attach_mode=attach_mode,
                        randomize_color_shape=randomize_color_shape,
                        randomize_gripper_width=randomize_gripper_width,
                        randomize_candidate_offsets=randomize_candidate_offsets,
                        min_object_distance=min_object_distance,
                        candidate_eval_mode=candidate_eval_mode,
                        candidate_eval_k=candidate_eval_k,
                        reset_per_candidate=reset_per_candidate,
                        scene=scene,
                        close_scene=False,
                    )
                rows = episode_to_csv_rows(episode)
                print("[vla_world_grasp] Appending rows to CSV...", flush=True)
                append_dataset_rows(csv_path, rows)
                print(f"[vla_world_grasp] CSV updated: {csv_path}", flush=True)
                print("[vla_world_grasp] Moving to next episode...", flush=True)
            except EpisodeTimeoutError as exc:
                print(f"[vla_world_grasp] WARNING: episode {idx:06d} timed out: {exc}", flush=True)
                _save_episode_error(
                    dirs["raw"],
                    csv_path,
                    idx,
                    scene_seed,
                    str(exc),
                    attach_mode,
                    episode_debug_attach,
                )
                print("[vla_world_grasp] Moving to next episode...", flush=True)
            except Exception as exc:
                if _is_isaac_startup_error(exc):
                    print(f"[vla_world_grasp] Isaac startup failed: {exc}", flush=True)
                    raise
                print(f"[vla_world_grasp] WARNING: episode {idx:06d} failed: {exc}", flush=True)
                _save_episode_error(
                    dirs["raw"],
                    csv_path,
                    idx,
                    scene_seed,
                    str(exc),
                    attach_mode,
                    episode_debug_attach,
                )
                print("[vla_world_grasp] Moving to next episode...", flush=True)
    finally:
        print("[vla_world_grasp] Closing Isaac app after dataset collection...", flush=True)
        scene.close()
    print("[vla_world_grasp] Dataset collection complete.", flush=True)
    print(f"[vla_world_grasp] dataset_csv={csv_path}", flush=True)
    return csv_path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Isaac grasp success prediction data.")
    parser.add_argument("--num_episodes", type=int, default=100)
    parser.add_argument("--candidates_per_episode", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--output_dir", default="data/grasp_success")
    parser.add_argument("--debug_attach_on_grasp", action="store_true")
    parser.add_argument("--attach_mode", choices=("attach", "no_attach", "mixed"), default="mixed")
    parser.add_argument("--randomize_color_shape", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_gripper_width", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--randomize_candidate_offsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min_object_distance", type=float, default=0.10)
    parser.add_argument("--candidate_eval_mode", choices=("selected_only", "top_k", "random_k"), default="selected_only")
    parser.add_argument("--candidate_eval_k", type=int, default=4)
    parser.add_argument("--reset_per_candidate", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--save_video_per_episode", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--episode_timeout_sec", type=int, default=180)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    collect_dataset(
        num_episodes=args.num_episodes,
        candidates_per_episode=args.candidates_per_episode,
        seed=args.seed,
        headless=args.headless,
        output_dir=args.output_dir,
        debug_attach_on_grasp=args.debug_attach_on_grasp,
        save_video_per_episode=args.save_video_per_episode,
        episode_timeout_sec=args.episode_timeout_sec,
        attach_mode=args.attach_mode,
        randomize_color_shape=args.randomize_color_shape,
        randomize_gripper_width=args.randomize_gripper_width,
        randomize_candidate_offsets=args.randomize_candidate_offsets,
        min_object_distance=args.min_object_distance,
        candidate_eval_mode=args.candidate_eval_mode,
        candidate_eval_k=args.candidate_eval_k,
        reset_per_candidate=args.reset_per_candidate,
    )


class _episode_timeout:
    def __init__(self, seconds: int) -> None:
        self.seconds = max(1, int(seconds))
        self.previous_handler = None

    def __enter__(self) -> None:
        self.previous_handler = signal.signal(signal.SIGALRM, self._handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        signal.alarm(0)
        if self.previous_handler is not None:
            signal.signal(signal.SIGALRM, self.previous_handler)

    def _handle_timeout(self, signum: int, frame: FrameType | None) -> None:
        raise EpisodeTimeoutError(f"episode exceeded timeout of {self.seconds} seconds")


def _is_isaac_startup_error(exc: Exception) -> bool:
    message = str(exc)
    startup_markers = (
        "IsaacLab is not importable",
        "failed to launch Isaac Sim",
        "failed to import Isaac/IsaacLab scene APIs",
    )
    return any(marker in message for marker in startup_markers)


def _parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _resolve_debug_attach(attach_mode: str, legacy_debug_attach: bool, rng: Random) -> bool:
    if attach_mode == "attach":
        return True
    if attach_mode == "no_attach":
        return False
    if attach_mode == "mixed":
        return rng.random() < 0.5
    return bool(legacy_debug_attach)


def _save_episode_error(
    raw_dir: Path,
    csv_path: Path,
    episode_index: int,
    scene_seed: int,
    error_msg: str,
    attach_mode: str,
    debug_attach_on_grasp: bool,
) -> None:
    ep_dir = raw_dir / f"episode_{episode_index:06d}"
    ep_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        ep_dir / "error_result.json",
        {
            "episode_id": episode_index,
            "scene_seed": scene_seed,
            "error_msg": error_msg,
            "attach_mode": attach_mode,
            "debug_attach_on_grasp": debug_attach_on_grasp,
        },
    )
    print("[vla_world_grasp] Appending rows to CSV...", flush=True)
    append_dataset_rows(
        csv_path,
        [error_to_csv_row(episode_index, scene_seed, error_msg, attach_mode, debug_attach_on_grasp)],
    )
    print(f"[vla_world_grasp] CSV updated: {csv_path}", flush=True)


if __name__ == "__main__":
    main()
