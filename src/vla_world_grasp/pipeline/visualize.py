"""Aggregate demo results and render simple report figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vla_world_grasp.utils.video import INK, MUTED, SUCCESS, WHITE, draw_rect, draw_text, new_canvas, write_png_rgb


def plot_success_rate(
    results_dir: str | Path = "outputs/recordings",
    output_path: str | Path = "outputs/figures/success_rate_bar.png",
) -> dict[str, Any]:
    results = load_results(results_dir)
    total = len(results)
    successes = sum(1 for item in results if item.get("success"))
    rate = successes / total if total else 0.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = new_canvas(760, 480, WHITE)
    _draw_success_bar(canvas, successes=successes, total=total, rate=rate)
    write_png_rgb(output, canvas)

    summary = {
        "results_dir": str(results_dir),
        "figure_path": str(output),
        "total": total,
        "successes": successes,
        "success_rate": rate,
        "episodes": [
            {
                "instruction": item.get("instruction"),
                "target": item.get("execution", {}).get("target_object"),
                "success": item.get("success"),
                "backend": item.get("vla_result", {}).get("backend"),
            }
            for item in results
        ],
    }
    summary_path = output.parent / "success_rate_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def plot_candidate_scores(
    candidates: list[dict[str, Any]],
    selected_candidate_id: int,
    output_path: str | Path,
) -> Path:
    """Render a compact candidate-score bar chart."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas = new_canvas(980, 520, WHITE)
    draw_text(canvas, 38, 34, "CANDIDATE BASELINE SCORES", INK, scale=4)
    draw_text(canvas, 42, 116, "SCORING FIGURE ONLY - NO ISAAC SCENE MARKERS", MUTED, scale=2)
    draw_text(canvas, 42, 88, f"BEST ID {selected_candidate_id}", SUCCESS, scale=3)

    chart_left, chart_top = 70, 165
    chart_width, chart_height = 860, 280
    draw_rect(canvas, chart_left, chart_top, chart_left + chart_width, chart_top + chart_height, (232, 234, 236))
    max_score = max((float(item.get("score", 0.0)) for item in candidates), default=1.0)
    max_score = max(max_score, 1.0)
    bar_gap = 6
    bar_count = max(1, len(candidates))
    bar_width = max(10, int((chart_width - bar_gap * (bar_count + 1)) / bar_count))

    for idx, item in enumerate(candidates):
        candidate_id = int(item["candidate_id"])
        score = float(item.get("score", 0.0))
        left = chart_left + bar_gap + idx * (bar_width + bar_gap)
        right = left + bar_width
        bar_height = int(chart_height * score / max_score)
        color = SUCCESS if candidate_id == selected_candidate_id else (64, 132, 210)
        draw_rect(canvas, left, chart_top + chart_height - bar_height, right, chart_top + chart_height, color)
        draw_text(canvas, left, chart_top + chart_height + 18, str(candidate_id), INK, scale=1)

    selected = next((item for item in candidates if int(item["candidate_id"]) == selected_candidate_id), None)
    if selected is not None:
        terms = selected.get("score_terms", {})
        draw_text(
            canvas,
            42,
            462,
            "BEST TERMS C {:.2f} R {:.2f} COL {:.2f} H {:.2f} W {:.2f} Y {:.2f}".format(
                float(terms.get("center_score", 0.0)),
                float(terms.get("reachability_score", 0.0)),
                float(terms.get("collision_score", 0.0)),
                float(terms.get("height_score", 0.0)),
                float(terms.get("gripper_width_score", 0.0)),
                float(terms.get("yaw_score", 0.0)),
            ),
            INK,
            scale=2,
        )
    draw_rect(canvas, chart_left - 3, chart_top, chart_left, chart_top + chart_height + 3, INK)
    draw_rect(canvas, chart_left - 3, chart_top + chart_height, chart_left + chart_width + 6, chart_top + chart_height + 3, INK)
    draw_text(canvas, 760, 492, "GREEN BAR IS BEST", SUCCESS, scale=2)
    write_png_rgb(output, canvas)
    return output


def load_results(results_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(results_dir)
    if not root.exists():
        return []
    results = []
    for path in sorted(root.rglob("result.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot success rate from recorded demo results.")
    parser.add_argument("--results_dir", default="outputs/recordings")
    parser.add_argument("--output_path", default="outputs/figures/success_rate_bar.png")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    summary = plot_success_rate(args.results_dir, args.output_path)
    print(f"figure_path={summary['figure_path']}")
    print(f"success_rate={summary['success_rate']:.3f}")
    print(f"episodes={summary['total']}")


def _draw_success_bar(canvas: list[list[tuple[int, int, int]]], successes: int, total: int, rate: float) -> None:
    width, height = len(canvas[0]), len(canvas)
    draw_text(canvas, 44, 36, "SUCCESS RATE", INK, scale=4)
    draw_text(canvas, 48, 92, "RECORDED FRANKA TOP DOWN GRASP DEMOS", MUTED, scale=2)

    chart_left, chart_top = 110, 150
    chart_width, chart_height = 190, 230
    draw_rect(canvas, chart_left, chart_top, chart_left + chart_width, chart_top + chart_height, (228, 230, 232))
    bar_height = int(chart_height * rate)
    draw_rect(
        canvas,
        chart_left,
        chart_top + chart_height - bar_height,
        chart_left + chart_width,
        chart_top + chart_height,
        SUCCESS,
    )
    draw_rect(canvas, chart_left - 3, chart_top, chart_left, chart_top + chart_height + 3, INK)
    draw_rect(canvas, chart_left - 3, chart_top + chart_height, chart_left + chart_width + 8, chart_top + chart_height + 3, INK)
    draw_text(canvas, chart_left + 36, chart_top + chart_height + 24, "SUCCESS", INK, scale=2)

    pct = int(round(rate * 100))
    draw_text(canvas, 390, 176, f"{pct} PERCENT", SUCCESS if rate > 0 else INK, scale=4)
    draw_text(canvas, 392, 238, f"{successes} OF {total} EPISODES", INK, scale=3)
    draw_text(canvas, 392, 292, "SOURCE OUTPUTS RECORDINGS", MUTED, scale=2)
    if total == 0:
        draw_text(canvas, 392, 330, "NO RESULT JSON FILES FOUND", (170, 55, 55), scale=2)


if __name__ == "__main__":
    main()
