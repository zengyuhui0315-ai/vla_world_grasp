"""Top-down grasp controller for the Franka demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vla_world_grasp.sim.scene import GraspScene
from vla_world_grasp.vla.schemas import GraspCandidate, Vector3


@dataclass(frozen=True)
class GraspExecutionResult:
    success: bool
    target_object: str
    target_type: str
    debug_attach_on_grasp: bool
    attached: bool
    initial_z: float
    final_z: float
    success_threshold_z: float
    attach_xy_threshold: float
    attach_z_threshold: float
    attach_offset: Vector3
    trajectory: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "target_object": self.target_object,
            "target_name": self.target_object,
            "target_type": self.target_type,
            "debug_attach_on_grasp": self.debug_attach_on_grasp,
            "attached": self.attached,
            "initial_z": self.initial_z,
            "final_z": self.final_z,
            "target_initial_z": self.initial_z,
            "target_final_z": self.final_z,
            "success_threshold_z": self.success_threshold_z,
            "attach_xy_threshold": self.attach_xy_threshold,
            "attach_z_threshold": self.attach_z_threshold,
            "attach_offset": self.attach_offset,
            "trajectory": self.trajectory,
        }


@dataclass(frozen=True)
class ObjectGraspParams:
    grasp_z_offset: float
    gripper_open_width: float
    gripper_close_width: float
    attach_threshold: float
    attach_xy_threshold: float
    attach_z_threshold: float
    attach_offset: Vector3 = (0.0, 0.0, -0.065)
    approach_offset: float = 0.28
    pre_grasp_offset: float = 0.14
    lift_offset: float = 0.32
    approach_steps: int = 120
    pre_grasp_steps: int = 80
    descend_steps: int = 80
    close_gripper_steps: int = 80
    lift_steps: int = 140
    final_hold_steps: int = 80


OBJECT_GRASP_PARAMS: dict[str, ObjectGraspParams] = {
    "cube": ObjectGraspParams(
        grasp_z_offset=0.045,
        gripper_open_width=0.035,
        gripper_close_width=0.0,
        attach_threshold=0.08,
        attach_xy_threshold=0.08,
        attach_z_threshold=0.12,
        attach_offset=(0.0, 0.0, -0.060),
    ),
    "cylinder": ObjectGraspParams(
        grasp_z_offset=0.075,
        gripper_open_width=0.080,
        gripper_close_width=0.0,
        attach_threshold=0.10,
        attach_xy_threshold=0.070,
        attach_z_threshold=0.120,
        attach_offset=(0.0, 0.0, -0.075),
        approach_offset=0.32,
        pre_grasp_offset=0.18,
        lift_offset=0.36,
        descend_steps=90,
        close_gripper_steps=100,
        lift_steps=160,
        final_hold_steps=120,
    ),
    "sphere": ObjectGraspParams(
        grasp_z_offset=0.060,
        gripper_open_width=0.030,
        gripper_close_width=0.0,
        attach_threshold=0.10,
        attach_xy_threshold=0.08,
        attach_z_threshold=0.12,
        attach_offset=(0.0, 0.0, -0.065),
        close_gripper_steps=80,
        lift_steps=140,
    ),
}


def execute_grasp(
    scene: GraspScene,
    candidate: GraspCandidate,
    lift_height: float = 0.16,
    debug_attach_on_grasp: bool = False,
) -> GraspExecutionResult:
    """Execute a fixed-orientation top-down grasp and final hold."""

    target_name = candidate.source.target_id
    target = scene.get_object(target_name)
    target_initial_z = target.position[2]
    params = _grasp_params(target.type)
    threshold_z = target_initial_z + 0.08
    target_x, target_y = _target_xy_for_grasp(target.type, target.position, candidate.position)

    approach_pose: Vector3 = (target_x, target_y, target_initial_z + params.approach_offset)
    pre_grasp_pose: Vector3 = (target_x, target_y, target_initial_z + params.pre_grasp_offset)
    grasp_pose: Vector3 = (target_x, target_y, target_initial_z + params.grasp_z_offset)
    lift_pose: Vector3 = (target_x, target_y, target_initial_z + params.lift_offset)

    top_down_quat = scene.robot.fixed_top_down_quat_wxyz()
    attached = False

    print(f"[vla_world_grasp] target_name={target_name}", flush=True)
    print(f"[vla_world_grasp] target_type={target.type}", flush=True)
    print(f"[vla_world_grasp] target_position={list(target.position)}", flush=True)
    print(f"[vla_world_grasp] selected_candidate.position={list(candidate.position)}", flush=True)
    print(f"[vla_world_grasp] current_ee_pose={scene.robot.current_ee_pose()}", flush=True)
    print(f"[vla_world_grasp] using {target.type} grasp params", flush=True)
    print(f"[vla_world_grasp] object_specific_grasp_z_offset={params.grasp_z_offset:.4f}", flush=True)
    print(f"[vla_world_grasp] gripper_open_width={params.gripper_open_width:.4f}", flush=True)
    print(f"[vla_world_grasp] gripper_close_width={params.gripper_close_width:.4f}", flush=True)
    print(f"[vla_world_grasp] gripper_width={params.gripper_open_width:.4f}", flush=True)
    print(f"[vla_world_grasp] attach_threshold={params.attach_threshold:.4f}", flush=True)
    print(f"[vla_world_grasp] attach_xy_threshold={params.attach_xy_threshold:.4f}", flush=True)
    print(f"[vla_world_grasp] attach_z_threshold={params.attach_z_threshold:.4f}", flush=True)
    print(f"[vla_world_grasp] ee_to_target_distance={_distance(scene.robot.current_ee_position(), target.position):.4f}", flush=True)
    print(f"[vla_world_grasp] debug_attach={debug_attach_on_grasp}", flush=True)
    print(f"[vla_world_grasp] target_initial_z={target_initial_z:.4f}", flush=True)

    maintain_open_width = params.gripper_open_width if target.type == "cylinder" else None
    scene.robot.open_gripper(params.gripper_open_width)
    if scene.backend == "isaac":
        print("[vla_world_grasp] Grasp phase: approach", flush=True)
        scene.robot.move_to_isaac(
            approach_pose,
            steps=params.approach_steps,
            label="approach",
            orientation_wxyz=top_down_quat,
            maintain_gripper_width=maintain_open_width,
        )
        print("[vla_world_grasp] Grasp phase: pre_grasp", flush=True)
        scene.robot.move_to_isaac(
            pre_grasp_pose,
            steps=params.pre_grasp_steps,
            label="pre_grasp",
            orientation_wxyz=top_down_quat,
            maintain_gripper_width=maintain_open_width,
        )
    else:
        scene.robot.move_to(approach_pose, label="approach")
        scene.step(params.approach_steps)
        scene.robot.move_to(pre_grasp_pose, label="pre_grasp")
        scene.step(params.pre_grasp_steps)

    if scene.backend == "isaac":
        print("[vla_world_grasp] Grasp phase: descend", flush=True)
        scene.robot.move_to_isaac(
            grasp_pose,
            steps=params.descend_steps,
            label="descend",
            orientation_wxyz=top_down_quat,
            maintain_gripper_width=maintain_open_width,
        )
    else:
        scene.robot.move_to(grasp_pose, label="descend")
        scene.step(params.descend_steps)

    scene.robot.close_gripper(params.gripper_close_width)
    if scene.backend == "isaac":
        print("[vla_world_grasp] Grasp phase: close_gripper", flush=True)
        scene.robot.hold_gripper(params.gripper_close_width, steps=params.close_gripper_steps, label="close_gripper_hold")
        ee_position = scene.robot.current_ee_position()
        ee_to_target_distance = _distance(ee_position, target.position)
        attach_xy_distance = _xy_distance(ee_position, target.position)
        attach_z_distance = abs(ee_position[2] - target.position[2])
        print(f"[vla_world_grasp] ee_to_target_distance={ee_to_target_distance:.4f}", flush=True)
        print(f"[vla_world_grasp] attach_xy_distance={attach_xy_distance:.4f}", flush=True)
        print(f"[vla_world_grasp] attach_z_distance={attach_z_distance:.4f}", flush=True)
        should_attach = (
            attach_xy_distance < params.attach_xy_threshold
            and attach_z_distance < params.attach_z_threshold
        )
        if debug_attach_on_grasp and should_attach:
            scene.attach_object_to_gripper(target_name, z_offset=params.attach_offset[2])
            attached = True
            print("[vla_world_grasp] debug_attach=True", flush=True)
            print(f"[vla_world_grasp] attached_target={target_name}", flush=True)
            print("[vla_world_grasp] attached=True", flush=True)
    else:
        scene.robot.hold_gripper(params.gripper_close_width, steps=params.close_gripper_steps, label="close_gripper_hold")
        scene.step(params.close_gripper_steps)

    if attached:
        print("[vla_world_grasp] maintaining attached target during lift", flush=True)
    if scene.backend == "isaac":
        print("[vla_world_grasp] Grasp phase: lift", flush=True)
        scene.robot.move_to_isaac(
            lift_pose,
            steps=params.lift_steps,
            label="lift",
            orientation_wxyz=top_down_quat,
            maintain_gripper_width=params.gripper_close_width,
        )
        if attached:
            scene.update_attached_object_to_gripper()
    else:
        scene.robot.move_to(lift_pose, label="lift")
        scene.set_object_position(target_name, (target_x, target_y, lift_pose[2] + params.attach_offset[2]))
        scene.step(params.lift_steps)

    if attached:
        print("[vla_world_grasp] maintaining attached target during final_hold", flush=True)
    if scene.backend == "isaac":
        print("[vla_world_grasp] Grasp phase: final_hold", flush=True)
        scene.robot.hold_gripper(params.gripper_close_width, steps=params.final_hold_steps, label="final_hold")
        if attached:
            scene.update_attached_object_to_gripper()
    else:
        scene.robot.hold_gripper(params.gripper_close_width, steps=params.final_hold_steps, label="final_hold")
        scene.step(params.final_hold_steps)

    final_z = scene.get_object(target_name).position[2]
    success = final_z > threshold_z
    print(f"[vla_world_grasp] target_final_z={final_z:.4f}", flush=True)
    print(f"[vla_world_grasp] success={success}", flush=True)
    return GraspExecutionResult(
        success=success,
        target_object=target_name,
        target_type=target.type,
        debug_attach_on_grasp=debug_attach_on_grasp,
        attached=attached,
        initial_z=target_initial_z,
        final_z=final_z,
        success_threshold_z=threshold_z,
        attach_xy_threshold=params.attach_xy_threshold,
        attach_z_threshold=params.attach_z_threshold,
        attach_offset=params.attach_offset,
        trajectory=scene.robot.command_log(),
    )


def _grasp_params(object_type: str) -> ObjectGraspParams:
    return OBJECT_GRASP_PARAMS.get(object_type, OBJECT_GRASP_PARAMS["cube"])


def _target_xy_for_grasp(object_type: str, target_position: Vector3, candidate_position: Vector3) -> tuple[float, float]:
    if object_type == "cylinder":
        return (candidate_position[0], candidate_position[1])
    return (target_position[0], target_position[1])


def _distance(a: Vector3, b: Vector3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2) ** 0.5


def _xy_distance(a: Vector3, b: Vector3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5
