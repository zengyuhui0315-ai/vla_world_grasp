"""Franka Panda robot wrapper for the grasp demo."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import pi
from typing import Any

from vla_world_grasp.vla.schemas import Vector3


@dataclass
class RobotCommand:
    name: str
    position: Vector3 | None = None
    gripper_width: float | None = None


@dataclass
class FrankaPandaRobot:
    """Small command facade around Franka Panda.

    The class keeps the command contract independent from Isaac imports. In an
    Isaac runtime these commands are the places to bridge into IK/action APIs;
    in smoke tests they provide a deterministic trace of the intended motion.
    """

    headless: bool = True
    prim_path: str = "/World/Franka"
    ee_position: Vector3 = (0.45, 0.0, 0.45)
    gripper_width: float = 0.08
    commands: list[RobotCommand] = field(default_factory=list)
    isaac_robot: Any | None = None
    sim: Any | None = None
    scene: Any | None = None
    grasp_scene: Any | None = None
    diff_ik_controller: Any | None = None
    robot_entity_cfg: Any | None = None
    ee_jacobi_idx: int | None = None
    ee_quat_w: Any | None = None
    torch: Any | None = None

    def reset(self) -> None:
        self.ee_position = (0.45, 0.0, 0.45)
        self.gripper_width = 0.08
        self.commands.clear()

    def attach_isaac(self, sim: Any, scene: Any, articulation: Any) -> None:
        """Attach IsaacLab articulation and initialize a differential IK controller."""

        import torch
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg

        self.sim = sim
        self.scene = scene
        self.isaac_robot = articulation
        self.torch = torch

        diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=sim.device)
        self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        self.robot_entity_cfg.resolve(scene)
        if articulation.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]
        self.ee_quat_w = articulation.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]][0, 3:7].clone()

    def reset_isaac(self) -> None:
        if self.isaac_robot is None:
            return
        joint_pos = self.isaac_robot.data.default_joint_pos.clone()
        joint_vel = self.isaac_robot.data.default_joint_vel.clone()
        self.isaac_robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.isaac_robot.reset()
        if self.diff_ik_controller is not None:
            self.diff_ik_controller.reset()
        if self.robot_entity_cfg is not None:
            self.ee_quat_w = self.isaac_robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]][0, 3:7].clone()

    def move_to(self, position: Vector3, label: str = "move") -> None:
        self.ee_position = position
        self.commands.append(RobotCommand(name=label, position=position))

    def fixed_top_down_quat_wxyz(self) -> Any:
        """Return a fixed vertical-down hand orientation with yaw locked to 0."""

        if self.torch is None or self.sim is None:
            return (0.0, 1.0, 0.0, 0.0)
        from isaaclab.utils.math import quat_from_euler_xyz

        torch = self.torch
        roll = torch.tensor([pi], device=self.sim.device, dtype=torch.float32)
        pitch = torch.tensor([0.0], device=self.sim.device, dtype=torch.float32)
        yaw = torch.tensor([0.0], device=self.sim.device, dtype=torch.float32)
        return quat_from_euler_xyz(roll, pitch, yaw)[0]

    def current_ee_position(self) -> Vector3:
        if self.isaac_robot is None or self.robot_entity_cfg is None:
            return self.ee_position
        ee_pose = self.isaac_robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]][0]
        return (float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2]))

    def current_ee_pose(self) -> dict[str, list[float]]:
        if self.isaac_robot is None or self.robot_entity_cfg is None:
            quat = self.ee_quat_w
            orientation = [float(v) for v in quat] if quat is not None and not isinstance(quat, tuple) else list(quat or [])
            return {"position": [float(v) for v in self.ee_position], "orientation_wxyz": orientation}
        ee_pose = self.isaac_robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]][0]
        return {
            "position": [float(ee_pose[0]), float(ee_pose[1]), float(ee_pose[2])],
            "orientation_wxyz": [float(v) for v in ee_pose[3:7]],
        }

    def move_to_isaac(
        self,
        position: Vector3,
        steps: int = 80,
        label: str = "move",
        orientation_wxyz: Any | None = None,
        maintain_gripper_width: float | None = None,
    ) -> None:
        self.move_to(position, label=label)
        if self.isaac_robot is None:
            return
        if self.sim is None or self.scene is None or self.diff_ik_controller is None or self.robot_entity_cfg is None:
            raise RuntimeError("Franka Isaac controller is not initialized")
        torch = self.torch
        if torch is None:
            raise RuntimeError("torch is not initialized for Franka controller")

        quat = orientation_wxyz
        if quat is None:
            if self.ee_quat_w is None:
                self.ee_quat_w = self.isaac_robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]][0, 3:7].clone()
            quat = self.ee_quat_w
        command = torch.zeros((1, 7), device=self.sim.device, dtype=torch.float32)
        command[0, 0:3] = torch.tensor(position, device=self.sim.device, dtype=torch.float32)
        command[0, 3:7] = torch.as_tensor(quat, device=self.sim.device, dtype=torch.float32)
        self.diff_ik_controller.reset()
        self.diff_ik_controller.set_command(command)
        joint_pos_des = self.isaac_robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
        for _ in range(steps):
            if maintain_gripper_width is not None:
                self._set_gripper_width_isaac(maintain_gripper_width)
            jacobian = self.isaac_robot.root_physx_view.get_jacobians()[
                :, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]
            ee_pose_w = self.isaac_robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
            root_pose_w = self.isaac_robot.data.root_pose_w
            joint_pos = self.isaac_robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            from isaaclab.utils.math import subtract_frame_transforms

            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            self.isaac_robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
            self._advance_isaac_step()

    def hold_gripper(self, width: float = 0.0, steps: int = 1, label: str = "hold_gripper") -> None:
        self.gripper_width = width
        self.commands.append(RobotCommand(name=label, gripper_width=width))
        if self.isaac_robot is None:
            return
        for _ in range(steps):
            self._set_gripper_width_isaac(width)
            self._advance_isaac_step()

    def open_gripper(self, width: float = 0.08) -> None:
        self.gripper_width = width
        self.commands.append(RobotCommand(name="open_gripper", gripper_width=width))
        self._set_gripper_width_isaac(width)

    def close_gripper(self, width: float = 0.0) -> None:
        self.gripper_width = width
        self.commands.append(RobotCommand(name="close_gripper", gripper_width=width))
        self._set_gripper_width_isaac(width)

    def _set_gripper_width_isaac(self, width: float) -> None:
        if self.isaac_robot is None:
            return
        names = list(self.isaac_robot.data.joint_names)
        finger_ids = [idx for idx, name in enumerate(names) if "finger_joint" in name]
        if not finger_ids:
            return
        torch = self.torch
        target = self.isaac_robot.data.joint_pos.clone()
        per_finger = max(0.0, min(0.04, float(width) / 2.0))
        for joint_id in finger_ids:
            target[:, joint_id] = per_finger
        self.isaac_robot.set_joint_position_target(target[:, finger_ids], joint_ids=finger_ids)
        if self.scene is not None:
            self.scene.write_data_to_sim()

    def _advance_isaac_step(self) -> None:
        if self.sim is None or self.scene is None:
            return
        sim_dt = self.sim.get_physics_dt()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(sim_dt)
        if self.grasp_scene is not None:
            self.grasp_scene.update_attached_object_to_gripper()
            camera = self.grasp_scene.isaac_context.get("camera")
            if camera is not None:
                camera.update(dt=sim_dt)
            recorder = self.grasp_scene.isaac_context.get("recorder")
            if recorder is not None:
                recorder.capture_from_scene(self.grasp_scene)

    def command_log(self) -> list[dict[str, Any]]:
        return [
            {
                "name": command.name,
                "position": command.position,
                "gripper_width": command.gripper_width,
            }
            for command in self.commands
        ]
