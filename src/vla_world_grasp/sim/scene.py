"""Tabletop Franka scene for Isaac Sim 4.5 / IsaacLab demos."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from vla_world_grasp.sim.objects import (
    DEFAULT_TABLE_POSITION,
    DEFAULT_TABLE_SIZE,
    SimObject,
    create_default_objects,
)
from vla_world_grasp.sim.robot import FrankaPandaRobot


@dataclass
class GraspScene:
    """Minimal tabletop scene with Franka Panda, a camera, and three objects."""

    headless: bool = True
    seed: int | None = None
    sim_backend: str = "isaac"
    robot: FrankaPandaRobot = field(init=False)
    objects: list[SimObject] = field(default_factory=list)
    backend: str = field(default="isaac", init=False)
    isaac_context: dict[str, Any] = field(default_factory=dict, init=False)
    object_entities: dict[str, Any] = field(default_factory=dict, init=False)
    attached_object_name: str | None = field(default=None, init=False)
    attached_object_z_offset: float = field(default=-0.075, init=False)

    def __post_init__(self) -> None:
        if self.sim_backend not in {"isaac", "mock"}:
            raise ValueError(f"unsupported sim_backend: {self.sim_backend}")
        self.backend = self.sim_backend
        self.robot = FrankaPandaRobot(headless=self.headless)

    def reset_scene(self, objects: list[SimObject] | None = None) -> list[dict[str, Any]]:
        self.robot.reset()
        next_objects = objects if objects is not None else create_default_objects(self.seed)
        if self.isaac_context.get("app") is not None:
            geometry_changed = self._object_geometry_signature(self.objects) != self._object_geometry_signature(next_objects)
            self.prepare_for_episode(clear_objects=geometry_changed)
            if geometry_changed:
                self.isaac_context["app"].close()
                self.isaac_context = {}
                self.object_entities = {}
        self.objects = next_objects
        if self.sim_backend == "isaac":
            self._initialize_isaac()
        else:
            self.isaac_context = {"available": False, "reason": "explicit mock backend"}
        return self.get_object_states()

    def prepare_for_episode(self, clear_objects: bool = False) -> None:
        """Clear transient episode state before spawning/resetting objects."""

        self.detach_object_from_gripper()
        if self.isaac_context.get("recorder") is not None:
            self.isaac_context.pop("recorder", None)
        self.isaac_context.pop("candidate_list", None)
        if self.backend != "isaac" or self.isaac_context.get("app") is None:
            return
        self._clear_stage_children("/World/Debug")
        if clear_objects:
            self._clear_stage_children("/World/envs/env_0/Objects")
        try:
            self.step(5)
        except Exception as exc:
            print(f"[vla_world_grasp] WARNING: cleanup settle step failed: {exc}", flush=True)

    def get_object_states(self) -> list[dict[str, Any]]:
        return [obj.to_state() for obj in self.objects]

    def get_object(self, name: str) -> SimObject:
        for obj in self.objects:
            if obj.name == name:
                return obj
        raise KeyError(f"unknown object: {name}")

    def get_object_position(self, name: str) -> tuple[float, float, float]:
        entity = self.object_entities.get(name)
        if entity is not None and self.backend == "isaac":
            try:
                position = entity.data.root_pos_w[0]
                return (float(position[0]), float(position[1]), float(position[2]))
            except Exception:
                pass
        return self.get_object(name).position

    def set_object_position(self, name: str, position: tuple[float, float, float]) -> None:
        obj = self.get_object(name)
        obj.position = position
        entity = self.object_entities.get(name)
        if entity is None or self.backend != "isaac":
            return
        torch = self.isaac_context["torch"]
        pose = torch.tensor([[position[0], position[1], position[2], 1.0, 0.0, 0.0, 0.0]], device=self.device)
        entity.write_root_pose_to_sim(pose)
        entity.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))

    def attach_object_to_gripper(self, name: str, z_offset: float = -0.075) -> None:
        if self.backend != "isaac":
            return
        self.attached_object_name = name
        self.attached_object_z_offset = z_offset
        self.update_attached_object_to_gripper()

    def detach_object_from_gripper(self) -> None:
        self.attached_object_name = None

    def update_attached_object_to_gripper(self) -> None:
        if self.backend != "isaac" or self.attached_object_name is None:
            return
        robot = self.robot.isaac_robot
        entity_cfg = self.robot.robot_entity_cfg
        if robot is None or entity_cfg is None:
            return
        ee_pose = robot.data.body_pose_w[:, entity_cfg.body_ids[0]][0]
        position = (
            float(ee_pose[0]),
            float(ee_pose[1]),
            float(ee_pose[2]) + self.attached_object_z_offset,
        )
        self.set_object_position(self.attached_object_name, position)

    def step(self, count: int = 1) -> None:
        if self.backend != "isaac":
            return
        sim = self.isaac_context["sim"]
        scene = self.isaac_context["scene"]
        camera = self.isaac_context.get("camera")
        recorder = self.isaac_context.get("recorder")
        sim_dt = sim.get_physics_dt()
        for _ in range(count):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim_dt)
            self.update_attached_object_to_gripper()
            if camera is not None:
                camera.update(dt=sim_dt)
            if recorder is not None:
                recorder.capture_from_scene(self)

    def attach_recorder(self, recorder: Any) -> None:
        self.isaac_context["recorder"] = recorder

    def visualize_grasp_candidates(
        self,
        candidates: list[Any],
        selected_candidate: Any | None = None,
        show_only_best: bool = False,
        top_k: int = 5,
        visual_only: bool = True,
    ) -> None:
        """Debug hook retained for compatibility; final demos never create candidate markers."""

        return

    def visualize_target_highlight(self, target_name: str, visual_only: bool = True) -> None:
        """Debug hook retained for compatibility; final demos never create target markers."""

        return

    def clear_candidate_markers(self, settle_steps: int = 3) -> None:
        self.clear_debug_markers(settle_steps=settle_steps)

    def clear_debug_markers(self, settle_steps: int = 3) -> None:
        if self.backend != "isaac":
            return
        self._clear_stage_children("/World/Debug")
        if settle_steps > 0:
            self.step(settle_steps)

    @property
    def device(self) -> Any:
        return self.isaac_context["sim"].device

    @property
    def table_state(self) -> dict[str, Any]:
        return {
            "name": "table",
            "type": "table",
            "position": DEFAULT_TABLE_POSITION,
            "size": DEFAULT_TABLE_SIZE,
        }

    def close(self) -> None:
        app = self.isaac_context.get("app")
        if app is not None:
            self.prepare_for_episode(clear_objects=False)
            app.close()

    def _initialize_isaac(self) -> None:
        """Launch Isaac Sim and create the actual rendered scene."""

        if self.isaac_context.get("app") is not None:
            scene = self.isaac_context.get("scene")
            if scene is not None:
                self.object_entities = {obj.name: scene[f"object_{idx}"] for idx, obj in enumerate(self.objects)}
            self._reset_isaac_entities()
            return

        try:
            from isaaclab.app import AppLauncher  # type: ignore
        except Exception as exc:
            raise RuntimeError("IsaacLab is not importable; cannot run --sim_backend isaac") from exc

        try:
            app_launcher = AppLauncher({"headless": self.headless, "enable_cameras": True, "device": "cuda:0"})
            simulation_app = app_launcher.app
        except Exception as exc:
            raise RuntimeError("failed to launch Isaac Sim; refusing to create mock demo video") from exc

        try:
            import torch
            import isaaclab.sim as sim_utils
            from isaaclab.assets import Articulation, AssetBaseCfg, RigidObjectCfg
            from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
            from isaaclab.sensors.camera import Camera, CameraCfg
            from isaaclab.utils import configclass
            from isaaclab_assets import FRANKA_PANDA_HIGH_PD_CFG
            from pxr import Sdf, UsdGeom
            import omni.usd
        except Exception as exc:
            simulation_app.close()
            raise RuntimeError("failed to import Isaac/IsaacLab scene APIs") from exc

        object_cfgs = {obj.name: self._rigid_object_cfg(obj, RigidObjectCfg, sim_utils) for obj in self.objects}

        @configclass
        class _SceneCfg(InteractiveSceneCfg):
            ground = AssetBaseCfg(
                prim_path="/World/defaultGroundPlane",
                spawn=sim_utils.GroundPlaneCfg(),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.02)),
            )
            light = AssetBaseCfg(
                prim_path="/World/Light",
                spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.78, 0.78, 0.78)),
            )
            robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Franka")
            table = AssetBaseCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=DEFAULT_TABLE_SIZE,
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.55, 0.50, 0.43)),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                ),
                init_state=AssetBaseCfg.InitialStateCfg(pos=DEFAULT_TABLE_POSITION),
            )
            camera = CameraCfg(
                prim_path="{ENV_REGEX_NS}/ObservationCamera",
                update_period=0.0,
                height=480,
                width=640,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=sim_utils.PinholeCameraCfg(
                    focal_length=22.0,
                    focus_distance=2.0,
                    horizontal_aperture=20.955,
                    clipping_range=(0.01, 100.0),
                ),
            )
            object_0 = object_cfgs[self.objects[0].name]
            object_1 = object_cfgs[self.objects[1].name]
            object_2 = object_cfgs[self.objects[2].name]

        sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 60.0, device="cuda:0")
        print("[vla_world_grasp] Creating Isaac SimulationContext...", flush=True)
        sim = sim_utils.SimulationContext(sim_cfg)
        sim.set_camera_view([1.25, -1.15, 0.82], [0.48, 0.0, 0.08])
        self._define_single_env_parent_prims(omni.usd.get_context().get_stage(), UsdGeom, Sdf)
        print("[vla_world_grasp] Spawning Franka/table/objects/camera...", flush=True)
        scene = InteractiveScene(_SceneCfg(num_envs=1, env_spacing=1.0))
        sim.reset()

        camera: Camera = scene["camera"]
        camera.set_world_poses_from_view(
            torch.tensor([[1.25, -1.15, 0.82]], device=sim.device),
            torch.tensor([[0.48, 0.0, 0.08]], device=sim.device),
        )
        camera.update(dt=sim.get_physics_dt())

        robot: Articulation = scene["robot"]
        self.object_entities = {obj.name: scene[f"object_{idx}"] for idx, obj in enumerate(self.objects)}
        self.isaac_context = {
            "available": True,
            "app": simulation_app,
            "sim": sim,
            "scene": scene,
            "camera": camera,
            "torch": torch,
            "sim_utils": sim_utils,
        }
        self.robot.attach_isaac(sim, scene, robot)
        self.robot.grasp_scene = self
        print("[vla_world_grasp] Isaac scene ready.", flush=True)
        self._reset_isaac_entities()

    def _reset_isaac_entities(self) -> None:
        self.robot.reset_isaac()
        for obj in self.objects:
            entity = self.object_entities.get(obj.name)
            if entity is None:
                continue
            torch = self.isaac_context["torch"]
            pose = torch.tensor(
                [[obj.position[0], obj.position[1], obj.position[2], 1.0, 0.0, 0.0, 0.0]],
                device=self.device,
            )
            entity.write_root_pose_to_sim(pose)
            entity.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
            entity.reset()
        self.step(20)

    def _rigid_object_cfg(self, obj: SimObject, rigid_object_cfg_cls: Any, sim_utils: Any) -> Any:
        material = sim_utils.PreviewSurfaceCfg(diffuse_color=obj.rgb)
        physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=obj.friction,
            dynamic_friction=max(0.1, obj.friction * 0.8),
            restitution=0.0,
            friction_combine_mode="max",
        )
        common = {
            "rigid_props": sim_utils.RigidBodyPropertiesCfg(),
            "mass_props": sim_utils.MassPropertiesCfg(mass=obj.mass),
            "collision_props": sim_utils.CollisionPropertiesCfg(),
            "visual_material": material,
            "physics_material": physics_material,
            "semantic_tags": [("class", obj.name)],
        }
        if obj.type == "cube":
            spawn = sim_utils.CuboidCfg(size=obj.size, **common)
        elif obj.type == "cylinder":
            spawn = sim_utils.CylinderCfg(
                radius=max(obj.size[0], obj.size[1]) / 2.0,
                height=obj.size[2],
                axis="Z",
                **common,
            )
        elif obj.type == "sphere":
            spawn = sim_utils.SphereCfg(radius=max(obj.size) / 2.0, **common)
        else:
            raise ValueError(f"unsupported object type: {obj.type}")
        return rigid_object_cfg_cls(
            prim_path=f"/World/envs/env_0/Objects/{obj.name}",
            spawn=spawn,
            init_state=rigid_object_cfg_cls.InitialStateCfg(pos=obj.position),
        )

    def _define_single_env_parent_prims(self, stage: Any, usd_geom: Any, sdf: Any) -> None:
        for prim_path in (
            "/World",
            "/World/envs",
            "/World/envs/env_0",
            "/World/envs/env_0/Objects",
        ):
            if not stage.GetPrimAtPath(prim_path):
                usd_geom.Xform.Define(stage, sdf.Path(prim_path))

    def _define_debug_parent_prims(self, stage: Any, usd_geom: Any, sdf: Any) -> None:
        for prim_path in ("/World", "/World/Debug"):
            if not stage.GetPrimAtPath(prim_path):
                usd_geom.Xform.Define(stage, sdf.Path(prim_path))

    def _set_display_color(self, geom: Any, vt: Any, gf: Any, color: tuple[float, float, float]) -> None:
        from pxr import UsdGeom

        gprim = UsdGeom.Gprim(geom.GetPrim())
        gprim.CreateDisplayColorAttr(vt.Vec3fArray([gf.Vec3f(*color)]))

    def _object_signature(self, objects: list[SimObject]) -> tuple[tuple[str, str, str], ...]:
        return tuple((obj.name, obj.type, obj.color) for obj in objects)

    def _object_geometry_signature(self, objects: list[SimObject]) -> tuple[tuple[str, tuple[float, float, float]], ...]:
        return tuple((obj.type, obj.size) for obj in objects)

    def _clear_stage_children(self, prim_path: str) -> None:
        try:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(prim_path)
            if not prim:
                return
            for child in list(prim.GetChildren()):
                stage.RemovePrim(child.GetPath())
        except Exception as exc:
            print(f"[vla_world_grasp] WARNING: failed to clear {prim_path}: {exc}", flush=True)
