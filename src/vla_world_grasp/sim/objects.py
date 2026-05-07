"""Simple tabletop objects used by the Franka grasp demo."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from random import Random
from typing import Any

from vla_world_grasp.vla.schemas import SceneObject, Vector3


ColorRGB = tuple[float, float, float]


@dataclass
class SimObject:
    """State for one graspable object on the table."""

    name: str
    type: str
    color: str
    position: Vector3
    size: Vector3
    rgb: ColorRGB
    mass: float = 0.05
    friction: float = 0.8

    @property
    def object_id(self) -> str:
        return self.name

    @property
    def label(self) -> str:
        return f"{self.color} {self.type}"

    def to_scene_object(self) -> SceneObject:
        return SceneObject(
            object_id=self.name,
            label=self.label,
            color=self.color,
            shape=self.type,
            position=self.position,
            size=self.size,
        )

    def to_state(self) -> dict[str, Any]:
        data = asdict(self)
        data["shape"] = self.type
        data["object_id"] = self.name
        data["label"] = self.label
        return data


DEFAULT_TABLE_POSITION: Vector3 = (0.5, 0.0, 0.0)
DEFAULT_TABLE_SIZE: Vector3 = (0.8, 0.6, 0.04)


def create_default_objects(seed: int | None = None) -> list[SimObject]:
    """Create the three required tabletop objects with stable names and colors."""

    rng = Random(seed)
    jitter = 0.0 if seed is None else 0.01

    def j(value: float) -> float:
        return value + rng.uniform(-jitter, jitter)

    table_top_z = DEFAULT_TABLE_POSITION[2] + DEFAULT_TABLE_SIZE[2] / 2.0
    return [
        SimObject(
            name="red_cube",
            type="cube",
            color="red",
            position=(j(0.45), j(-0.12), table_top_z + 0.025),
            size=(0.05, 0.05, 0.05),
            rgb=(1.0, 0.05, 0.03),
            mass=0.05,
            friction=0.8,
        ),
        SimObject(
            name="blue_cylinder",
            type="cylinder",
            color="blue",
            position=(j(0.52), j(0.03), table_top_z + 0.05),
            size=(0.08, 0.08, 0.10),
            rgb=(0.05, 0.24, 1.0),
            mass=0.03,
            friction=1.2,
        ),
        SimObject(
            name="green_sphere",
            type="sphere",
            color="green",
            position=(j(0.40), j(0.11), table_top_z + 0.03),
            size=(0.06, 0.06, 0.06),
            rgb=(0.05, 0.75, 0.18),
            mass=0.04,
            friction=0.8,
        ),
    ]


SUPPORTED_COLORS: tuple[str, ...] = ("red", "blue", "green", "yellow", "white", "black")
SUPPORTED_SHAPES: tuple[str, ...] = ("cube", "cylinder", "sphere")
COLOR_RGB: dict[str, ColorRGB] = {
    "red": (1.0, 0.05, 0.03),
    "blue": (0.05, 0.24, 1.0),
    "green": (0.05, 0.75, 0.18),
    "yellow": (1.0, 0.82, 0.05),
    "white": (0.92, 0.92, 0.88),
    "black": (0.02, 0.02, 0.02),
}
SIZE_BY_SHAPE: dict[str, Vector3] = {
    "cube": (0.05, 0.05, 0.05),
    "cylinder": (0.08, 0.08, 0.10),
    "sphere": (0.06, 0.06, 0.06),
}
MASS_BY_SHAPE: dict[str, float] = {
    "cube": 0.05,
    "cylinder": 0.03,
    "sphere": 0.04,
}
FRICTION_BY_SHAPE: dict[str, float] = {
    "cube": 0.8,
    "cylinder": 1.2,
    "sphere": 0.8,
}


def create_randomized_objects(
    seed: int | None = None,
    x_range: tuple[float, float] = (0.35, 0.60),
    y_range: tuple[float, float] = (-0.18, 0.18),
    min_object_distance: float = 0.10,
    max_position_attempts: int = 100,
    max_scene_attempts: int = 300,
    log_prefix: str = "[vla_world_grasp]",
) -> list[SimObject]:
    """Create three tabletop objects with randomized color/shape pairings and positions."""

    table_top_z = DEFAULT_TABLE_POSITION[2] + DEFAULT_TABLE_SIZE[2] / 2.0
    distances = [float(min_object_distance)]
    if min_object_distance > 0.08:
        distances.append(0.08)

    for distance in distances:
        print(f"{log_prefix} Sampling object positions with min_distance={distance:.3f}", flush=True)
        for scene_attempt in range(1, max_scene_attempts + 1):
            print(f"{log_prefix} Placement attempt {scene_attempt}/{max_scene_attempts}", flush=True)
            rng = Random(None if seed is None else int(seed) + scene_attempt - 1)
            colors = list(SUPPORTED_COLORS)
            rng.shuffle(colors)
            shapes = list(SUPPORTED_SHAPES)
            positions: list[Vector3] = []
            objects: list[SimObject] = []
            try:
                for index in range(3):
                    color = colors[index]
                    shape = shapes[index]
                    size = SIZE_BY_SHAPE[shape]
                    z = table_top_z + size[2] / 2.0
                    position = _sample_non_overlapping_position(
                        rng,
                        x_range,
                        y_range,
                        z,
                        positions,
                        distance,
                        max_attempts=max_position_attempts,
                    )
                    positions.append(position)
                    objects.append(
                        SimObject(
                            name=f"{color}_{shape}_{index}",
                            type=shape,
                            color=color,
                            position=position,
                            size=size,
                            rgb=COLOR_RGB[color],
                            mass=MASS_BY_SHAPE[shape],
                            friction=FRICTION_BY_SHAPE[shape],
                        )
                    )
            except RuntimeError:
                continue
            print(f"{log_prefix} Placement succeeded.", flush=True)
            return objects

        if distance != distances[-1]:
            print(f"{log_prefix} Placement failed, relaxing min_distance to {distances[-1]:.3f}", flush=True)

    raise RuntimeError(
        f"failed to sample randomized objects after {max_scene_attempts * len(distances)} scene attempts"
    )


def _sample_non_overlapping_position(
    rng: Random,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z: float,
    existing: list[Vector3],
    min_distance: float,
    max_attempts: int = 100,
) -> Vector3:
    for _ in range(max_attempts):
        position = (rng.uniform(*x_range), rng.uniform(*y_range), z)
        if all(_xy_distance(position, other) >= min_distance for other in existing):
            return position
    raise RuntimeError(f"failed to sample non-overlapping object positions with min_distance={min_distance}")


def _xy_distance(a: Vector3, b: Vector3) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def object_height_threshold(obj: SimObject, lift_margin: float = 0.08) -> float:
    return obj.position[2] + lift_margin
