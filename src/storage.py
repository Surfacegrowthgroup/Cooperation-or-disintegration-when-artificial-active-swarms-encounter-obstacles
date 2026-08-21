"""HDF5 v1 源数据存储（任务 04）。

:class:`EncounterDataStore` 只负责在**一次求解全部完成后**把一份
``RawTrajectory`` 写成一份 HDF5 v1 文件，并提供后处理与动画需要的
只读访问（元数据、单帧、帧索引批量、完整轨迹）。写入先落在同卷临时
文件，完整关闭、重开校验后原子提升；目标已存在时拒绝，不覆盖。

命名契约（文件与提交目录名）定义在本模块，供中控与 CLI 使用。
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

from .config import format_scalar, json_dumps, stable_settings_summary
from .settings import EncounterSettings
from .types import RawTrajectory, RunIdentity

HDF5_SCHEMA = "hdf5-v1"
# 存储过滤器：时间分块（每块至多 256 帧）、LZF、shuffle；空数组用连续布局
HDF5_CHUNKED = True
HDF5_COMPRESSION = "lzf"
HDF5_SHUFFLE = True
HDF5_MAX_FRAMES_PER_CHUNK = 256

PRODUCER = "encounter"
PRODUCER_VERSION = "0.1.0"

# 根属性：schema、参数 JSON、任务身份、seed、版本信息与创建时间
HDF5_ATTRIBUTES: tuple[str, ...] = (
    "schema",  # 恒为 HDF5_SCHEMA（名称含版本）
    "settings_json",  # 求解参数 JSON 字符串（不含扫描/调度/后处理参数）
    "submission",  # 提交名（文件名安全化前的原始名）
    "scan_parameter",  # 扫描参数名；无扫描时为 ""
    "scan_value",  # 该文件对应的扫描参数值；无扫描时为 0.0
    "scan_index",  # 扫描序号；无扫描时为 0
    "repeat_index",  # 独立重复序号（0 起）
    "seed_mode",  # "explicit" | "entropy"
    "seed",  # 实际 seed；entropy 模式为该次运行的熵值（int）
    "producer",  # 产生者标识 "encounter"
    "producer_version",  # 产生者版本字符串
    "created_at",  # ISO-8601 UTC 创建时间字符串
)

# 文件名：每个"扫描参数点 × 独立重复"一份文件；扫描参数与值、扫描/重复
# 序号与实际 seed 标识入文件名（无扫描时省略参数段）
HDF5_FILENAME_TEMPLATE = (
    "{submission}_{scan_part}{scan_index:04d}_{repeat_index:04d}_seed{seed}.h5"
)

_REQUIRED_ATTRIBUTES = HDF5_ATTRIBUTES


def sanitize_filename_part(name: object) -> str:
    """把提交名转为文件名安全段：仅保留 [A-Za-z0-9._-]，首尾剥除点/下划线。"""
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(name))
    return cleaned.strip("._") or "submission"


def hdf5_filename(
    submission: str,
    scan_index: int,
    repeat_index: int,
    seed: int,
    *,
    scan_parameter: str = "",
    scan_value: float | None = None,
) -> str:
    """构造源 HDF5 文件名；submission 调用方须先经 sanitize_filename_part。

    无扫描（``scan_parameter=""``）时省略参数段。扫描参数与值、扫描/
    重复序号与实际 seed 标识入文件名，便于事后追溯。
    """
    scan_part = ""
    if scan_parameter:
        scan_part = f"{scan_parameter}{format_scalar(scan_value)}_"
    return HDF5_FILENAME_TEMPLATE.format(
        submission=submission,
        scan_part=scan_part,
        scan_index=int(scan_index),
        repeat_index=int(repeat_index),
        seed=int(seed),
    )


def _same_volume(a: Path, b: Path) -> bool:
    """Windows 上按盘符比较；其他平台视为同一卷。"""
    if os.name == "nt":
        return os.path.splitdrive(os.path.abspath(a))[0] == os.path.splitdrive(
            os.path.abspath(b)
        )[0]
    return True


def _validate_trajectory(trajectory: RawTrajectory) -> None:
    """形状与契约校验：T+1 帧、dtype、step 连续。"""
    step = trajectory.step
    position = trajectory.position
    angle = trajectory.angle
    if step.dtype != np.dtype("int64") or step.ndim != 1:
        raise ValueError(f"step 必须是 int64 一维数组，得到 {step.dtype}/{step.ndim} 维")
    n_frames = step.shape[0]
    if n_frames < 1:
        raise ValueError("轨迹至少包含 t=0 初态（T+1 >= 1）")
    if not np.array_equal(step, np.arange(n_frames)):
        raise ValueError("step 必须是 0..T 的连续序列")
    if position.shape != (n_frames, position.shape[1], 2) or position.dtype != np.dtype(
        "float64"
    ):
        raise ValueError(f"position 必须是 float64[T+1,N,2]，得到 {position.shape}")
    if angle.shape != (n_frames, position.shape[1]) or angle.dtype != np.dtype(
        "float64"
    ):
        raise ValueError(f"angle 必须是 float64[T+1,N]，得到 {angle.shape}")
    if trajectory.obstacle_position.shape[1] != 2:
        raise ValueError("obstacle_position 必须是 [M,2]")
    if trajectory.obstacle_angle.ndim != 1:
        raise ValueError("obstacle_angle 必须是 [M]")


def _chunk_shape(shape: tuple[int, ...], axis: int = 0) -> tuple[int, ...]:
    """沿指定轴（时间）每块至多 256 帧的分块形状；0 尺寸返回 None（连续布局）。"""
    if any(size == 0 for size in shape):
        return None
    chunks = list(shape)
    chunks[axis] = min(HDF5_MAX_FRAMES_PER_CHUNK, shape[axis])
    return tuple(chunks)


@dataclass(frozen=True, slots=True)
class EncounterDataStore:
    """HDF5 v1 源数据存储；``temp_dir`` 必须与最终输出同一卷。"""

    temp_dir: str | Path = "work/tmp/04-hdf5-storage"

    # ------------------------------------------------------------ 命名

    def submission_dir_name(self, name: str | None, settings: EncounterSettings) -> str:
        """提交目录名：经校验的 ``--name``；未提供时用 UTC 时间与参数摘要。"""
        if name is not None:
            return sanitize_filename_part(name)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        summary = stable_settings_summary(settings)
        return f"run_{stamp}_{summary[:40]}"

    # ------------------------------------------------------------ 预检

    def precheck(self, targets: list[str | Path]) -> None:
        """整次提交预检：目标互不重复、任一已存在或跨卷均在求解前拒绝。"""
        seen: set[str] = set()
        resolved: list[Path] = []
        for raw in targets:
            target = Path(raw)
            absolute = os.path.abspath(target)
            if absolute in seen:
                raise ValueError(f"目标路径重复：{target}")
            seen.add(absolute)
            resolved.append(target)
        for target in resolved:
            if target.exists():
                raise FileExistsError(f"目标已存在，拒绝覆盖：{target}")
            if not _same_volume(Path(self.temp_dir), target.parent):
                raise ValueError(
                    f"输出与临时目录不同卷，无法原子提升：{target} vs {self.temp_dir}"
                )

    # ------------------------------------------------------------ 写入

    def write(self, trajectory: RawTrajectory, target: str | Path) -> Path:
        """把一份完整轨迹写成 HDF5 v1 并原子提升；失败不产生最终文件。

        流程：校验形状 -> 预检目标 -> 写入同卷临时文件 -> 关闭重开校验
        -> 再次确认目标不存在 -> os.replace 提升。仅清理本次创建的临时文件。
        """
        _validate_trajectory(trajectory)
        final = Path(target)
        self.precheck([final])
        final.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(self.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        partial = temp_dir / f".{final.name}.{os.getpid()}.{uuid.uuid4().hex}.partial"
        try:
            self._write_unchecked(trajectory, partial)
            self._verify(partial, trajectory)
            if final.exists():
                raise FileExistsError(f"目标已存在，拒绝覆盖：{final}")
            os.replace(partial, final)
        except BaseException:
            partial.unlink(missing_ok=True)
            raise
        return final

    def _write_unchecked(self, trajectory: RawTrajectory, path: Path) -> None:
        """写入数据集与根属性（不做形状预检，供 write 内部使用）。"""
        n_frames, n_particles, _ = trajectory.position.shape
        n_obstacles = trajectory.obstacle_position.shape[0]
        with h5py.File(path, "w") as f:
            position_chunks = _chunk_shape(trajectory.position.shape)
            if position_chunks is not None:
                f.create_dataset(
                    "trajectory/position",
                    data=trajectory.position,
                    chunks=position_chunks,
                    compression=HDF5_COMPRESSION,
                    shuffle=HDF5_SHUFFLE,
                )
            else:
                f.create_dataset("trajectory/position", data=trajectory.position)

            angle_chunks = _chunk_shape(trajectory.angle.shape)
            if angle_chunks is not None:
                f.create_dataset(
                    "trajectory/angle",
                    data=trajectory.angle,
                    chunks=angle_chunks,
                    compression=HDF5_COMPRESSION,
                    shuffle=HDF5_SHUFFLE,
                )
            else:
                f.create_dataset("trajectory/angle", data=trajectory.angle)

            step_chunks = _chunk_shape(trajectory.step.shape)
            if step_chunks is not None:
                f.create_dataset(
                    "trajectory/step",
                    data=trajectory.step,
                    chunks=step_chunks,
                    compression=HDF5_COMPRESSION,
                    shuffle=HDF5_SHUFFLE,
                )
            else:
                f.create_dataset("trajectory/step", data=trajectory.step)

            obstacle_position_chunks = _chunk_shape(
                trajectory.obstacle_position.shape
            )
            if obstacle_position_chunks is not None:
                f.create_dataset(
                    "obstacles/position",
                    data=trajectory.obstacle_position,
                    chunks=obstacle_position_chunks,
                    compression=HDF5_COMPRESSION,
                    shuffle=HDF5_SHUFFLE,
                )
            else:
                f.create_dataset("obstacles/position", data=trajectory.obstacle_position)

            obstacle_angle_chunks = _chunk_shape(trajectory.obstacle_angle.shape)
            if obstacle_angle_chunks is not None:
                f.create_dataset(
                    "obstacles/angle",
                    data=trajectory.obstacle_angle,
                    chunks=obstacle_angle_chunks,
                    compression=HDF5_COMPRESSION,
                    shuffle=HDF5_SHUFFLE,
                )
            else:
                f.create_dataset("obstacles/angle", data=trajectory.obstacle_angle)

            identity = trajectory.identity
            attributes = {
                "schema": HDF5_SCHEMA,
                "settings_json": json_dumps(trajectory.settings),
                "submission": identity.submission,
                "scan_parameter": identity.scan_parameter,
                "scan_value": (
                    0.0 if identity.scan_value is None else float(identity.scan_value)
                ),
                "scan_index": int(identity.scan_index),
                "repeat_index": int(identity.repeat_index),
                "seed_mode": identity.seed_mode,
                "seed": int(identity.seed),
                "producer": PRODUCER,
                "producer_version": PRODUCER_VERSION,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            for key, value in attributes.items():
                f.attrs[key] = value

    def _verify(self, path: Path, trajectory: RawTrajectory) -> None:
        """重开文件校验：数据集逐位一致且元数据完整。"""
        with h5py.File(path, "r") as f:
            for name, expected in (
                ("trajectory/step", trajectory.step),
                ("trajectory/position", trajectory.position),
                ("trajectory/angle", trajectory.angle),
                ("obstacles/position", trajectory.obstacle_position),
                ("obstacles/angle", trajectory.obstacle_angle),
            ):
                if name not in f:
                    raise RuntimeError(f"校验失败：缺少数据集 {name}")
                stored = f[name][()]
                if stored.shape != expected.shape or stored.dtype != expected.dtype:
                    raise RuntimeError(f"校验失败：{name} 形状/类型不一致")
                if not np.array_equal(stored, expected):
                    raise RuntimeError(f"校验失败：{name} 数据不一致")
            for key in _REQUIRED_ATTRIBUTES:
                if key not in f.attrs:
                    raise RuntimeError(f"校验失败：缺少根属性 {key}")

    # ------------------------------------------------------------ 读取

    def _open_readonly(self, path: str | Path) -> h5py.File:
        """打开并校验 schema；调用方负责 close。"""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"源文件不存在：{source}")
        f = h5py.File(source, "r")
        schema = f.attrs.get("schema")
        if schema != HDF5_SCHEMA:
            f.close()
            raise ValueError(f"schema 不兼容：{schema!r}")
        return f

    def metadata(self, path: str | Path) -> dict[str, object]:
        """只读根属性；缺失必填属性时明确拒绝。"""
        with self._open_readonly(path) as f:
            missing = [key for key in _REQUIRED_ATTRIBUTES if key not in f.attrs]
            if missing:
                raise ValueError(f"元数据缺失：{missing}")
            return {key: f.attrs[key] for key in _REQUIRED_ATTRIBUTES}

    def _frame_count(self, f: h5py.File) -> int:
        return f["trajectory/position"].shape[0]

    def _check_frame(self, f: h5py.File, frame_index: int) -> None:
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, int)
            or not 0 <= frame_index < self._frame_count(f)
        ):
            raise ValueError(
                f"帧索引越界：{frame_index!r}（有效 0..{self._frame_count(f) - 1}）"
            )

    def read_frame(
        self, path: str | Path, frame_index: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """按帧读取 (position[N,2], angle[N])。"""
        with self._open_readonly(path) as f:
            self._check_frame(f, frame_index)
            position = f["trajectory/position"][frame_index]
            angle = f["trajectory/angle"][frame_index]
            return position, angle

    def read_frames(
        self, path: str | Path, frame_indices: list[int]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """按帧索引批量读取；索引全部校验后一次读取。"""
        with self._open_readonly(path) as f:
            for frame_index in frame_indices:
                self._check_frame(f, frame_index)
            positions = f["trajectory/position"][frame_indices]
            angles = f["trajectory/angle"][frame_indices]
            return list(zip(positions, angles))

    def read_obstacles(self, path: str | Path) -> tuple[np.ndarray, np.ndarray]:
        """只读固定障碍（position[M,2], angle[M]），供动画等场景使用。"""
        with self._open_readonly(path) as f:
            return f["obstacles/position"][()], f["obstacles/angle"][()]

    def read_trajectory(self, path: str | Path) -> RawTrajectory:
        """完整轨迹读取：数据集 + 属性重建 RawTrajectory。"""
        meta = self.metadata(path)
        with self._open_readonly(path) as f:
            trajectory = RawTrajectory(
                step=f["trajectory/step"][()],
                position=f["trajectory/position"][()],
                angle=f["trajectory/angle"][()],
                obstacle_position=f["obstacles/position"][()],
                obstacle_angle=f["obstacles/angle"][()],
                settings=EncounterSettings(**json.loads(str(meta["settings_json"]))),
                identity=RunIdentity(
                    submission=str(meta["submission"]),
                    scan_parameter=str(meta["scan_parameter"]),
                    scan_value=(
                        None
                        if str(meta["scan_parameter"]) == ""
                        else float(meta["scan_value"])
                    ),
                    scan_index=int(meta["scan_index"]),
                    repeat_index=int(meta["repeat_index"]),
                    seed_mode=str(meta["seed_mode"]),
                    seed=int(meta["seed"]),
                ),
            )
        return trajectory


__all__ = [
    "EncounterDataStore",
    "HDF5_SCHEMA",
    "HDF5_ATTRIBUTES",
    "HDF5_FILENAME_TEMPLATE",
    "sanitize_filename_part",
    "hdf5_filename",
]
