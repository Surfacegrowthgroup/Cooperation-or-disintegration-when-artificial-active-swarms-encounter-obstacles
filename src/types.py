"""encounter 的公共数据类型：内存源轨迹与运行身份。

``RawTrajectory`` 保存一次求解的全部内存源状态（任务 03 的
``solve()`` 返回值），字段命名与 HDF5 v1 契约（tests/contracts.py）
对齐；``RunIdentity`` 记录该次运行的提交名、扫描位置与随机身份，
随轨迹写入 HDF5 元数据。模块不导入求解器、存储、后处理、动画或
multiprocessing。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .settings import EncounterSettings


@dataclass(frozen=True, slots=True)
class RunIdentity:
    """一次原子运行的身份：提交名、扫描位置、重复序号与实际随机身份。"""

    submission: str
    scan_parameter: str  # 无扫描时为 ""
    scan_value: float | None  # 无扫描时为 None
    scan_index: int  # 无扫描时为 0
    repeat_index: int
    seed_mode: Literal["explicit", "entropy"]  # 显式 seed 或系统熵
    seed: int  # 显式 seed 值或该次运行的实际熵值

    def is_scan(self) -> bool:
        return self.scan_parameter != ""


@dataclass(slots=True)
class RawTrajectory:
    """一次求解的全部内存源轨迹（t=0 初态 + T 次推进后状态，共 T+1 帧）。

    ``step``/``position``/``angle`` 的形状与 dtype 与 HDF5 v1 契约
    一致：step int64[T+1]，position float64[T+1, N, 2]，angle
    float64[T+1, N]；障碍 position float64[M, 2]、angle float64[M]
    为原始障碍角（不含强度 H）。不保存速度与任何后处理特征。
    """

    step: np.ndarray  # int64 [T+1]
    position: np.ndarray  # float64 [T+1, N, 2]
    angle: np.ndarray  # float64 [T+1, N]
    obstacle_position: np.ndarray  # float64 [M, 2]
    obstacle_angle: np.ndarray  # float64 [M]
    settings: EncounterSettings
    identity: RunIdentity


__all__ = ["RawTrajectory", "RunIdentity"]
