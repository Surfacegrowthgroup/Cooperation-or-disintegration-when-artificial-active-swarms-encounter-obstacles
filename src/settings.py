from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EncounterSettings:
    """群团-障碍（Cooperation encounter）模型的数值参数。

    只包含求解器读取的物理与时间字段；扫描定义（ScanSpec）、运行
    调度（RunConfig）与后处理（PostprocessConfig）参数分别由
    ``encounter.config`` 管理，不在此声明。

    长度单位无量纲；``speed`` 是每时间步的位移（模型为离散时间映射，
    无显式 dt）。初始粒子角度均匀分布在 (−π/2, π/2)（群体向右迁移的
    有意设定，见 ``simulation.reset``）。
    """

    par_density: float = 0.1
    que_density: float = 0.01

    times: int = 6000

    width: float = 50.0
    length: float = 400.0

    left_length: float = -100.0
    place_length: float = 50.0
    white_length: float = 10.0
    end_length: float = 100.0

    radius: float = 1.0
    strength: float = 0.01
    speed: float = 0.3

    que_stren: float = 1.0
    que_radius: float = 0.5

    @property
    def begin_length(self) -> float:
        return self.place_length + self.white_length

    @property
    def particles(self) -> int:
        return int(self.width * self.place_length * self.par_density)

    def obstacle_count(self, length: float | None = None) -> int:
        current_length = self.length if length is None else length
        return int(self.width * current_length * self.que_density)
