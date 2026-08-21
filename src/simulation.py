from __future__ import annotations

from collections.abc import Callable

import numpy as np

from .settings import EncounterSettings
from .types import RawTrajectory, RunIdentity
from .vm_engine import (
    build_obstacle_grid,
    query_obstacle_grid,
    compute_vicsek_angles_open_x_periodic_y,
)


def wrap_angle(angles: np.ndarray) -> np.ndarray:
    """把角度折返到 [−π, π]；界内项零扰动，界外项数学等价（cos/sin 不变）。

    折返保证存储的角度恒为规范值，使按角度数值渲染/配色的下游（如动画
    direction 配色）不会因角度漂出 [−π, π] 而出现颜色冻结或跳变。
    """
    out = np.array(angles, copy=True)
    mask = np.abs(out[0]) > np.pi
    if np.any(mask):
        out[0][mask] = np.mod(out[0][mask] + np.pi, 2.0 * np.pi) - np.pi
    return out


class EncounterSimulation:
    """Cooperation encounter 模型的串行纯求解器。

    只负责初始化状态、reset、step 与 solve：在内存中推进并形成完整
    源轨迹，不计算特征、不写文件、不创建进程池、不管理扫描或 worker。
    """

    def __init__(
        self,
        settings: EncounterSettings | None = None,
        seed: int | np.random.SeedSequence | None = None,
    ):
        self.settings = settings or EncounterSettings()
        self.seed = seed
        if seed is None:
            # 无 seed：每个实例独立以系统熵初始化，不使用模块级 RNG
            self._seed_sequence = np.random.SeedSequence()
            self.entropy = int(self._seed_sequence.entropy)
            self.seed_mode = "entropy"
        elif isinstance(seed, np.random.SeedSequence):
            self._seed_sequence = seed
            self.entropy = int(seed.entropy)
            self.seed_mode = "entropy"
        else:
            self._seed_sequence = np.random.SeedSequence(seed)
            self.entropy = int(seed)
            self.seed_mode = "explicit"
        self.rng = np.random.default_rng(self._seed_sequence)
        self.length = self.settings.length
        self.pos = np.empty((0, 2), dtype=float)
        self.angle = np.empty((1, 0), dtype=float)
        self.vel = np.empty((0, 2), dtype=float)
        self.obstacle_pos = np.empty((0, 2), dtype=float)
        self.obstacle_angle = np.empty((1, 0), dtype=float)
        self._obstacle_grid = None
        self.reset(self.length)

    def reset(self, length: float | None = None) -> None:
        if length is not None:
            self.length = float(length)

        n_particles = self.settings.particles
        self.pos = self.rng.random((n_particles, 2), dtype=float)
        self.pos[:, 0] *= self.settings.place_length
        self.pos[:, 1] *= self.settings.width

        # 初始角度 U(−π/2, π/2)：cosθ 恒正，群体初始向右迁移（有意设定）
        self.angle = (self.rng.random((1, n_particles), dtype=float) - 0.5) * np.pi
        self.vel = np.column_stack((np.cos(self.angle[0]), np.sin(self.angle[0])))

        n_obstacles = self.settings.obstacle_count(self.length)
        if n_obstacles == 0:
            self.obstacle_pos = np.empty((0, 2), dtype=float)
            self.obstacle_angle = np.empty((1, 0), dtype=float)
            self._obstacle_grid = None
            return

        self.obstacle_pos = self.rng.random((n_obstacles, 2), dtype=float)
        self.obstacle_pos[:, 0] = self.settings.begin_length + self.obstacle_pos[:, 0] * self.length
        self.obstacle_pos[:, 1] *= self.settings.width
        raw = self.rng.random((1, n_obstacles), dtype=float) - 0.5
        # 障碍角存原始均匀分布 U(-π, π)；强度 H=que_stren 由 quenched_influence 施加
        self.obstacle_angle = 2.0 * np.pi * raw

        # 障碍物在本次运行内固定，网格索引构建一次缓存复用
        self._obstacle_grid = build_obstacle_grid(
            self.obstacle_pos,
            self.obstacle_angle,
            float(self.settings.width),
            float(self.settings.que_radius),
        )

    def alignment_angle(self) -> np.ndarray:
        return compute_vicsek_angles_open_x_periodic_y(
            self.pos,
            self.angle,
            float(self.settings.width),
            float(self.settings.radius),
        )

    def quenched_influence(self) -> np.ndarray:
        """返回论文 Eq.(1c) 的淬火项 H·arg0[Σexp(iθ_o)]（H=que_stren）。

        引擎查询只给出障碍角圆平均（纯几何量），强度 H 在此处施加，
        与论文修订公式中 H 乘在圆平均之外的定义一致。
        """
        if self._obstacle_grid is None:
            return np.zeros((1, self.pos.shape[0]), dtype=float)
        return self.settings.que_stren * query_obstacle_grid(
            self.pos,
            *self._obstacle_grid,
            float(self.settings.width),
            float(self.settings.que_radius),
        )

    def step(self) -> None:
        annealed = self.settings.strength * self.rng.standard_normal((1, self.pos.shape[0]))
        self.angle = wrap_angle(
            self.alignment_angle() + annealed + self.quenched_influence()
        )
        self.vel = np.column_stack((np.cos(self.angle[0]), np.sin(self.angle[0])))
        self.pos = self.pos + self.settings.speed * self.vel
        self.pos[:, 1] = self.pos[:, 1] % self.settings.width

    def solve(self, on_step: Callable[[int, int], None] | None = None) -> RawTrajectory:
        """串行推进 ``settings.times`` 步，返回完整内存源轨迹（T+1 帧）。

        第 0 帧保存初始化状态，第 k 帧保存第 k 次 step 后的状态；不计算
        特征、不写文件、不创建进程池。``on_step(t, total)`` 每步轻量通知，
        不得持有存储器、写文件或改变数值状态。返回的轨迹帧与求解器状态
        相互独立（副本），障碍为固定位置的副本。
        """
        times = self.settings.times
        n_particles = self.pos.shape[0]
        position = np.empty((times + 1, n_particles, 2), dtype=np.float64)
        angle = np.empty((times + 1, n_particles), dtype=np.float64)
        position[0] = self.pos
        angle[0] = self.angle[0]
        for t in range(times):
            self.step()
            if on_step is not None:
                on_step(t, times)
            position[t + 1] = self.pos
            angle[t + 1] = self.angle[0]

        identity = RunIdentity(
            submission="",
            scan_parameter="",
            scan_value=None,
            scan_index=0,
            repeat_index=0,
            seed_mode=self.seed_mode,
            seed=self.entropy,
        )
        return RawTrajectory(
            step=np.arange(times + 1, dtype=np.int64),
            position=position,
            angle=angle,
            obstacle_position=self.obstacle_pos.copy(),
            obstacle_angle=self.obstacle_angle[0].copy(),
            settings=self.settings,
            identity=identity,
        )
