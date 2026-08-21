"""形貌演化动画：只读 HDF5 源数据并渲染粒子群穿越障碍区的全过程。

动画不驱动求解器：:meth:`EncounterAnimate.from_source` 从源文件加载
选定帧（初态、``frame_step`` 整除帧与最终帧），渲染所需的场景范围、
障碍与阈值全部由源元数据恢复。Matplotlib 在渲染方法内惰性导入，因此
导入 ``encounter`` 时不会创建图窗或修改全局绘图样式。
"""

from __future__ import annotations

import json
import numbers
from pathlib import Path

import numpy as np

from .postprocess import passing_ratio
from .settings import EncounterSettings
from .storage import EncounterDataStore

_PARTICLE_COLOR = "#0C5DA5"
_OBSTACLE_ZONE_COLOR = "#FBF1E8"
_OBSTACLE_LIGHT_COLOR = "#EFC7AE"
_OBSTACLE_DARK_COLOR = "#B6533C"
_FRAME_COLOR = "#6B7280"
_ANNOTATION_COLOR = "#7C4A3A"
_DIRECTION_CMAP = "twilight_shifted"
_FIGURE_WIDTH = 10.0
_READABLE_MIN_HEIGHT = 2.2

_COLOR_MODES = frozenset({"uniform", "direction"})
_ASPECT_MODES = frozenset({"readable", "equal"})
_OUTPUT_SUFFIXES = frozenset({".gif", ".mp4"})
_FRAME_SUFFIXES = frozenset({".pdf", ".png", ".svg"})

# 快照序号（subplot 标号）绘制常数：位置为模拟区域内左下角，字号固定。
# 仅由 label 参数控制序号文本，字号与位置不外传。
_LABEL_FONT_SIZE = 12
_LABEL_POSITION = (0.02, 0.03)  # ax.transAxes：模拟区域内左下角


def _particle_colors(angles: np.ndarray, color_mode: str):
    """返回统一颜色字符串或按方向编码的颜色数组。"""
    if color_mode == "uniform":
        return _PARTICLE_COLOR
    import matplotlib.cm as cm
    import matplotlib.colors as mcolors

    norm = mcolors.Normalize(-np.pi, np.pi)
    direction_cmap = cm.get_cmap(_DIRECTION_CMAP)
    return direction_cmap(norm(angles))


def _integer_at_least(value, name: str, minimum: int) -> int:
    """返回不小于 minimum 的整数；拒绝布尔值与有损转换。"""
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} 必须是整数")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} 必须不小于 {minimum}")
    return result


class EncounterAnimate:
    """单组参数的形貌动画：只读源数据帧，再构建或导出动画。

    帧通过 :meth:`from_source` 从 HDF5 源文件加载；渲染所需的场景范围、
    障碍与阈值从源元数据恢复，不创建求解器。``render_snapshot`` 提供
    不加载全部帧的单帧快照。
    """

    def __init__(self, settings: EncounterSettings | None = None):
        self.settings = settings or EncounterSettings()
        self.obstacle_position = np.empty((0, 2), dtype=float)
        self.obstacle_angle = np.empty((0,), dtype=float)
        self.frames: list[tuple[np.ndarray, np.ndarray]] = []
        self.frame_times: list[int] = []
        self.frame_step = 1
        self._animation = None
        self._figure = None
        self._status_text = None
        self._label_text = None

    @classmethod
    def from_source(cls, path: str | Path, frame_step: int = 20) -> "EncounterAnimate":
        """从 HDF5 源文件加载初态、frame_step 整除帧与最终帧（未整除补录）。

        只加载选定帧，帧数据与源文件独立（副本）；角度按动画内部约定
        保持 (1, N) 形状。损坏源文件或非法 ``frame_step`` 明确拒绝。
        """
        frame_step = _integer_at_least(frame_step, "frame_step", 1)
        store = EncounterDataStore()
        meta = store.metadata(path)
        settings = EncounterSettings(**json.loads(str(meta["settings_json"])))
        with store._open_readonly(path) as f:
            times = f["trajectory/position"].shape[0] - 1
        frame_times = list(range(0, times + 1, frame_step))
        if frame_times[-1] != times:
            frame_times.append(times)
        pairs = store.read_frames(path, frame_times)
        animator = cls(settings=settings)
        animator.obstacle_position, animator.obstacle_angle = store.read_obstacles(path)
        animator.frames = [(pos, angle.reshape(1, -1)) for pos, angle in pairs]
        animator.frame_times = frame_times
        animator.frame_step = frame_step
        return animator

    @staticmethod
    def render_snapshot(
        source: str | Path,
        output: str | Path,
        step: int,
        *,
        dpi: int = 160,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
        label: str | None = None,
    ) -> None:
        """从源文件按真实 solver step 渲染单帧静态图，不加载其他帧。

        ``step`` 为源轨迹中的真实步数（0..T），由源文件按需读取。
        静态快照固定只显示时间步（不显示通过率），``show_passing_ratio``
        参数保留仅为兼容调用方，不改变快照内容。``label`` 为可选的
        subplot 序号文本（如 ``"(a)"``），字号与位置为模块内部常数。
        """
        step = _integer_at_least(step, "step", 0)
        store = EncounterDataStore()
        meta = store.metadata(source)
        settings = EncounterSettings(**json.loads(str(meta["settings_json"])))
        pos, angle = store.read_frame(source, step)
        animator = EncounterAnimate(settings=settings)
        animator.obstacle_position, animator.obstacle_angle = store.read_obstacles(
            source
        )
        animator.frames = [(pos, angle.reshape(1, -1))]
        animator.frame_times = [step]
        animator.frame_step = 1
        animator.render_frame(
            output,
            0,
            dpi=dpi,
            color_mode=color_mode,
            aspect_mode=aspect_mode,
            show_passing_ratio=show_passing_ratio,
            label=label,
        )

    def _validate_build_options(
        self,
        fps: int,
        color_mode: str,
        aspect_mode: str,
        show_passing_ratio: bool,
    ) -> int:
        if not self.frames:
            raise RuntimeError("尚未加载帧：请先调用 from_source()")
        if len(self.frame_times) != len(self.frames):
            raise RuntimeError("帧数据与时间索引数量不一致，请重新调用 collect()")
        fps = _integer_at_least(fps, "fps", 1)
        if color_mode not in _COLOR_MODES:
            choices = ", ".join(sorted(_COLOR_MODES))
            raise ValueError(f"color_mode 必须是以下值之一：{choices}")
        if aspect_mode not in _ASPECT_MODES:
            choices = ", ".join(sorted(_ASPECT_MODES))
            raise ValueError(f"aspect_mode 必须是以下值之一：{choices}")
        if not isinstance(show_passing_ratio, bool):
            raise ValueError("show_passing_ratio 必须是布尔值")
        return fps

    @staticmethod
    def _style_names() -> list[str]:
        try:
            import scienceplots  # noqa: F401  # 注册 science/no-latex 样式

            return ["science", "no-latex"]
        except ImportError:
            return []

    def _figure_size(self, aspect_mode: str) -> tuple[float, float]:
        s = self.settings
        x_span = (s.begin_length + s.length + s.end_length) - s.left_length
        if x_span <= 0.0 or s.width <= 0.0:
            raise ValueError("模拟区域的宽度和横向跨度必须为正数")
        physical_height = _FIGURE_WIDTH * s.width / x_span
        if aspect_mode == "readable":
            return _FIGURE_WIDTH, max(_READABLE_MIN_HEIGHT, physical_height)
        # 等比例模式为顶部状态栏预留少量空间，数据区域仍严格保持 1:1。
        return _FIGURE_WIDTH, max(1.35, physical_height + 0.50)

    def _draw_static(self, ax):
        """绘制低干扰的障碍区、边界、障碍物和区域标注。"""
        import matplotlib.colors as mcolors
        from matplotlib.patches import Rectangle

        s = self.settings
        x_min = s.left_length
        obstacle_left = s.begin_length
        obstacle_right = s.begin_length + s.length
        x_max = obstacle_right + s.end_length

        ax.add_patch(
            Rectangle(
                (obstacle_left, 0.0),
                s.length,
                s.width,
                facecolor=_OBSTACLE_ZONE_COLOR,
                edgecolor="none",
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (x_min, 0.0),
                x_max - x_min,
                s.width,
                fill=False,
                edgecolor=_FRAME_COLOR,
                linewidth=0.75,
                zorder=3,
            )
        )
        for boundary in (obstacle_left, obstacle_right):
            ax.axvline(
                boundary,
                color=_ANNOTATION_COLOR,
                linewidth=0.75,
                alpha=0.72,
                zorder=3,
            )

        obstacle_artist = None
        n_obstacles = self.obstacle_position.shape[0]
        if n_obstacles:
            obstacle_cmap = mcolors.LinearSegmentedColormap.from_list(
                "encounter_obstacles",
                [_OBSTACLE_LIGHT_COLOR, _OBSTACLE_DARK_COLOR],
            )
            # 障碍角存原始均匀分布 U(-π, π)，强度 H 由求解层施加
            theoretical_amplitude = np.pi
            obstacle_artist = ax.scatter(
                self.obstacle_position[:, 0],
                self.obstacle_position[:, 1],
                c=np.abs(self.obstacle_angle),
                cmap=obstacle_cmap,
                norm=mcolors.Normalize(0.0, theoretical_amplitude, clip=True),
                s=6.5,
                marker="o",
                linewidths=0.0,
                alpha=0.78,
                zorder=2,
            )

        return obstacle_artist

    @staticmethod
    def _arrow_dimensions(n_particles: int) -> tuple[float, float]:
        """返回箭头长度与箭杆宽度（英寸），按粒子密度有界缩放。"""
        if n_particles <= 0:
            density_factor = 1.0
        else:
            density_factor = float(np.clip((250.0 / n_particles) ** 0.18, 0.75, 1.15))
        return 0.13 * density_factor, 0.0105 * density_factor

    def _status(self, frame_index: int, show_passing_ratio: bool) -> str:
        time = self.frame_times[frame_index]
        if not show_passing_ratio:
            return rf"$t = {time}$"
        threshold = self.settings.begin_length + self.settings.length
        ratio = passing_ratio(self.frames[frame_index][0], threshold)
        return rf"$t = {time}$     $R(t) = {ratio:.3f}$"

    def _build_artists(
        self,
        color_mode: str,
        aspect_mode: str,
        show_passing_ratio: bool,
        snapshot_layout: bool = False,
        label: str | None = None,
    ):
        """创建图形与除 FuncAnimation 外的全部静态/动态艺术家。

        供 ``build()`` 与 ``render_frame()`` 共用，保证动画帧与静态快照
        使用同一套场景布局、样式与粒子绘制方式。``snapshot_layout`` 为
        True 时（静态快照路径）时间信息置于模拟区域内左上角，不占用
        区域外空间。``label`` 非空时在模拟区域外左上角绘制 subplot
        序号文本（字号与位置为模块内部常数）。
        """
        import matplotlib.pyplot as plt

        s = self.settings
        x_max = s.begin_length + s.length + s.end_length
        figsize = self._figure_size(aspect_mode)

        fig, ax = plt.subplots(figsize=figsize)
        # 手动控制边距，保证超宽场景中的状态栏和区域标注不会被裁切。
        fig.subplots_adjust(left=0.025, right=0.985, bottom=0.12, top=0.80)
        ax.set_xlim(s.left_length, x_max)
        ax.set_ylim(0.0, s.width)
        ax.set_aspect("auto" if aspect_mode == "readable" else "equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        self._draw_static(ax)

        positions, angles = self.frames[0]
        arrow_length, arrow_width = self._arrow_dimensions(positions.shape[0])
        quiver = ax.quiver(
            positions[:, 0],
            positions[:, 1],
            np.cos(angles[0]),
            np.sin(angles[0]),
            color=_particle_colors(angles[0], color_mode),
            angles="uv",
            scale_units="inches",
            scale=1.0 / arrow_length,
            units="inches",
            width=arrow_width,
            headwidth=3.25,
            headlength=4.25,
            headaxislength=3.8,
            minlength=0.0,
            alpha=0.90,
            zorder=5,
        )

        if snapshot_layout:
            # 静态快照：时间信息置于模拟区域内左上角
            self._status_text = ax.text(
                0.02,
                0.97,
                "",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=10,
                color="#1F2937",
                zorder=6,
            )
        else:
            self._status_text = ax.text(
                0.0,
                1.13,
                "",
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=10,
                color="#1F2937",
                zorder=6,
            )

        self._label_text = None
        if label:
            # 快照序号：模拟区域内左下角，避免与左上角的时间信息重叠
            self._label_text = ax.text(
                *_LABEL_POSITION,
                label,
                transform=ax.transAxes,
                ha="left",
                va="bottom",
                fontsize=_LABEL_FONT_SIZE,
                color="#1F2937",
                zorder=6,
            )

        if color_mode == "direction":
            import matplotlib.cm as cm
            import matplotlib.colors as mcolors

            norm = mcolors.Normalize(-np.pi, np.pi)
            direction_cmap = cm.get_cmap(_DIRECTION_CMAP)
            scalar_mappable = cm.ScalarMappable(norm=norm, cmap=direction_cmap)
            scalar_mappable.set_array(np.empty(0))
            colorbar_ax = ax.inset_axes([0.72, 1.105, 0.26, 0.055])
            colorbar = fig.colorbar(
                scalar_mappable,
                cax=colorbar_ax,
                orientation="horizontal",
            )
            colorbar.set_ticks([-np.pi, 0.0, np.pi])
            colorbar.set_ticklabels([r"$-\pi$", "$0$", r"$\pi$"])
            colorbar.ax.tick_params(labelsize=7, length=2, pad=1)
            colorbar.ax.set_title(r"direction $\theta$", fontsize=7, pad=2)
            colorbar.outline.set_linewidth(0.45)

        return fig, quiver

    def _set_frame_artists(self, quiver, frame_index: int, color_mode: str) -> None:
        """把指定帧的粒子位置、方向与颜色应用到已创建的艺术家。"""
        current_pos, current_angle = self.frames[frame_index]
        quiver.set_offsets(current_pos)
        quiver.set_UVC(np.cos(current_angle[0]), np.sin(current_angle[0]))
        quiver.set_color(_particle_colors(current_angle[0], color_mode))

    def build(
        self,
        fps: int = 5,
        *,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
    ):
        """构建并返回 ``FuncAnimation``，但不显示或写入文件。

        ``uniform`` 使用统一深蓝色；``direction`` 使用周期色图编码方向。
        ``readable`` 适度放大纵向；``equal`` 严格保持物理纵横比例。
        """
        fps = self._validate_build_options(
            fps, color_mode, aspect_mode, show_passing_ratio
        )

        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation

        with plt.style.context(self._style_names()):
            fig, quiver = self._build_artists(
                color_mode, aspect_mode, show_passing_ratio
            )

            def update(frame_index: int):
                self._set_frame_artists(quiver, frame_index, color_mode)
                self._status_text.set_text(
                    self._status(frame_index, show_passing_ratio)
                )
                return quiver, self._status_text

            self._animation = FuncAnimation(
                fig,
                update,
                frames=len(self.frames),
                init_func=lambda: update(0),
                blit=True,
                interval=1000.0 / fps,
            )
            self._figure = fig

        return self._animation

    def save(
        self,
        output: str | Path,
        fps: int = 5,
        dpi: int = 160,
        *,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
    ) -> None:
        """保存 GIF 或 MP4，并在完成或失败后释放该次渲染的图对象。"""
        import shutil

        import matplotlib.pyplot as plt
        from matplotlib.animation import FFMpegWriter, PillowWriter

        dpi = _integer_at_least(dpi, "dpi", 1)
        output = Path(output)
        suffix = output.suffix.lower()
        if suffix not in _OUTPUT_SUFFIXES:
            raise ValueError("输出格式仅支持 .gif 或 .mp4")

        if suffix == ".mp4":
            if shutil.which("ffmpeg") is None:
                raise RuntimeError("系统未安装 ffmpeg，无法保存 MP4；请改用 .gif 输出")
            writer = FFMpegWriter(fps=_integer_at_least(fps, "fps", 1))
        else:
            writer = PillowWriter(fps=_integer_at_least(fps, "fps", 1))

        output.parent.mkdir(parents=True, exist_ok=True)
        animation = self.build(
            fps=fps,
            color_mode=color_mode,
            aspect_mode=aspect_mode,
            show_passing_ratio=show_passing_ratio,
        )
        figure = self._figure
        try:
            animation.save(output, writer=writer, dpi=dpi)
        finally:
            if figure is not None:
                plt.close(figure)
            self._figure = None
            self._animation = None
            self._status_text = None

    def render_frame(
        self,
        output: str | Path,
        frame_index: int | None = None,
        *,
        dpi: int = 160,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
        label: str | None = None,
    ) -> None:
        """把某一帧形貌导出为静态图（PDF/PNG/SVG）。

        ``frame_index`` 缺省为最后一帧，与 ``collect`` 记录的
        ``frame_times`` 一一对应。``label`` 为可选的 subplot 序号文本
        （如 ``"(a)"``），字号与位置为模块内部常数。渲染完成后释放
        图对象，不影响已采集帧。
        """
        self._validate_build_options(
            fps=5, color_mode=color_mode, aspect_mode=aspect_mode,
            show_passing_ratio=show_passing_ratio,
        )
        dpi = _integer_at_least(dpi, "dpi", 1)
        if frame_index is None:
            frame_index = len(self.frames) - 1
        if (
            isinstance(frame_index, bool)
            or not isinstance(frame_index, numbers.Integral)
            or frame_index < 0
            or frame_index >= len(self.frames)
        ):
            raise ValueError(f"frame_index 必须是 0 到 {len(self.frames) - 1} 的整数")
        output = Path(output)
        if output.suffix.lower() not in _FRAME_SUFFIXES:
            raise ValueError("静态快照输出格式仅支持 .pdf/.png/.svg")

        import matplotlib.pyplot as plt

        with plt.style.context(self._style_names()):
            fig, quiver = self._build_artists(
                color_mode, aspect_mode, show_passing_ratio, snapshot_layout=True,
                label=label,
            )
            self._set_frame_artists(quiver, int(frame_index), color_mode)
            # 静态快照固定只显示时间步，不显示通过率
            self._status_text.set_text(self._status(int(frame_index), False))
            output.parent.mkdir(parents=True, exist_ok=True)
            try:
                fig.savefig(output, dpi=dpi, bbox_inches="tight")
            finally:
                plt.close(fig)
                self._figure = None
                self._animation = None
                self._status_text = None
                self._label_text = None

    def save_frames(
        self,
        directory: str | Path,
        *,
        dpi: int = 160,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
    ) -> None:
        """把全部采集帧批量导出为静态 PDF（``step_{t:06d}.pdf``）。

        一次构建艺术家后循环更新并保存，避免逐帧重建图形；完成或失败
        后释放图对象，不影响已采集帧。
        """
        self._validate_build_options(
            fps=5,
            color_mode=color_mode,
            aspect_mode=aspect_mode,
            show_passing_ratio=show_passing_ratio,
        )
        dpi = _integer_at_least(dpi, "dpi", 1)
        output_dir = Path(directory)
        output_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        with plt.style.context(self._style_names()):
            fig, quiver = self._build_artists(
                color_mode, aspect_mode, show_passing_ratio, snapshot_layout=True
            )
            try:
                for index in range(len(self.frames)):
                    self._set_frame_artists(quiver, index, color_mode)
                    # 静态快照固定只显示时间步，不显示通过率
                    self._status_text.set_text(self._status(index, False))
                    step = self.frame_times[index]
                    fig.savefig(
                        output_dir / f"step_{step:06d}.pdf",
                        dpi=dpi,
                        bbox_inches="tight",
                    )
            finally:
                plt.close(fig)
                self._figure = None
                self._animation = None
                self._status_text = None

    def show(
        self,
        fps: int = 5,
        *,
        color_mode: str = "uniform",
        aspect_mode: str = "readable",
        show_passing_ratio: bool = True,
    ) -> None:
        """在图窗中播放动画并在窗口关闭后释放图对象。"""
        import matplotlib.pyplot as plt

        self.build(
            fps=fps,
            color_mode=color_mode,
            aspect_mode=aspect_mode,
            show_passing_ratio=show_passing_ratio,
        )
        figure = self._figure
        try:
            plt.show()
        finally:
            if figure is not None:
                plt.close(figure)
            self._figure = None
            self._animation = None
            self._status_text = None
