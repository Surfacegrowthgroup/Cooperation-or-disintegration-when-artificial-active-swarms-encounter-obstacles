"""项目论文图和特征量图共享的 science 绘图样式。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

COL_R = "#0C5DA5"
COL_V = "#0C5DA5"
COL_C = "#E63946"
COL_RATIO = "#2A9D8F"
COL_INSET_LINE = "#D55E00"

MARKER_KW = dict(
    markersize=4.4,
    markeredgewidth=0.55,
    markeredgecolor="white",
    linestyle="none",
    alpha=0.95,
)
RAW_LINE_KW = dict(
    markersize=4.4,
    markeredgewidth=0.55,
    markeredgecolor="white",
    linestyle="-",
    linewidth=1.25,
    alpha=0.95,
)
FIT_KW = dict(linewidth=1.65, alpha=0.98)
PANEL_FIGSIZE = (3.4, 2.8)
WIDE_FIGSIZE = (3.8, 2.8)


def setup_style() -> None:
    """应用 science 风格和项目统一的字体、字号与线宽。"""
    try:
        import scienceplots  # noqa: F401

        plt.style.use(["science"])
    except (ImportError, OSError):
        # 允许仅安装 matplotlib 的环境运行数据契约和绘图测试。
        pass
    plt.rcParams.update(
        {
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.linewidth": 0.75,
            "lines.linewidth": 1.25,
            "savefig.pad_inches": 0.04,
        }
    )


def add_panel_label(ax, label: str, loc=(0.04, 0.06)) -> None:
    """在坐标轴内角添加带白色底衬的面板标签。"""
    ax.text(
        loc[0],
        loc[1],
        label,
        transform=ax.transAxes,
        fontsize=10,
        va="bottom",
        ha="left",
        bbox=dict(
            boxstyle="square,pad=0.16",
            facecolor="white",
            edgecolor="none",
            alpha=0.9,
        ),
    )


def _save(fig, target: str | Path, *, dpi: int = 200) -> Path:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {} if path.suffix.lower() == ".pdf" else {"dpi": dpi}
    fig.savefig(path, bbox_inches="tight", **save_kwargs)
    plt.close(fig)
    return path


def fig3_panels(
    t,
    ratio,
    order_left,
    cluster_left,
    order_right,
    cluster_right,
    out_dir: str | Path,
    *,
    xlim: tuple[float, float] | None = (0.0, 12000.0),
    formats: tuple[str, ...] = ("pdf",),
    dpi: int = 200,
    marker_step: int | None = None,
    output_names: tuple[str, str] = ("Fig3a", "Fig3b"),
) -> list[Path]:
    """绘制 Fig. 3 的两个时间序列面板并返回生成文件。"""
    t = np.asarray(t)
    panels = [
        (
            output_names[0],
            "(a)",
            [
                (r"$R$", ratio, COL_RATIO),
                (r"$V_{1}$", order_left, COL_V),
                (r"$C_{1}$", cluster_left, COL_C),
            ],
            r"$R, V_{1}, C_{1}$",
        ),
        (
            output_names[1],
            "(b)",
            [
                (r"$R$", ratio, COL_RATIO),
                (r"$V_{2}$", order_right, COL_V),
                (r"$C_{2}$", cluster_right, COL_C),
            ],
            r"$R, V_{2}, C_{2}$",
        ),
    ]
    marker_step = marker_step or max(1, len(t) // 36)
    line_styles = ["-", "--", "-."]
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if xlim is None:
        xlim = (float(t[0]), float(t[-1]))

    outputs: list[Path] = []
    for panel, label, series, ylabel in panels:
        fig, ax = plt.subplots(figsize=WIDE_FIGSIZE)
        for idx, (legend, values, color) in enumerate(series):
            ax.plot(
                t,
                values,
                color=color,
                label=legend,
                linewidth=1.05,
                linestyle=line_styles[idx],
                marker=["o", "o", "s"][idx],
                markevery=marker_step,
                markersize=2.8,
                markeredgewidth=0.0,
                alpha=0.9,
            )
        ax.set_xlabel(r"$t$")
        ax.set_ylabel(ylabel)
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.0)
        ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.yaxis.set_major_locator(MaxNLocator(6))
        add_panel_label(ax, label, loc=(0.84, 0.06))
        ax.legend(
            loc="best",
            frameon=True,
            framealpha=0.92,
            facecolor="white",
            edgecolor="none",
            fontsize=8,
        )
        for fmt in formats:
            target = output_dir / f"{panel}.{fmt}"
            target.parent.mkdir(parents=True, exist_ok=True)
            save_kwargs = {} if fmt == "pdf" else {"dpi": dpi}
            fig.savefig(target, bbox_inches="tight", **save_kwargs)
            outputs.append(target)
        plt.close(fig)
    return outputs


def scan_panel(
    var,
    series,
    xlabel: str,
    ylabel: str,
    target: str | Path,
    *,
    panel_label: str | None = None,
    xlim: tuple[float, float] | None = None,
    dpi: int = 200,
) -> Path:
    """绘制一个扫描参数面板。

    ``series`` 为 ``(label, values, color, marker)`` 元组序列。所有曲线
    使用同一套论文颜色、marker、线宽和坐标轴范围，不执行平滑或拟合。
    """
    var = np.asarray(var, dtype=float)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    for label, values, color, marker in series:
        ax.plot(
            var,
            np.asarray(values, dtype=float),
            color=color,
            label=label,
            marker=marker,
            **RAW_LINE_KW,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xlim is None:
        left, right = float(var[0]), float(var[-1])
        if left == right:
            pad = max(abs(left) * 0.05, 0.5)
        else:
            pad = (right - left) * 0.03
        xlim = (left - pad, right + pad)
    ax.set_xlim(*xlim)
    ax.set_ylim(0, 1.0)
    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(6))
    if panel_label:
        add_panel_label(ax, panel_label)
    ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.92,
        facecolor="white",
        edgecolor="none",
        handlelength=1.5,
        borderpad=0.25,
        labelspacing=0.2,
        fontsize=8,
    )
    return _save(fig, target, dpi=dpi)


__all__ = [
    "COL_R",
    "COL_V",
    "COL_C",
    "COL_RATIO",
    "COL_INSET_LINE",
    "MARKER_KW",
    "RAW_LINE_KW",
    "FIT_KW",
    "PANEL_FIGSIZE",
    "WIDE_FIGSIZE",
    "setup_style",
    "add_panel_label",
    "fig3_panels",
    "scan_panel",
]
