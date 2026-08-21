"""cooperation 命令行入口（任务 08）。

统一编排运行提交、后处理、动画与快照；命令行只调用现有模块，不复制
求解、存储、后处理或渲染逻辑。退出码：成功 0、运行或数据失败 1、
参数错误 2（argparse 默认）。不提供 checkpoint、overwrite、后台队列、
自动续跑或外部调度。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .animate import EncounterAnimate
from .config import PostprocessConfig, RunConfig, ScanSpec, coerce_set_value, parse_set_expression
from .controller import EncounterController
from .feature_plot import EncounterFeaturePlotter
from .postprocess import EncounterPostprocessor
from .settings import EncounterSettings
from .storage import EncounterDataStore

_SUBCOMMANDS = ("run", "postprocess", "plot", "animate", "snapshot")
_PROG = "cooperation"


class _UsageError(ValueError):
    """参数错误：CLI 层捕获后以退出码 2 结束（区别于运行/数据失败 1）。"""


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description="Cooperation encounter 数值模拟与后处理工具",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="子命令")

    run = subparsers.add_parser("run", help="运行一次提交（默认子命令）")
    run.add_argument(
        "--set", action="append", default=[], metavar="key=value",
        help="覆盖求解参数，可重复（例如 --set times=6000 --set length=400）",
    )
    run.add_argument(
        "--scan", default=None, metavar="parameter:start:stop:step",
        help="单参数扫描（例如 --scan length:0:1200:50）",
    )
    run.add_argument("--loop-times", type=int, default=1, help="独立重复数（默认 1）")
    run.add_argument("--workers", type=int, default=1, help="并行 worker 数（默认 1）")
    run.add_argument("--seed", type=int, default=None, help="随机种子（默认 None=随机）")
    run.add_argument("--name", default=None, help="提交名（目录名；缺省用时间与参数摘要）")
    run.add_argument(
        "--output-dir", default=None, help="源数据根目录（默认 data/raw）"
    )
    run.add_argument("--no-progress", action="store_false", dest="progress", help="关闭进度显示")

    post = subparsers.add_parser("postprocess", help="从源 HDF5 生成特征文件")
    post.add_argument("input", metavar="INPUT", help="单份源 HDF5 或提交目录")
    post.add_argument("--start-average", type=int, default=3000, help="特征时间平均起点（默认 3000）")
    post.add_argument(
        "--output-dir", default=None, help="特征输出根目录（默认 data/processed）"
    )

    plot = subparsers.add_parser("plot", help="从特征量 NPZ 生成单页 PDF 图")
    plot.add_argument("input", metavar="INPUT", help="features.npz 或 summary.npz")
    plot.add_argument(
        "--output-dir",
        default=None,
        help="绘图输出根目录（默认 tmp/feature-plots）",
    )

    animate = subparsers.add_parser("animate", help="从源 HDF5 生成 GIF/MP4 动画")
    animate.add_argument("source", metavar="SOURCE", help="源 HDF5 文件")
    animate.add_argument("--output", required=True, help="输出文件（.gif 或 .mp4）")
    animate.add_argument("--frame-step", type=int, default=20, help="帧间隔步数（默认 20）")
    animate.add_argument("--fps", type=int, default=5, help="帧率（默认 5）")
    animate.add_argument("--dpi", type=int, default=160, help="分辨率（默认 160）")
    animate.add_argument(
        "--color-mode", choices=("uniform", "direction"), default="uniform",
        help="粒子颜色（默认 uniform）",
    )
    animate.add_argument(
        "--aspect-mode", choices=("readable", "equal"), default="readable",
        help="画面比例（默认 readable）",
    )

    snapshot = subparsers.add_parser("snapshot", help="从源 HDF5 渲染单帧静态图")
    snapshot.add_argument("source", metavar="SOURCE", help="源 HDF5 文件")
    snapshot.add_argument("--output", required=True, help="输出文件（.pdf/.png/.svg）")
    snapshot.add_argument("--step", type=int, default=None, help="真实步数（默认最后一帧）")
    snapshot.add_argument("--dpi", type=int, default=160, help="分辨率（默认 160）")
    snapshot.add_argument(
        "--color-mode", choices=("uniform", "direction"), default="uniform",
        help="粒子颜色（默认 uniform）",
    )
    snapshot.add_argument(
        "--aspect-mode", choices=("readable", "equal"), default="readable",
        help="画面比例（默认 readable）",
    )
    snapshot.add_argument(
        "--label", default=None, help="subplot 序号文本（如 (a)）；缺省不显示",
    )
    return parser


def _route_default_run(argv: list[str]) -> list[str]:
    """无子命令且首个参数为选项时自动路由到 run。"""
    if argv and argv[0] not in _SUBCOMMANDS and argv[0].startswith("-"):
        return ["run", *argv]
    return argv


def _apply_sets(settings: EncounterSettings, sets: list[str]) -> set[str]:
    """按字段类型解析 --set；未知字段、重复字段、派生属性均拒绝。"""
    applied: set[str] = set()
    for expr in sets:
        name, text = parse_set_expression(expr)
        if name in applied:
            raise _UsageError(f"重复的 --set 参数：{name}")
        try:
            value = coerce_set_value(name, text, EncounterSettings)
        except ValueError as exc:
            raise _UsageError(str(exc)) from exc
        setattr(settings, name, value)
        applied.add(name)
    return applied


def _parse_scan(text: str, applied: set[str]) -> ScanSpec:
    parts = text.split(":")
    if len(parts) != 4:
        raise _UsageError(
            f"无效的 --scan：{text!r}（应为 parameter:start:stop:step）"
        )
    parameter, start, stop, step = parts
    if not parameter:
        raise _UsageError("--scan 参数名不能为空")
    if parameter in applied:
        raise _UsageError(f"--scan 与 --set 参数冲突：{parameter}")
    try:
        return ScanSpec(parameter, float(start), float(stop), float(step))
    except ValueError as exc:
        raise _UsageError(str(exc)) from exc


def _cmd_run(args: argparse.Namespace) -> int:
    if args.loop_times < 1:
        raise _UsageError("--loop-times 必须不小于 1")
    if args.workers < 1:
        raise _UsageError("--workers 必须不小于 1")
    settings = EncounterSettings()
    applied = _apply_sets(settings, args.set)
    scan = _parse_scan(args.scan, applied) if args.scan is not None else None
    run_config = RunConfig(
        loop_times=args.loop_times,
        workers=args.workers,
        seed=args.seed,
        name=args.name,
        output_dir=args.output_dir or "data/raw",
        progress=args.progress,
    )
    controller = EncounterController()
    paths = controller.run(settings, run_config, scan)
    for path in paths:
        print(f"已生成：{path}")
    print(f"完成：共 {len(paths)} 份源文件")
    return 0


def _cmd_postprocess(args: argparse.Namespace) -> int:
    if args.start_average < 0:
        raise _UsageError("--start-average 必须不小于 0")
    config = PostprocessConfig(start_average=args.start_average)
    processor = EncounterPostprocessor(
        config=config, output_dir=args.output_dir or "data/processed"
    )
    target = Path(args.input)
    if target.is_file():
        result = processor.process_file(target)
        print(f"已生成：{result}")
    elif target.is_dir():
        result = processor.process_submission(target)
        print(f"已生成：{result}")
    else:
        raise FileNotFoundError(f"输入不存在：{args.input}")
    return 0


def _cmd_plot(args: argparse.Namespace) -> int:
    target = Path(args.input)
    if not target.is_file():
        raise FileNotFoundError(f"输入不存在：{args.input}")
    plotter = EncounterFeaturePlotter()
    outputs = plotter.plot(target, output_dir=args.output_dir)
    for output in outputs:
        print(f"已生成：{output}")
    return 0


def _cmd_animate(args: argparse.Namespace) -> int:
    if args.frame_step < 1:
        raise _UsageError("--frame-step 必须不小于 1")
    if args.fps < 1:
        raise _UsageError("--fps 必须不小于 1")
    if args.dpi < 1:
        raise _UsageError("--dpi 必须不小于 1")
    animator = EncounterAnimate.from_source(args.source, frame_step=args.frame_step)
    animator.save(
        args.output,
        fps=args.fps,
        dpi=args.dpi,
        color_mode=args.color_mode,
        aspect_mode=args.aspect_mode,
    )
    print(f"已生成：{args.output}")
    return 0


def _cmd_snapshot(args: argparse.Namespace) -> int:
    if args.step is not None and args.step < 0:
        raise _UsageError("--step 必须不小于 0")
    if args.dpi < 1:
        raise _UsageError("--dpi 必须不小于 1")
    if args.step is None:
        store = EncounterDataStore()
        with store._open_readonly(args.source) as f:
            step = f["trajectory/position"].shape[0] - 1
    else:
        step = args.step
    EncounterAnimate.render_snapshot(
        args.source,
        args.output,
        step,
        dpi=args.dpi,
        color_mode=args.color_mode,
        aspect_mode=args.aspect_mode,
        label=args.label,
    )
    print(f"已生成：{args.output}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """cooperation 入口；返回退出码（0 成功、1 运行失败、2 参数错误）。"""
    arguments = list(sys.argv[1:] if argv is None else argv)
    arguments = _route_default_run(arguments)
    parser = _build_parser()
    args = parser.parse_args(arguments)
    if args.command is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "postprocess":
            return _cmd_postprocess(args)
        if args.command == "plot":
            return _cmd_plot(args)
        if args.command == "animate":
            return _cmd_animate(args)
        if args.command == "snapshot":
            return _cmd_snapshot(args)
        raise ValueError(f"未知子命令：{args.command}")
    except _UsageError as exc:
        print(f"参数错误：{exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001 运行或数据失败统一 stderr + 退出码 1
        print(f"错误：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
