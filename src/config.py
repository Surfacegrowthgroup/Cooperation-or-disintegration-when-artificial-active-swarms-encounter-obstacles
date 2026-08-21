"""encounter 的扫描、调度与后处理配置类型及公共辅助函数。

本模块只承载参数职责：求解器参数保留在 ``settings.EncounterSettings``，
扫描定义、运行调度、后处理配置与本模块的序列化/摘要/解析辅助分离在
此。模块不导入求解器、存储、后处理、动画或 multiprocessing。
"""

from __future__ import annotations

import dataclasses
import json
import math
import typing
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .settings import EncounterSettings


@dataclass(frozen=True, slots=True)
class ScanSpec:
    """单参数扫描定义：一次只扫描一个可写数值型求解参数。

    ``values()`` 生成从 ``start`` 到 ``stop`` 的等差序列（含终点，
    按数值容差包含），支持正向与反向扫描；``step`` 不能为零且方向
    必须能到达 ``stop``。整数字段的每个扫描值必须是整数。
    """

    parameter: str
    start: float
    stop: float
    step: float

    def __post_init__(self) -> None:
        hints = typing.get_type_hints(EncounterSettings)
        if self.parameter not in hints or hints[self.parameter] not in (int, float):
            raise ValueError(
                f"扫描参数必须是 EncounterSettings 中的可写数值字段，"
                f"得到 {self.parameter!r}"
            )
        if self.step == 0:
            raise ValueError("扫描步长 step 不能为零")
        span = float(self.stop) - float(self.start)
        if span * float(self.step) < 0:
            raise ValueError("扫描方向无法到达 stop：step 与 (stop - start) 必须同号")
        if hints[self.parameter] is int:
            for name, value in (
                ("start", self.start),
                ("stop", self.stop),
                ("step", self.step),
            ):
                if abs(float(value) - round(float(value))) > 1e-12:
                    raise ValueError(f"整数字段 {self.parameter} 的 {name} 必须是整数")

    @property
    def is_integer_parameter(self) -> bool:
        return typing.get_type_hints(EncounterSettings)[self.parameter] is int

    def values(self) -> np.ndarray:
        """生成扫描值序列；终点按数值容差包含，不产生超过容差的额外点。"""
        start = float(self.start)
        stop = float(self.stop)
        step = float(self.step)
        if start == stop:
            return np.array([start], dtype=float)
        tol = 1e-9 * max(1.0, abs(step))
        # 容差沿扫描方向：正向加 tol、反向减 tol，保证终点按容差包含
        n = int(math.floor((stop - start + math.copysign(tol, step)) / step)) + 1
        out = start + step * np.arange(n, dtype=float)
        if self.is_integer_parameter:
            rounded = np.round(out)
            if not np.all(np.abs(out - rounded) <= 1e-9):
                raise ValueError(
                    f"整数字段 {self.parameter} 的扫描值必须全部为整数"
                )
            out = rounded
        return out


@dataclass(frozen=True, slots=True)
class RunConfig:
    """运行调度配置：重复数、worker 数、seed、提交名与输出根目录。

    ``seed`` 默认 None 且保持 None，不在配置层替换为固定值；显式
    seed 由中控按重复/扫描序号确定性派生。``output_dir`` 为源数据
    根目录，实际文件写入 ``output_dir/<submission-name>/``。
    """

    loop_times: int = 1
    workers: int = 1
    seed: int | None = None
    name: str | None = None
    output_dir: str | Path = "data/raw"
    progress: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.loop_times, bool)
            or not isinstance(self.loop_times, int)
            or self.loop_times < 1
        ):
            raise ValueError("loop_times 必须是不小于 1 的整数")
        if (
            isinstance(self.workers, bool)
            or not isinstance(self.workers, int)
            or self.workers < 1
        ):
            raise ValueError("workers 必须是不小于 1 的整数")


@dataclass(frozen=True, slots=True)
class PostprocessConfig:
    """后处理配置：特征时间平均起点。"""

    start_average: int = 3000

    def __post_init__(self) -> None:
        if (
            isinstance(self.start_average, bool)
            or not isinstance(self.start_average, int)
            or self.start_average < 0
        ):
            raise ValueError("start_average 必须是不小于 0 的整数")


# ---------------------------------------------------------------- 序列化

def _json_default(obj: object) -> object:
    if isinstance(obj, Path):
        if obj.is_absolute():
            raise ValueError("拒绝序列化设备绝对路径")
        return obj.as_posix()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return dataclasses.asdict(obj)
    raise TypeError(f"无法 JSON 序列化：{type(obj).__name__}")


def json_dumps(obj: object) -> str:
    """安全 JSON 序列化：排序键、紧凑分隔符；不使用 pickle 或 object dtype。"""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), default=_json_default
    )


def json_roundtrip(obj: object) -> str:
    """dumps -> loads -> dumps，用于验证配置与身份的 JSON 往返一致。"""
    return json_dumps(json.loads(json_dumps(obj)))


# ------------------------------------------------------------ 稳定摘要

def format_scalar(value: object) -> str:
    """确定性标量格式：浮点使用最短往返表示（repr），整数直写。"""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"不支持的标量类型：{type(value).__name__}")


def stable_settings_summary(settings: EncounterSettings) -> str:
    """稳定参数摘要：按字段声明序拼接 ``name=value``，浮点确定性表示。"""
    parts = []
    for field in dataclasses.fields(settings):
        parts.append(f"{field.name}={format_scalar(getattr(settings, field.name))}")
    return ";".join(parts)


# ---------------------------------------------------------- --set 解析

def parse_set_expression(expr: str) -> tuple[str, str]:
    """解析 ``name=value`` 赋值表达式，返回 (字段名, 值文本)。"""
    name, sep, text = expr.partition("=")
    name = name.strip()
    text = text.strip()
    if not sep or not name or not text:
        raise ValueError(f"无效的 --set 表达式：{expr!r}（应为 name=value）")
    return name, text


def coerce_set_value(name: str, text: str, settings_cls=EncounterSettings) -> object:
    """按 EncounterSettings 字段类型把 --set 值文本解析为 Python 标量。"""
    hints = typing.get_type_hints(settings_cls)
    if name not in hints:
        raise ValueError(f"未知参数：{name}")
    target = hints[name]
    if target is int:
        return int(text)
    if target is float:
        return float(text)
    if target is bool:
        return text.strip().lower() in ("true", "1", "yes", "on")
    if target is str:
        return text
    raise ValueError(f"不支持的参数类型：{name}: {target.__name__}")


__all__ = [
    "ScanSpec",
    "RunConfig",
    "PostprocessConfig",
    "json_dumps",
    "json_roundtrip",
    "format_scalar",
    "stable_settings_summary",
    "parse_set_expression",
    "coerce_set_value",
]
