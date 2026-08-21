"""从后处理 NPZ 生成特征量 PDF 图。

本模块只负责读取 ``encounter-features-v1`` 和 ``encounter-summary-v1``，
不依赖 HDF5、求解器或后处理器。Matplotlib 仅在实际绘图时延迟加载。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .postprocess import (
    FEATURES_SCHEMA,
    FEATURE_KEYS,
    FEATURE_TS_KEYS,
    SUMMARY_SCHEMA,
)

_FEATURE_METADATA_KEYS = {
    "submission",
    "scan_parameter",
    "scan_value",
    "scan_index",
    "repeat_index",
    "seed_mode",
    "seed",
    "source",
}
_SCAN_LABELS = {
    "length": r"$L_{\mathrm{o}}$",
    "que_density": r"$\rho_{\mathrm{o}}$",
    "que_radius": r"$R_{\mathrm{o}}$",
    "end_length": r"$H$",
    "speed": r"$v_{0}$",
    "par_density": r"$\rho$",
}


@dataclass(frozen=True, slots=True)
class _LoadedData:
    schema: str
    payload: dict[str, np.ndarray]
    metadata: dict[str, Any]


class EncounterFeaturePlotter:
    """将单运行特征或扫描汇总绘制为一组单页 PDF。"""

    def __init__(self, *, dpi: int = 200) -> None:
        if isinstance(dpi, bool) or dpi < 1:
            raise ValueError("dpi 必须是正整数")
        self.dpi = int(dpi)

    def plot(
        self,
        source: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """自动识别 NPZ schema 并生成 PDF，返回稳定排序的输出路径。"""
        loaded = _load_npz(Path(source))
        target_dir = self._target_dir(loaded.metadata, output_dir)
        if loaded.schema == FEATURES_SCHEMA:
            targets = self._feature_targets(loaded, target_dir)
        elif loaded.schema == SUMMARY_SCHEMA:
            targets = self._summary_targets(target_dir)
        else:  # pragma: no cover - _load_npz 已经覆盖该分支
            raise ValueError(f"不支持的 NPZ schema：{loaded.schema}")
        _preflight_targets(targets)

        # 延迟导入，确保普通 encounter 导入不加载绘图库或修改样式。
        from .plot_style import (
            COL_C,
            COL_R,
            COL_V,
            fig3_panels,
            scan_panel,
            setup_style,
        )

        setup_style()
        if loaded.schema == FEATURES_SCHEMA:
            return fig3_panels(
                loaded.payload["step"],
                loaded.payload["ratio_ts"],
                loaded.payload["OrderL_ts"],
                loaded.payload["ClusterL_ts"],
                loaded.payload["OrderR_ts"],
                loaded.payload["ClusterR_ts"],
                target_dir,
                xlim=_time_xlim(loaded.payload["step"]),
                formats=("pdf",),
                dpi=self.dpi,
                output_names=(targets[0].stem, targets[1].stem),
            )

        parameter = str(loaded.metadata["scan_parameter"])
        var = loaded.payload["var"]
        outputs = [
            scan_panel(
                var,
                [(r"$R$", loaded.payload["ratio"], COL_R, "o")],
                _scan_parameter_label(parameter),
                r"$R$",
                targets[0],
                panel_label="(a)",
                dpi=self.dpi,
            ),
            scan_panel(
                var,
                [
                    (r"$V_{1}$", loaded.payload["OrderL"], COL_V, "o"),
                    (r"$C_{1}$", loaded.payload["ClusterL"], COL_C, "s"),
                ],
                _scan_parameter_label(parameter),
                r"$V_{1}, C_{1}$",
                targets[1],
                panel_label="(b)",
                dpi=self.dpi,
            ),
            scan_panel(
                var,
                [
                    (r"$V_{2}$", loaded.payload["OrderR"], COL_V, "o"),
                    (r"$C_{2}$", loaded.payload["ClusterR"], COL_C, "s"),
                ],
                _scan_parameter_label(parameter),
                r"$V_{2}, C_{2}$",
                targets[2],
                panel_label="(c)",
                dpi=self.dpi,
            ),
        ]
        return outputs

    @staticmethod
    def _target_dir(
        metadata: dict[str, Any], output_dir: str | Path | None
    ) -> Path:
        root = Path(output_dir) if output_dir is not None else Path("tmp/feature-plots")
        return root / _safe_filename(metadata["submission"])

    @staticmethod
    def _feature_targets(loaded: _LoadedData, target_dir: Path) -> list[Path]:
        source = Path(str(loaded.metadata["source"])).name
        run = _safe_filename(Path(source).stem)
        return [
            target_dir / f"{run}-time-returning.pdf",
            target_dir / f"{run}-time-passing.pdf",
        ]

    @staticmethod
    def _summary_targets(target_dir: Path) -> list[Path]:
        return [
            target_dir / "scan-passage-rate.pdf",
            target_dir / "scan-returning.pdf",
            target_dir / "scan-passing.pdf",
        ]


def _load_npz(path: Path) -> _LoadedData:
    if not path.is_file():
        raise FileNotFoundError(f"输入不存在：{path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            payload = {key: data[key] for key in data.files}
    except ValueError as exc:
        raise ValueError(f"无法安全读取 NPZ：{path}") from exc

    schema = _read_single_string(payload, "schema")
    if schema == FEATURES_SCHEMA:
        metadata = _validate_features(payload)
    elif schema == SUMMARY_SCHEMA:
        metadata = _validate_summary(payload)
    else:
        raise ValueError(f"不支持的 NPZ schema：{schema}")
    return _LoadedData(schema, payload, metadata)


def _validate_features(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    _require_keys(
        payload,
        {
            "step",
            *FEATURE_KEYS,
            *FEATURE_TS_KEYS,
            "source_sha256",
            "metadata_json",
        },
    )
    step = payload["step"]
    if step.ndim != 1 or step.size == 0 or step.dtype.kind not in "iu":
        raise ValueError("features.step 必须是非空整数一维数组")
    if not np.array_equal(step, np.arange(step.size, dtype=step.dtype)):
        raise ValueError("features.step 必须是从 0 开始的连续序列")
    for key in FEATURE_TS_KEYS:
        values = payload[key]
        if values.ndim != 1 or values.size != step.size:
            raise ValueError(f"{key} 必须与 step 等长的一维数组")
        _require_finite(key, values)
    for key in FEATURE_KEYS:
        values = payload[key]
        if values.ndim != 1 or values.size != 1:
            raise ValueError(f"{key} 必须是长度为 1 的一维数组")
        _require_finite(key, values)
    _validate_single_string(payload, "source_sha256", nonempty=True)
    return _read_metadata(payload, _FEATURE_METADATA_KEYS)


def _validate_summary(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    _require_keys(payload, {"var", "valid_steps", *FEATURE_KEYS, *FEATURE_TS_KEYS, "metadata_json"})
    var = payload["var"]
    valid_steps = payload["valid_steps"]
    if var.ndim != 1 or var.size == 0:
        raise ValueError("summary.var 必须是非空一维数组")
    _require_finite("var", var)
    if valid_steps.ndim != 1 or valid_steps.size != var.size or valid_steps.dtype.kind not in "iu":
        raise ValueError("summary.valid_steps 必须是与 var 等长的整数一维数组")
    if np.any(valid_steps < 1):
        raise ValueError("summary.valid_steps 必须为正数")
    for key in FEATURE_KEYS:
        values = payload[key]
        if values.ndim != 1 or values.size != var.size:
            raise ValueError(f"summary.{key} 必须与 var 等长的一维数组")
        _require_finite(key, values)
    max_steps = None
    for key in FEATURE_TS_KEYS:
        values = payload[key]
        if values.ndim != 2 or values.shape[0] != var.size or values.shape[1] == 0:
            raise ValueError(f"summary.{key} 必须是 [扫描点, 步长] 二维数组")
        if max_steps is None:
            max_steps = values.shape[1]
        elif values.shape[1] != max_steps:
            raise ValueError("summary 时间序列的步长维度不一致")
        if np.any(valid_steps > values.shape[1]):
            raise ValueError("summary.valid_steps 超出时间序列长度")
        for row, steps in zip(values, valid_steps):
            _require_finite(key, row[: int(steps)])
            if np.any(np.isfinite(row[int(steps) :])):
                raise ValueError(f"{key} 的 valid_steps 之后只能使用 NaN 填充")
    metadata = _read_metadata(payload, {"submission", "scan_parameter"})
    if not str(metadata["scan_parameter"]):
        raise ValueError("扫描汇总 metadata_json 缺少非空 scan_parameter")
    return metadata


def _read_metadata(payload: dict[str, np.ndarray], required: set[str]) -> dict[str, Any]:
    raw = _read_single_string(payload, "metadata_json")
    try:
        metadata = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("metadata_json 不是有效 JSON") from exc
    if not isinstance(metadata, dict):
        raise ValueError("metadata_json 必须编码 JSON 对象")
    missing = sorted(required - metadata.keys())
    if missing:
        raise ValueError(f"metadata_json 缺少字段：{missing}")
    if not str(metadata["submission"]):
        raise ValueError("metadata_json.submission 不能为空")
    return metadata


def _read_single_string(payload: dict[str, np.ndarray], key: str) -> str:
    if key not in payload:
        raise ValueError(f"NPZ 缺少字段：{key}")
    values = payload[key]
    if values.ndim != 1 or values.size != 1 or values.dtype.kind not in "SU":
        raise ValueError(f"{key} 必须是长度为 1 的字符串数组")
    value = values[0]
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_single_string(
    payload: dict[str, np.ndarray], key: str, *, nonempty: bool = False
) -> None:
    value = _read_single_string(payload, key)
    if nonempty and not value:
        raise ValueError(f"{key} 不能为空")


def _require_keys(payload: dict[str, np.ndarray], required: set[str]) -> None:
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"NPZ 缺少字段：{missing}")


def _require_finite(name: str, values: np.ndarray) -> None:
    if values.dtype.kind not in "buif" or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} 含有非有限数值或不支持的数据类型")


def _preflight_targets(targets: list[Path]) -> None:
    existing = [path for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"输出已存在，拒绝覆盖：{existing[0]}")


def _time_xlim(step: np.ndarray) -> tuple[float, float]:
    left, right = float(step[0]), float(step[-1])
    if left == right:
        pad = max(abs(left) * 0.05, 0.5)
        return left - pad, right + pad
    return left, right


def _scan_parameter_label(parameter: str) -> str:
    if parameter in _SCAN_LABELS:
        return _SCAN_LABELS[parameter]
    # EncounterSettings 字段名只允许安全字符；未知字段不接受任意 TeX 片段。
    safe = re.sub(r"[^A-Za-z0-9_]", "", parameter)
    if not safe:
        raise ValueError("scan_parameter 不是有效字段名")
    return "$\\mathrm{" + safe.replace("_", r"\_") + "}$"


def _safe_filename(value: object) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]", "_", str(value))
    return cleaned.strip("._") or "submission"


__all__ = ["EncounterFeaturePlotter"]
