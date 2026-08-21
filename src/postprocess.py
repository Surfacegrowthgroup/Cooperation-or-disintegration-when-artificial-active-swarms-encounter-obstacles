"""独立特征后处理（任务 06）。

:class:`EncounterPostprocessor` 只读取完整 HDF5 v1 源数据，计算通过率、
左右有序度与左右聚类系数的时间序列与最终标量，生成逐运行特征文件与
完整扫描汇总；绝不修改源文件。特征语义与任务 01 冻结一致：对 HDF5
第 1..T 帧（推进后状态）计算，输出索引 t 对应帧 t+1；ratio 取最终
状态，其余量对 ``start_average`` 之后的时间段平均；无有效样本或区域
为空时返回零。按帧小批次读取，不一次加载整个大轨迹。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import PostprocessConfig, json_dumps
from .storage import EncounterDataStore
from .vm_engine import compute_cluster_coefficients_open_x_periodic_y

FEATURE_KEYS = ("ratio", "OrderL", "OrderR", "ClusterL", "ClusterR")
# 逐时刻键：{key}_ts
FEATURE_TS_KEYS = tuple(f"{key}_ts" for key in FEATURE_KEYS)
FEATURES_SCHEMA = "encounter-features-v1"
SUMMARY_SCHEMA = "encounter-summary-v1"


def passing_ratio(positions: np.ndarray, threshold: float) -> float:
    """论文定义的瞬时通过率：x 坐标大于阈值的粒子占比；空粒子场景约定为零。"""
    if positions.shape[0] == 0:
        return 0.0
    return float(np.mean(positions[:, 0] > threshold))


def _order_from_angles(angles: np.ndarray) -> float:
    """有序度：|mean(cosθ, sinθ)|；角度可直接推导速度，无需保存速度。"""
    if angles.shape[0] == 0:
        return 0.0
    v = np.column_stack((np.cos(angles), np.sin(angles)))
    return float(np.linalg.norm(v.sum(axis=0) / v.shape[0]))


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class EncounterPostprocessor:
    """从 HDF5 源文件计算特征；默认写入 data/processed/<submission>/。"""

    store: EncounterDataStore = field(default_factory=EncounterDataStore)
    config: PostprocessConfig = field(default_factory=PostprocessConfig)
    output_dir: str | Path = "data/processed"

    # ------------------------------------------------------------ 单文件

    def process_file(
        self, source: str | Path, *, output_dir: str | Path | None = None
    ) -> Path:
        """为一份源 HDF5 生成同名 ``.features.npz``；已有目标拒绝覆盖。

        特征只计算 HDF5 帧 1..T（推进后状态），输出 step 为 0..T-1。
        """
        source_path = Path(source)
        meta = self.store.metadata(source_path)
        submission = str(meta["submission"])
        root = Path(output_dir) if output_dir is not None else Path(self.output_dir)
        target = root / submission / f"{source_path.stem}.features.npz"
        if target.exists():
            raise FileExistsError(f"特征文件已存在，拒绝覆盖：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)

        length = float(meta["scan_value"]) if str(meta["scan_parameter"]) else None
        features = self._compute_features(
            source_path,
            submission=submission,
            length=length,
            settings_json=str(meta["settings_json"]),
        )
        payload = _feature_payload(
            features,
            source_path,
            meta,
            source_sha256=_source_sha256(source_path),
        )
        np.savez(target, **payload)
        return target

    # ------------------------------------------------------------ 汇总

    def process_submission(
        self, source_dir: str | Path, *, output_dir: str | Path | None = None
    ) -> Path:
        """验证完整重复矩阵并生成一份 ``summary.npz``；已有目标拒绝覆盖。

        先完整验证全部源文件的 schema、提交身份、扫描参数/值与重复矩阵，
        再开始写出。缺失的逐运行特征文件自动生成；已存在的特征文件校验
        ``source_sha256`` 与源文件一致。可变 ``times`` 以 NaN 补齐到最大
        长度并记录 ``valid_steps``。
        """
        directory = Path(source_dir)
        records: list[_SourceRecord] = []
        for source in sorted(directory.glob("*.h5")):
            meta = self.store.metadata(source)
            records.append(_SourceRecord(source, meta, {}))

        submission = _validate_submission(records)
        root = Path(output_dir) if output_dir is not None else Path(self.output_dir)
        target = root / submission / "summary.npz"
        if target.exists():
            raise FileExistsError(f"summary 已存在，拒绝覆盖：{target}")
        target.parent.mkdir(parents=True, exist_ok=True)

        # 逐源生成或复用特征文件，并校验哈希
        for record in records:
            features_path = (
                root / submission / f"{record.path.stem}.features.npz"
            )
            if features_path.exists():
                record.features = _load_features(features_path)
                _validate_feature_payload(record.features)
                expected = str(record.features["source_sha256"][0])
                actual = _source_sha256(record.path)
                if expected != actual:
                    raise ValueError(
                        f"特征文件哈希与源文件不一致：{record.path.name}"
                    )
            else:
                features = self._compute_features(
                    record.path,
                    submission=submission,
                    length=(
                        float(record.meta["scan_value"])
                        if str(record.meta["scan_parameter"])
                        else None
                    ),
                    settings_json=str(record.meta["settings_json"]),
                )
                record.features = _feature_payload(
                    features,
                    record.path,
                    record.meta,
                    source_sha256=_source_sha256(record.path),
                )
                np.savez(features_path, **record.features)

        # 按扫描顺序聚合：var -> {重复列表}
        scan_indices = sorted({int(r.meta["scan_index"]) for r in records})
        var = np.array(
            [
                float(
                    next(
                        r.meta["scan_value"]
                        for r in records
                        if int(r.meta["scan_index"]) == idx
                    )
                )
                for idx in scan_indices
            ],
            dtype=np.float64,
        )
        max_steps = max(
            record.features["ratio_ts"].shape[0] for record in records
        )
        summary: dict[str, object] = {
            "schema": np.array([SUMMARY_SCHEMA]),
            "var": var,
            "valid_steps": np.zeros(var.shape, dtype=np.int64),
        }
        for key in FEATURE_KEYS:
            summary[key] = np.zeros(var.shape, dtype=np.float64)
            summary[f"{key}_ts"] = np.full(
                (var.shape[0], max_steps), np.nan, dtype=np.float64
            )

        for var_index, scan_index in enumerate(scan_indices):
            group = [
                r
                for r in records
                if int(r.meta["scan_index"]) == scan_index
            ]
            steps = group[0].features["ratio_ts"].shape[0]
            if any(
                r.features["ratio_ts"].shape[0] != steps for r in group
            ):
                raise ValueError(
                    f"同一参数点的重复长度不一致：scan_index={scan_index}"
                )
            summary["valid_steps"][var_index] = steps
            for key in FEATURE_KEYS:
                ts = np.mean(
                    np.stack([r.features[f"{key}_ts"] for r in group]), axis=0
                )
                summary[f"{key}_ts"][var_index, :steps] = ts
                summary[key][var_index] = float(
                    np.mean([float(r.features[key][0]) for r in group])
                )
        summary["metadata_json"] = np.array(
            [
                json_dumps(
                    {
                        "submission": submission,
                        "scan_parameter": str(records[0].meta["scan_parameter"]),
                    }
                )
            ]
        )
        np.savez(target, **summary)
        return target

    def _compute_features(
        self,
        source: Path,
        *,
        submission: str,
        length: float | None,
        settings_json: str,
    ) -> dict[str, np.ndarray]:
        """逐帧读取并计算特征时间序列与标量（帧 1..T）。"""
        settings = json.loads(settings_json)
        width = float(settings["width"])
        radius = float(settings["radius"])
        begin_length = float(settings["place_length"]) + float(settings["white_length"])
        if length is None:
            length = float(settings["length"])
        threshold = begin_length + length
        start_average = self.config.start_average
        times = max(0, int(settings["times"]))
        n_frames = times + 1

        with self.store._open_readonly(source) as f:
            n_particles = f["trajectory/position"].shape[1]
            ts = {key: np.zeros(times, dtype=np.float64) for key in FEATURE_KEYS}
            for t in range(times):
                frame = t + 1  # 推进后记录：输出索引 t 对应 HDF5 帧 t+1
                position = f["trajectory/position"][frame]
                angle = f["trajectory/angle"][frame]
                left = position[:, 0] < begin_length
                right = position[:, 0] > threshold
                ts["ratio"][t] = passing_ratio(position, threshold)
                ts["OrderL"][t] = _order_from_angles(angle[left])
                ts["OrderR"][t] = _order_from_angles(angle[right])
                ts["ClusterL"][t] = _cluster_mean(f, position, width, radius, left)
                ts["ClusterR"][t] = _cluster_mean(f, position, width, radius, right)

            result: dict[str, np.ndarray] = {
                "step": np.arange(times, dtype=np.int64),
                **{f"{key}_ts": ts[key] for key in FEATURE_KEYS},
            }
            for key in FEATURE_KEYS:
                if key == "ratio":
                    value = passing_ratio(
                        f["trajectory/position"][n_frames - 1], threshold
                    )
                else:
                    samples = ts[key][start_average:]
                    value = float(np.mean(samples)) if samples.size else 0.0
                result[key] = np.array([value])
            return result


def _cluster_mean(
    f, position: np.ndarray, width: float, radius: float, mask: np.ndarray
) -> float:
    """局部聚类系数在 mask 内的均值；区域为空返回 0。"""
    if not np.any(mask):
        return 0.0
    values = compute_cluster_coefficients_open_x_periodic_y(
        position, width, radius, mask
    )
    return float(np.mean(values[0][mask]))


@dataclass(slots=True)
class _SourceRecord:
    path: Path
    meta: dict[str, object]
    features: dict[str, np.ndarray] = field(default_factory=dict)


def _load_features(features_path: Path) -> dict[str, np.ndarray]:
    with np.load(features_path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _feature_payload(
    features: dict[str, np.ndarray],
    source: Path,
    meta: dict[str, object],
    *,
    source_sha256: str,
) -> dict[str, np.ndarray]:
    """组装单运行特征契约，供单文件和扫描共享。"""
    identity = {
        "submission": str(meta["submission"]),
        "scan_parameter": str(meta["scan_parameter"]),
        "scan_value": float(meta["scan_value"]),
        "scan_index": int(meta["scan_index"]),
        "repeat_index": int(meta["repeat_index"]),
        "seed_mode": str(meta["seed_mode"]),
        "seed": int(meta["seed"]),
        "source": source.name,
    }
    return {
        "schema": np.array([FEATURES_SCHEMA]),
        **features,
        "source_sha256": np.array([source_sha256]),
        "metadata_json": np.array([json_dumps(identity)]),
    }


def _validate_feature_payload(payload: dict[str, np.ndarray]) -> None:
    """校验后处理生成的单运行特征文件契约。"""
    schema = payload.get("schema")
    if schema is None or schema.ndim != 1 or schema.shape[0] != 1:
        raise ValueError("特征 NPZ 缺少 schema")
    if str(schema[0]) != FEATURES_SCHEMA:
        raise ValueError("特征 NPZ schema 不支持，请重新运行后处理")
    required = {
        "step",
        *FEATURE_KEYS,
        *FEATURE_TS_KEYS,
        "source_sha256",
        "metadata_json",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"特征 NPZ 缺少字段：{missing}")


def _validate_submission(records: list[_SourceRecord]) -> str:
    """校验提交身份、扫描定义与重复矩阵完整性；返回 submission。"""
    if not records:
        raise ValueError("目录中没有源 HDF5 文件")
    submissions = {str(r.meta["submission"]) for r in records}
    if len(submissions) != 1:
        raise ValueError(f"混合提交：{sorted(submissions)}")
    submission = submissions.pop()
    parameters = {str(r.meta["scan_parameter"]) for r in records}
    if len(parameters) != 1:
        raise ValueError(f"混合扫描参数：{sorted(parameters)}")
    scan_parameter = parameters.pop()
    if scan_parameter:
        values_by_index = {
            int(r.meta["scan_index"]): float(r.meta["scan_value"]) for r in records
        }
        if sorted(values_by_index) != list(range(max(values_by_index) + 1)):
            raise ValueError("扫描序号不连续：缺失参数点")
        if len(set(values_by_index.values())) != len(values_by_index):
            raise ValueError("同一扫描序号出现多个不同参数值")
    else:
        if any(int(r.meta["scan_index"]) != 0 for r in records):
            raise ValueError("无扫描提交中出现了非 0 扫描序号")
    seen: set[tuple[int, int]] = set()
    for r in records:
        key = (int(r.meta["repeat_index"]), int(r.meta["scan_index"]))
        if key in seen:
            raise ValueError(f"重复身份：repetition={key[0]} scan_index={key[1]}")
        seen.add(key)
    # 重复矩阵完整性：每个重复必须覆盖全部扫描序号
    repeats = sorted({int(r.meta["repeat_index"]) for r in records})
    indices = sorted({int(r.meta["scan_index"]) for r in records})
    expected = {(rep, idx) for rep in repeats for idx in indices}
    missing = expected - seen
    if missing:
        raise ValueError(f"重复矩阵不完整，缺失：{sorted(missing)}")
    return submission


__all__ = [
    "EncounterPostprocessor",
    "FEATURE_KEYS",
    "FEATURE_TS_KEYS",
    "FEATURES_SCHEMA",
    "SUMMARY_SCHEMA",
]
