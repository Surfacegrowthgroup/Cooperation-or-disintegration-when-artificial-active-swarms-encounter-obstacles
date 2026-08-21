"""中控：提交展开、扫描链调度与 worker 管理（任务 05）。

:class:`EncounterController.run` 把一个提交展开为 ``loop_times`` 条
重复扫描链：每个 worker 串行执行一条完整链（scan_index 严格递增），
单次求解保持串行；求解结束后由同一 worker 调用存储模块写入一份源
HDF5。并行只存在于本层；求解器模块不导入 multiprocessing。

seed 规则：显式 seed 以 ``SeedSequence(user_seed)`` 为根，按
(repetition_index, scan_index) 确定性派生子 seed，与 worker 数、PID
和调度无关；无 seed 时每个原子运行在展开阶段单独以系统熵创建
``SeedSequence``，entropy 写入文件身份与文件名。失败时停止提交、
取消未启动 future、保留已完成文件并返回非零状态。
"""

from __future__ import annotations

import dataclasses
import multiprocessing
import queue as std_queue
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path

import numpy as np

from .config import RunConfig, ScanSpec
from .settings import EncounterSettings
from .simulation import EncounterSimulation
from .storage import EncounterDataStore, hdf5_filename

# 并行模式下等待任务完成的最长时间（秒），超时视为 worker 挂死
_WATCHDOG_S = 2 * 60 * 60
# 主循环轮询间隔（秒）：任务进行中也要消费进度消息，保持 worker 条逐帧推进
_POLL_S = 0.2

# 并行进度队列容量（条数）；满时 worker 用 put_nowait 丢帧，记账按高水位补差
_PROGRESS_QUEUE_MAXSIZE = 64
# 步进进度上报粒度目标：每任务至多约 times / _PROGRESS_EVERY_TARGET 条消息
_PROGRESS_EVERY_TARGET = 100

# worker 进程内通过 initializer 注入的进度队列与上报粒度（spawn 下每进程独立副本）
_PROGRESS_QUEUE: multiprocessing.Queue | None = None
_PROGRESS_EVERY: int = 1


@dataclass(frozen=True, slots=True)
class _PointSpec:
    """链上的一个原子运行：扫描位置、实际 seed 与目标路径。"""

    scan_index: int
    parameter_value: float | None  # None 表示无扫描（使用 settings 当前值）
    seed: int
    seed_mode: str
    target: Path
    fail_at: bool = False  # 测试/诊断用失败注入点


@dataclass(frozen=True, slots=True)
class _ChainSpec:
    """一条重复链：一个 worker 串行执行的全部扫描点。"""

    repetition_index: int
    points: list[_PointSpec]
    settings: EncounterSettings
    scan_parameter: str
    submission: str


def _sweep_worker_init(progress_queue: multiprocessing.Queue, k_every: int) -> None:
    """worker 进程初始化：注入进度队列与上报粒度（spawn 下每进程独立副本）。"""
    global _PROGRESS_QUEUE, _PROGRESS_EVERY
    _PROGRESS_QUEUE = progress_queue
    _PROGRESS_EVERY = k_every
    progress_queue.cancel_join_thread()


def _emit_progress(slot: int, t: int, times: int) -> None:
    """worker 内步进回调：按粒度把 (slot, t) 投递到主进程，满则丢帧。"""
    q = _PROGRESS_QUEUE
    if q is None:
        return
    if (t + 1) % _PROGRESS_EVERY == 0 or t == times - 1:
        try:
            q.put_nowait(("step", slot, t))
        except (std_queue.Full, OSError, ValueError):
            pass  # 丢帧可接受：主进程按高水位补差对齐


def _emit_point_done(slot: int, repetition_index: int, scan_index: int) -> None:
    q = _PROGRESS_QUEUE
    if q is None:
        return
    try:
        q.put_nowait(("point", slot, repetition_index, scan_index))
    except (std_queue.Full, OSError, ValueError):
        pass


def _progress_delta(hw: int, t: int, times: int) -> tuple[int, int] | None:
    """返回 (增量, 新高水位)；过期或越界消息返回 None。"""
    if t > hw and t < times:
        return t - hw, t
    return None


def _progress_postfix(proc, format_dict: dict) -> dict[str, str]:
    """构造进度条 postfix：CPU 占用、内存占用与预计完成时刻。"""
    import psutil

    cpu = proc.cpu_percent(interval=None)
    rss_mb = (
        proc.memory_info().rss
        + sum(child.memory_info().rss for child in proc.children())
    ) / 1048576.0
    n_done = format_dict.get("n", 0)
    total_units = format_dict.get("total") or 0
    elapsed = format_dict.get("elapsed", 0.0)
    finish = ""
    if n_done > 0 and total_units and elapsed > 0.0:
        remaining = elapsed * (total_units - n_done) / n_done
        finish = (datetime.now() + timedelta(seconds=remaining)).strftime("%H:%M:%S")
    postfix = {"cpu": f"{cpu:.0f}%", "mem": f"{rss_mb:.1f}MB"}
    if finish:
        postfix["finish"] = finish
    return postfix


def _run_chain(chain: _ChainSpec, store: EncounterDataStore, slot: int) -> list[Path]:
    """串行执行一条完整重复链：构造点参数 -> 初始化 RNG -> solve -> write。

    每个原子运行由 ``_PointSpec`` 携带实际 seed（显式派生或系统熵），
    求解结束立即由本 worker 写入源文件并释放轨迹，再进入下一个扫描点。
    """
    written: list[Path] = []
    for point in chain.points:
        if point.fail_at:
            raise RuntimeError(
                f"注入失败：repetition={chain.repetition_index} "
                f"scan_index={point.scan_index}"
            )
        if chain.scan_parameter:
            import typing

            value = point.parameter_value
            if typing.get_type_hints(EncounterSettings)[chain.scan_parameter] is int:
                value = int(value)
            else:
                value = float(value)
            point_settings = dataclasses.replace(
                chain.settings, **{chain.scan_parameter: value}
            )
        else:
            point_settings = chain.settings
        sim = EncounterSimulation(
            settings=point_settings, seed=np.random.SeedSequence(point.seed)
        )
        traj = sim.solve(on_step=partial(_emit_progress, slot))
        traj.identity = replace(
            traj.identity,
            submission=chain.submission,
            scan_parameter=chain.scan_parameter,
            scan_value=(
                None
                if point.parameter_value is None
                else float(point.parameter_value)
            ),
            scan_index=point.scan_index,
            repeat_index=chain.repetition_index,
            seed_mode=point.seed_mode,
            seed=point.seed,
        )
        written.append(store.write(traj, point.target))
        _emit_point_done(slot, chain.repetition_index, point.scan_index)
        del traj
    return written


def derive_task_seed(root_seed: int, repetition_index: int, scan_index: int) -> int:
    """显式 seed 的确定性任务派生：SeedSequence([root, rep, idx])。"""
    return int(
        np.random.SeedSequence([root_seed, repetition_index, scan_index])
        .generate_state(1)[0]
    )


@dataclass(frozen=True, slots=True)
class EncounterController:
    """中控：展开提交、预检、调度 worker 并返回成功生成的源文件路径。"""

    store: EncounterDataStore = field(default_factory=EncounterDataStore)

    def run(
        self,
        settings: EncounterSettings,
        run_config: RunConfig,
        scan: ScanSpec | None = None,
        *,
        _fail_at: tuple[int, int] | None = None,
    ) -> list[Path]:
        """执行一次提交：展开重复链 -> 预检 -> 调度 -> 稳定排序返回路径。

        ``_fail_at`` 为 (repetition_index, scan_index) 失败注入点，仅
        测试与诊断使用。任何 worker 失败时停止提交后续链、取消未启动
        future、保留已完成文件并抛出 ``RuntimeError``。
        """
        chains, submission = self._expand(settings, run_config, scan, _fail_at)
        all_targets = [point.target for chain in chains for point in chain.points]
        if not all_targets:
            return []
        # 启动 worker 前一次性预检全部最终路径
        self.store.precheck(all_targets)

        progress = bool(run_config.progress)
        n_slots = min(run_config.workers, len(chains))
        times = settings.times
        n_points = len(all_targets)

        bars: list = []
        total_bar = None
        proc = None
        progress_queue = None
        hw = [0] * n_slots  # 每 slot 当前任务内已计入进度的步数高水位
        if progress:
            import psutil
            from tqdm import tqdm

            proc = psutil.Process()
            for k in range(n_slots):
                bars.append(
                    tqdm(
                        total=len(chains[k].points) * times,
                        desc=f"worker {k + 1}/{n_slots}",
                        unit="step",
                        position=k,
                        leave=True,
                    )
                )
            total_bar = tqdm(
                total=n_points, desc="submission", unit="task", position=n_slots,
                leave=True,
            )
            progress_queue = multiprocessing.Queue(maxsize=_PROGRESS_QUEUE_MAXSIZE)
            k_every = max(1, times // _PROGRESS_EVERY_TARGET)

        executor = ProcessPoolExecutor(
            max_workers=n_slots,
            initializer=_sweep_worker_init if progress else None,
            initargs=(progress_queue, k_every) if progress else None,
        )
        pending: dict = {}  # future -> (chain_index, slot)
        free_slots = list(range(n_slots))  # 条号池：进度条索引，循环复用
        next_chain = 0  # 按 repetition 顺序领取下一条链
        collected: list[tuple[int, list[Path]]] = []  # (chain_index, paths)
        failed: tuple[int, BaseException] | None = None
        try:
            last_completion = time.monotonic()  # 挂死时钟：距上次有任务完成的时刻
            for _ in range(n_slots):
                if next_chain < len(chains):
                    slot = free_slots.pop()
                    pending[
                        executor.submit(_run_chain, chains[next_chain], self.store, slot)
                    ] = (next_chain, slot)
                    next_chain += 1
            while pending:
                done, _ = wait(
                    tuple(pending), timeout=_POLL_S, return_when=FIRST_COMPLETED
                )
                # 先 drain 进度消息：无论本轮有无任务完成都实时消费，worker
                # 条在任务进行中逐帧推进（而非任务完成时一次跳满）。
                if progress:
                    self._drain_progress(progress_queue, bars, hw, times, total_bar, proc)
                if not done:
                    # 轮询空转：任务运行中的正常等待不算挂死，靠挂死时钟兜底
                    if time.monotonic() - last_completion > _WATCHDOG_S:
                        raise RuntimeError(f"{_WATCHDOG_S}s 内无任务完成，疑似 worker 挂死")
                    continue
                last_completion = time.monotonic()
                # 阶段 A：先收集整批完成结果并检测失败——失败链与成功链同时
                # 完成时，必须先确认本批无失败，再领取新链（否则已启动的
                # 新链无法取消，会继续产出文件）
                completed: list[tuple[int, int]] = []
                for fut in done:
                    chain_index, slot = pending.pop(fut)
                    free_slots.append(slot)
                    try:
                        collected.append((chain_index, fut.result()))
                    except BaseException as exc:  # noqa: BLE001
                        failed = (chain_index, exc)
                        break
                    completed.append((chain_index, slot))
                    if progress:
                        total_bar.update(1)
                        total_bar.set_postfix(**_progress_postfix(proc, total_bar.format_dict))
                # 阶段 B：本批全部成功才领取下一条未开始的链
                if failed is None:
                    for _ in completed:
                        if next_chain < len(chains):
                            next_slot = free_slots.pop()
                            pending[
                                executor.submit(
                                    _run_chain, chains[next_chain], self.store, next_slot
                                )
                            ] = (next_chain, next_slot)
                            next_chain += 1
                if failed is not None:
                    break
        finally:
            if progress:
                for bar in bars + [total_bar]:
                    bar.close()
                progress_queue.close()
                progress_queue.cancel_join_thread()
            for fut in pending:
                fut.cancel()
            # 等待正在完成原子提升的写入安全结束
            executor.shutdown(wait=True, cancel_futures=True)

        if failed is not None:
            chain_index, exc = failed
            raise RuntimeError(
                f"worker（链 {chain_index}，共 {len(chains[chain_index].points)} "
                f"个扫描点）失败：{exc}"
            ) from exc

        # 结果按 (repetition_index, scan_index) 稳定排序后返回
        flat: list[tuple[int, int, Path]] = []
        for chain_index, paths in collected:
            chain = chains[chain_index]
            for point, path in zip(chain.points, paths):
                flat.append((point.scan_index, chain.repetition_index, path))
        flat.sort(key=lambda item: (item[1], item[0]))
        return [path for _, _, path in flat]

    def _drain_progress(self, progress_queue, bars, hw, times, total_bar, proc) -> None:
        while not progress_queue.empty():
            try:
                message = progress_queue.get_nowait()
            except std_queue.Empty:
                break
            kind = message[0]
            if kind == "step":
                _, slot, t = message
                delta = _progress_delta(hw[slot], t, times)
                if delta is not None:
                    bars[slot].update(delta[0])
                    hw[slot] = delta[1]
            elif kind == "point":
                _, slot, _rep, _idx = message
                bars[slot].update(times - hw[slot])
                hw[slot] = 0

    def _expand(
        self,
        settings: EncounterSettings,
        run_config: RunConfig,
        scan: ScanSpec | None,
        fail_at: tuple[int, int] | None,
    ) -> tuple[list[_ChainSpec], str]:
        """把提交展开为 loop_times 条链并确定全部目标路径与实际 seed。"""
        submission = self.store.submission_dir_name(run_config.name, settings)
        root = Path(run_config.output_dir) / submission
        scan_parameter = scan.parameter if scan is not None else ""
        if scan is None:
            values: list[float | None] = [None]  # 无扫描：EncounterSettings 当前值
        else:
            values = [float(v) for v in scan.values()]
        points_per_chain = len(values)

        # 实际 seed：显式按 (rep, idx) 派生；无 seed 每个原子运行独立系统熵。
        # 系统熵为 128 位整数，超出 HDF5 属性 int64 范围，截断为 63 位正数
        # 作标识（唯一性足够，仍可凭该标识重建 SeedSequence 追溯该运行）。
        seed_matrix: list[list[int]] = []
        for rep in range(run_config.loop_times):
            row: list[int] = []
            for idx in range(points_per_chain):
                if run_config.seed is not None:
                    row.append(derive_task_seed(run_config.seed, rep, idx))
                else:
                    row.append(int(np.random.SeedSequence().entropy % (1 << 63)))
            seed_matrix.append(row)

        chains: list[_ChainSpec] = []
        for rep in range(run_config.loop_times):
            points: list[_PointSpec] = []
            for idx, value in enumerate(values):
                seed = seed_matrix[rep][idx]
                target = root / hdf5_filename(
                    submission,
                    idx,
                    rep,
                    seed,
                    scan_parameter=scan_parameter,
                    scan_value=value,
                )
                points.append(
                    _PointSpec(
                        scan_index=idx,
                        parameter_value=value,
                        seed=seed,
                        seed_mode=(
                            "explicit" if run_config.seed is not None else "entropy"
                        ),
                        target=target,
                        fail_at=fail_at == (rep, idx),
                    )
                )
            chains.append(
                _ChainSpec(
                    repetition_index=rep,
                    points=points,
                    settings=settings,
                    scan_parameter=scan_parameter,
                    submission=submission,
                )
            )
        return chains, submission


__all__ = ["EncounterController", "derive_task_seed"]
