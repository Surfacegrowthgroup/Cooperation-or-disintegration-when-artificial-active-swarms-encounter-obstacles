from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def _x_cell(x: float, x0: float, cell_size: float, ncx: int) -> int:
    # 网格数学上覆盖 [min_x, max_x]，粒子必属 [0, ncx-1]；浮点误差可能使
    # 边界粒子算出 -1/ncx，clamp 到边界格是正确归属，避免下游越界访问
    cell = int((x - x0) / cell_size)
    if cell < 0:
        return 0
    if cell >= ncx:
        return ncx - 1
    return cell


@njit(cache=True, fastmath=True)
def _y_cell(y: float, width: float, cell_size: float, ncy: int) -> int:
    y_mod = y % width
    cell = int(y_mod / cell_size)
    if cell >= ncy:
        cell = ncy - 1
    return cell


@njit(cache=True, fastmath=True)
def _build_cell_index(
    pos: np.ndarray,
    x0: float,
    cell_x: float,
    cell_y: float,
    ncx: int,
    ncy: int,
    width: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将粒子散列到网格，返回 (cell_counts, cell_starts, sorted_pos, original_indices)。

    sorted_pos 的 y 坐标已取模 width；original_indices 记录每个排序项的原数组下标，
    供调用方按 payload 回填。纯逻辑重构，与各调用方原有散列行为逐位一致。
    """
    n_particles = pos.shape[0]
    cell_counts = np.zeros((ncx, ncy), dtype=np.int32)
    for i in range(n_particles):
        cx = _x_cell(pos[i, 0], x0, cell_x, ncx)
        cy = _y_cell(pos[i, 1], width, cell_y, ncy)
        if 0 <= cx < ncx:
            cell_counts[cx, cy] += 1

    cell_starts = np.empty((ncx, ncy), dtype=np.int32)
    cursor = 0
    for cx in range(ncx):
        for cy in range(ncy):
            cell_starts[cx, cy] = cursor
            cursor += cell_counts[cx, cy]

    sorted_pos = np.empty((n_particles, 2), dtype=np.float64)
    original_indices = np.empty(n_particles, dtype=np.int32)
    offsets = np.zeros((ncx, ncy), dtype=np.int32)

    for i in range(n_particles):
        cx = _x_cell(pos[i, 0], x0, cell_x, ncx)
        cy = _y_cell(pos[i, 1], width, cell_y, ncy)
        idx = cell_starts[cx, cy] + offsets[cx, cy]
        sorted_pos[idx, 0] = pos[i, 0]
        sorted_pos[idx, 1] = pos[i, 1] % width
        original_indices[idx] = i
        offsets[cx, cy] += 1

    return cell_counts, cell_starts, sorted_pos, original_indices


@njit(cache=True, fastmath=True)
def compute_vicsek_angles_open_x_periodic_y(
    pos: np.ndarray,
    angles: np.ndarray,
    width: float,
    radius: float,
) -> np.ndarray:
    """邻居方向角（含自身）的圆平均，open x + periodic y 边界。

    对齐方向定义为邻居角度的圆平均（atan2 分量向量和），邻域
    始终包含粒子自身，因此孤立粒子保留原方向，不会被重置。
    """
    n_particles = pos.shape[0]
    out = np.empty((1, n_particles), dtype=np.float64)
    if n_particles == 0:
        return out

    min_x = pos[0, 0]
    max_x = pos[0, 0]
    for i in range(1, n_particles):
        if pos[i, 0] < min_x:
            min_x = pos[i, 0]
        if pos[i, 0] > max_x:
            max_x = pos[i, 0]

    x_span = max_x - min_x + 1.0e-12
    ncx = int(x_span / radius)
    if ncx < 1:
        ncx = 1
    ncy = int(width / radius)
    if ncy < 1:
        ncy = 1

    x0 = min_x
    cell_x = x_span / ncx
    cell_y = width / ncy

    cell_counts, cell_starts, sorted_pos, original_indices = _build_cell_index(
        pos, x0, cell_x, cell_y, ncx, ncy, width
    )

    sin_angles = np.sin(angles[0])
    cos_angles = np.cos(angles[0])

    sorted_sin = np.empty(n_particles, dtype=np.float64)
    sorted_cos = np.empty(n_particles, dtype=np.float64)
    for i in range(n_particles):
        sorted_sin[i] = sin_angles[original_indices[i]]
        sorted_cos[i] = cos_angles[original_indices[i]]

    radius_sq = radius * radius
    half_width = 0.5 * width

    for i in range(n_particles):
        xi = sorted_pos[i, 0]
        yi = sorted_pos[i, 1]
        cx = _x_cell(xi, x0, cell_x, ncx)
        cy = _y_cell(yi, width, cell_y, ncy)

        sum_sin = 0.0
        sum_cos = 0.0

        for dx in range(-1, 2):
            nx = cx + dx
            if nx < 0 or nx >= ncx:
                continue
            for dy in range(-1, 2):
                ny = (cy + dy) % ncy
                start = cell_starts[nx, ny]
                end = start + cell_counts[nx, ny]
                for j in range(start, end):
                    dx_ij = sorted_pos[j, 0] - xi
                    dy_ij = sorted_pos[j, 1] - yi
                    if dy_ij > half_width:
                        dy_ij -= width
                    elif dy_ij < -half_width:
                        dy_ij += width

                    if dx_ij * dx_ij + dy_ij * dy_ij <= radius_sq:
                        sum_sin += sorted_sin[j]
                        sum_cos += sorted_cos[j]

        out[0, original_indices[i]] = np.arctan2(sum_sin, sum_cos)

    return out


@njit(cache=True, fastmath=True)
def build_obstacle_grid(
    obstacle_pos: np.ndarray,
    obstacle_angles: np.ndarray,
    width: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float, int, int]:
    """构建障碍物哈希网格索引（障碍物在 reset 后固定，可跨步缓存复用）。

    返回 (cell_starts, cell_counts, sorted_pos, sorted_angle, x0, cell_x, cell_y, ncx, ncy)，
    供 :func:`query_obstacle_grid` 成对使用；``compute_quenched_influence_*``
    是一次性便捷包装（内部即建网格 + 查询）。
    """
    n_obstacles = obstacle_pos.shape[0]
    min_x = obstacle_pos[0, 0]
    max_x = obstacle_pos[0, 0]
    for i in range(1, n_obstacles):
        if obstacle_pos[i, 0] < min_x:
            min_x = obstacle_pos[i, 0]
        if obstacle_pos[i, 0] > max_x:
            max_x = obstacle_pos[i, 0]

    x0 = min_x - radius - 1.0e-12
    x_span = max_x - min_x + 2.0 * radius + 2.0e-12
    ncx = int(x_span / radius)
    if ncx < 1:
        ncx = 1
    ncy = int(width / radius)
    if ncy < 1:
        ncy = 1

    cell_x = x_span / ncx
    cell_y = width / ncy

    cell_counts, cell_starts, sorted_pos, original_indices = _build_cell_index(
        obstacle_pos, x0, cell_x, cell_y, ncx, ncy, width
    )

    sorted_angle = np.empty(n_obstacles, dtype=np.float64)
    for i in range(n_obstacles):
        sorted_angle[i] = obstacle_angles[0, original_indices[i]]

    return cell_starts, cell_counts, sorted_pos, sorted_angle, x0, cell_x, cell_y, ncx, ncy


@njit(cache=True, fastmath=True)
def query_obstacle_grid(
    pos: np.ndarray,
    cell_starts: np.ndarray,
    cell_counts: np.ndarray,
    sorted_pos: np.ndarray,
    sorted_angle: np.ndarray,
    x0: float,
    cell_x: float,
    cell_y: float,
    ncx: int,
    ncy: int,
    width: float,
    radius: float,
) -> np.ndarray:
    """用已构建的障碍物网格查询每个粒子的淬火影响（邻近障碍角度的圆平均）。

    返回纯几何的圆平均，不含强度 H；H 按论文 Eq.(1c) 由调用方施加。
    与 :func:`build_obstacle_grid` 成对使用：障碍固定时构建一次、跨步缓存复用。
    """
    n_particles = pos.shape[0]
    out = np.zeros((1, n_particles), dtype=np.float64)
    radius_sq = radius * radius
    half_width = 0.5 * width

    for i in range(n_particles):
        cx = _x_cell(pos[i, 0], x0, cell_x, ncx)
        # _x_cell 已把越界坐标 clamp 到边界格，无需再判 cx 越界
        cy = _y_cell(pos[i, 1], width, cell_y, ncy)
        xi = pos[i, 0]
        yi = pos[i, 1] % width

        count = 0
        sum_sin = 0.0
        sum_cos = 0.0
        for dx in range(-1, 2):
            nx = cx + dx
            if nx < 0 or nx >= ncx:
                continue
            for dy in range(-1, 2):
                ny = (cy + dy) % ncy
                start = cell_starts[nx, ny]
                end = start + cell_counts[nx, ny]
                for j in range(start, end):
                    dx_ij = sorted_pos[j, 0] - xi
                    dy_ij = sorted_pos[j, 1] - yi
                    if dy_ij > half_width:
                        dy_ij -= width
                    elif dy_ij < -half_width:
                        dy_ij += width

                    if dx_ij * dx_ij + dy_ij * dy_ij <= radius_sq:
                        sum_sin += np.sin(sorted_angle[j])
                        sum_cos += np.cos(sorted_angle[j])
                        count += 1

        if count > 0:
            out[0, i] = np.arctan2(sum_sin, sum_cos)

    return out


@njit(cache=True, fastmath=True)
def compute_quenched_influence_open_x_periodic_y(
    pos: np.ndarray,
    obstacle_pos: np.ndarray,
    obstacle_angles: np.ndarray,
    width: float,
    radius: float,
) -> np.ndarray:
    """每个粒子受到的障碍固定角度增量（圆平均），open x + periodic y。

    障碍以固定角度增量（局部偏转）表示，q 取影响半径内障碍角度的
    圆平均（atan2 分量向量和）；无邻近障碍时为零（孤立粒子不受偏转）。
    返回纯圆平均，不含强度 H；H 按论文 Eq.(1c) 由调用方施加。
    """
    n_particles = pos.shape[0]
    n_obstacles = obstacle_pos.shape[0]
    out = np.zeros((1, n_particles), dtype=np.float64)
    if n_particles == 0 or n_obstacles == 0:
        return out

    cell_starts, cell_counts, sorted_pos, sorted_angle, x0, cell_x, cell_y, ncx, ncy = (
        build_obstacle_grid(obstacle_pos, obstacle_angles, width, radius)
    )
    return query_obstacle_grid(
        pos, cell_starts, cell_counts, sorted_pos, sorted_angle,
        x0, cell_x, cell_y, ncx, ncy, width, radius,
    )


@njit(cache=True, fastmath=True)
def compute_cluster_coefficients_open_x_periodic_y(
    pos: np.ndarray,
    width: float,
    radius: float,
    mask: np.ndarray,
) -> np.ndarray:
    """局部聚类系数（Watts-Strogatz），open x + periodic y 边界。

    语义与 networkx.clustering 一致：无自环简单图；邻居为距离 <= radius
    的点（含等号，y 方向取 min-image 周期修正）；mask 外节点输出 0.0，
    但 mask 内节点的邻居包含 mask 外粒子；度数 < 2 时系数为 0。
    返回 (1, N) float64，与引擎其他函数返回形状一致。
    """
    n_particles = pos.shape[0]
    out = np.zeros((1, n_particles), dtype=np.float64)
    if n_particles == 0:
        return out

    min_x = pos[0, 0]
    max_x = pos[0, 0]
    for i in range(1, n_particles):
        if pos[i, 0] < min_x:
            min_x = pos[i, 0]
        if pos[i, 0] > max_x:
            max_x = pos[i, 0]

    x_span = max_x - min_x + 1.0e-12
    ncx = int(x_span / radius)
    if ncx < 1:
        ncx = 1
    ncy = int(width / radius)
    if ncy < 1:
        ncy = 1

    x0 = min_x
    cell_x = x_span / ncx
    cell_y = width / ncy

    cell_counts, cell_starts, sorted_pos, original_indices = _build_cell_index(
        pos, x0, cell_x, cell_y, ncx, ncy, width
    )

    radius_sq = radius * radius
    half_width = 0.5 * width
    nbrs = np.empty(n_particles, dtype=np.int32)

    for i in range(n_particles):
        if not mask[original_indices[i]]:
            continue

        xi = sorted_pos[i, 0]
        yi = sorted_pos[i, 1]
        cx = _x_cell(xi, x0, cell_x, ncx)
        cy = _y_cell(yi, width, cell_y, ncy)

        deg = 0
        for dx in range(-1, 2):
            nx = cx + dx
            if nx < 0 or nx >= ncx:
                continue
            for dy in range(-1, 2):
                ny = (cy + dy) % ncy
                start = cell_starts[nx, ny]
                end = start + cell_counts[nx, ny]
                for j in range(start, end):
                    if j == i:
                        continue  # 无自环
                    dx_ij = sorted_pos[j, 0] - xi
                    dy_ij = sorted_pos[j, 1] - yi
                    if dy_ij > half_width:
                        dy_ij -= width
                    elif dy_ij < -half_width:
                        dy_ij += width
                    if dx_ij * dx_ij + dy_ij * dy_ij <= radius_sq:
                        nbrs[deg] = j
                        deg += 1

        if deg < 2:
            continue

        # 三角形数：邻居对 (a, b) 中互为邻居的边数
        triangles = 0
        for a in range(deg):
            na = nbrs[a]
            xa = sorted_pos[na, 0]
            ya = sorted_pos[na, 1]
            for b in range(a + 1, deg):
                nb = nbrs[b]
                dx_ab = sorted_pos[nb, 0] - xa
                dy_ab = sorted_pos[nb, 1] - ya
                if dy_ab > half_width:
                    dy_ab -= width
                elif dy_ab < -half_width:
                    dy_ab += width
                if dx_ab * dx_ab + dy_ab * dy_ab <= radius_sq:
                    triangles += 1

        out[0, original_indices[i]] = 2.0 * triangles / (deg * (deg - 1))

    return out
