# Author: ghb@Ashly
# 时间: 2024/9/4 15:21
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.collections as clt
import matplotlib.cm as cm
import matplotlib.colors as clr
import networkx
import time
import pandas as pd

from counter_settings import Setup
from tqdm.rich import tqdm

from scipy.stats import gaussian_kde


def timer(func):
    def func_wrapped(*args, **kwargs):
        st = time.perf_counter()
        res = func(*args, **kwargs)
        ed = time.perf_counter()
        print('%s : %.3f s' % (func.__name__, ed-st))
        return res
    return func_wrapped


class Main:

    def __init__(self):
        self.setup = Setup()

        self.pos = [self.setup.place_length, self.setup.width] * np.random.rand(self.setup.particles, 2)
        # self.Angle = np.zeros([1, self.setup.particles])  # 初始方向全都设置为0保证可以快速形成一致向右的集群
        self.Angle = (np.random.rand(1, self.setup.particles) - 0.5) * np.pi
        self.vel = np.vstack([np.cos(self.Angle), np.sin(self.Angle)]).T  # 每行是一个粒子的速度向量

        # 淬火区偏移
        self.que_pos = [self.setup.length, self.setup.width] * np.random.rand(self.setup.que_num, 2) + \
                       [self.setup.begin_length, 0]
        self.que_init = np.random.rand(1, self.setup.que_num) - 0.5
        self.que_noise = self.setup.que_stren * 2 * np.pi * self.que_init
        self.tree_que = scipy.spatial.KDTree(self.que_pos, boxsize=[0, self.setup.width])

        self.fig, self.ax = matplotlib.pyplot.subplots(1, 1)
        self.neigh = None
        self.loind = self.pos[:, 0] < 0
        self.psind = self.pos[:, 0] > (self.setup.begin_length + self.setup.length)
        self.net = None

        self.loOrder = np.zeros([1, self.setup.times])
        self.psOrder = np.zeros([1, self.setup.times])
        self.loCluster = np.zeros([1, self.setup.times])
        self.psCluster = np.zeros([1, self.setup.times])
        self.Ratio = np.zeros([1, self.setup.times])

    def image_init(self):
        matplotlib.pyplot.ion()  # 检测语句
        self.ax.axis('square')
        self.ax.set_xlim((self.setup.left_length, self.setup.begin_length + self.setup.length + self.setup.end_length))
        self.ax.get_xaxis().set_visible(False);  self.ax.get_yaxis().set_visible(False)
        self.ax.set_ylim((0, self.setup.width))

    def vel_show(self):
        # self.ax.set_xlim((self.setup.left_length, self.setup.begin_length + self.setup.length + self.setup.end_length))
        # self.ax.set_ylim((0, self.setup.width))
        self.ax.set_xlim((0, 400))
        # for i, v in enumerate(self.que_pos):  # 淬火点上色  为使效果明显将原有淬火噪声数组数值翻倍
        #     self.ax.add_patch(matplotlib.patches.Circle(v, radius=self.setup.que_radius,
        #                                                 color=[1 - 2 * abs(self.que_init[0][i])] + [1, 1]))
        cmap = cm.get_cmap('gray')
        norm = clr.Normalize(vmin=0, vmax=1)
        colors = cmap(norm(np.abs(self.que_init[0]) * 2))
        circles = clt.CircleCollection(
            sizes=[2 * self.setup.que_radius * np.pi**2] * self.setup.particles,  # 直径 = 2 * 半径
            offsets=self.que_pos,  # 圆心坐标 (N,2)
            facecolors=colors,  # 颜色数组 (N,4 RGBA)
            edgecolors='none',  # 无边框
            transOffset=self.ax.transData,  # 坐标系与数据一致
            zorder=1  # 图层顺序
        )
        self.ax.add_collection(circles)
        plt.show()
        self.ax.quiver(self.pos[:, 0], self.pos[:, 1], np.cos(self.Angle), np.sin(self.Angle), color='#0072BD',
                       scale=30, width=2e-3)
        plt.pause(self.setup.pause_time);  plt.cla()

    def rho_show(self):
        nbins = 150
        k = gaussian_kde(self.pos.T)
        xi, yi = np.mgrid[self.setup.left_length:self.setup.begin_length + self.setup.length + self.setup.end_length:nbins * 1j,
              0:self.setup.width:nbins * 1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        self.ax.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto', vmin=0, vmax=3e-4)
        for i, v in enumerate(self.que_pos):  # 淬火点上色  为使效果明显将原有淬火噪声数组数值翻倍
            self.ax.add_patch(matplotlib.patches.Circle(v, radius=self.setup.que_radius,
                                                        color=[1 - 2 * abs(self.que_init[0][i])] + [1, 1]))
        self.ax.set_ylim((0, self.setup.width))
        plt.show();  plt.pause(self.setup.pause_time)

    def cal_order(self, ind):
        locvel = self.vel[ind]
        return np.linalg.norm(np.sum(locvel, axis=0) / len(locvel)) if len(locvel) != 0 else 0

    def cal_ratio(self):
        return len(np.argwhere(self.pos[:, 0] > self.setup.begin_length + self.setup.length)) / self.setup.particles

    def cal_cluster(self, ind):
        c = networkx.clustering(self.net, nodes=np.arange(self.setup.particles)[ind])
        return sum(c.values()) / len(c) if len(c) != 0 else 0

    def prepare_network(self):
        self.net = networkx.Graph()
        edges = [(i, neighbor) for i, neighbors in enumerate(self.neigh) for neighbor in neighbors if i != neighbor]   #
        self.net.add_edges_from(edges)

    def calcu_angle(self):
        tree = scipy.spatial.KDTree(self.pos, boxsize=[0, self.setup.width])
        self.neigh = tree.query_ball_point(self.pos, r=self.setup.radius, workers=-1)
        angles_flat = self.Angle[0]
        sin_values = np.sin(angles_flat);  cos_values = np.cos(angles_flat)
        num_neighbors = [len(nlist) for nlist in self.neigh]

        all_neighbors = np.concatenate(self.neigh)
        indptr = np.zeros(self.setup.particles + 1, dtype=int)
        indptr[1:] = np.cumsum(num_neighbors)

        sum_sin = np.add.reduceat(sin_values[all_neighbors], indptr[:-1])
        sum_cos = np.add.reduceat(cos_values[all_neighbors], indptr[:-1])

        counts = np.array(num_neighbors, dtype=np.float32)  # 计算平均角度
        avg_sin = sum_sin / counts;  avg_cos = sum_cos / counts
        new_angles = np.arctan2(avg_sin, avg_cos)

        return new_angles.reshape(1, -1)

    def check_quench(self):
        influ_angle = np.zeros([1, self.setup.particles])
        que_neigh = self.tree_que.query_ball_point(self.pos, r=self.setup.que_radius, workers=-1)
        for i in range(self.setup.particles):
            ls = que_neigh[i]  # 第i个粒子的淬火邻居
            if list(ls):
                influ_angle[0][i] = self.que_noise[0][ls].mean()

        return influ_angle

    def update_position(self):
        self.Angle = self.calcu_angle() + \
                     self.setup.strength * np.random.randn(1, self.setup.particles) + self.check_quench()
        self.vel = np.vstack([np.cos(self.Angle), np.sin(self.Angle)]).T  # 每行是一个粒子的速度向量

        self.prepare_network()  # 创建当前时刻的图

        self.pos = self.pos + self.setup.speed * self.vel
        self.pos[:, 1] = self.pos[:, 1] % self.setup.width  # 横向非周期性边界，所以仅在宽度方向上进行周期性边界处理

        self.loind = self.pos[:, 0] < 0
        self.psind = self.pos[:, 0] > (self.setup.begin_length + self.setup.length)
        print(f'lef:{np.count_nonzero(self.loind)} right:{np.count_nonzero(self.psind)}')

    def save_data(self):
        filename = f'Finiete Time Efftct'
        df = pd.DataFrame({
            'ratio': self.Ratio[0],
            'V1': self.loOrder[0],
            'V2': self.psOrder[0],
            'C1': self.loCluster[0],
            'C2': self.psCluster[0],
        })  # 序参量和辐角时间序列
        df.to_excel(filename+f'.xlsx', index=False)

    def run(self):
        # self.image_init()
        for t in tqdm(range(self.setup.times)):
            # self.vel_show()
            self.update_position()

            self.loOrder[0][t] = self.cal_order(self.loind);  self.psOrder[0][t] = self.cal_order(self.psind)
            self.loCluster[0][t] = self.cal_cluster(self.loind);  self.psCluster[0][t] = self.cal_cluster(self.psind)
            self.Ratio[0][t] = self.cal_ratio()

        fitt = np.arange(self.setup.times)
        plt.plot(fitt, self.loOrder[0], '-o', label='$\psi_o$');  plt.plot(fitt, self.psOrder[0], '-o', label='$\psi_e$')
        plt.plot(fitt, self.loCluster[0], '-o', label='$C_o$');  plt.plot(fitt, self.psCluster[0], '-o', label='$C_e$')
        plt.plot(fitt, self.Ratio[0], '-o', label='$R$')
        plt.legend(loc=0)

        self.save_data()

        plt.show()


if __name__ == "__main__":
    ai = Main()
    ai.run()
