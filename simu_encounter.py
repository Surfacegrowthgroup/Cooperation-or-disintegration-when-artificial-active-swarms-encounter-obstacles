# Author: ghb@Ashly
# 时间: 2024/9/4 15:21
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.colors

from counter_settings import Setup
from tqdm.rich import tqdm


class Main:

    def __init__(self):
        self.setup = Setup()

        self.pos = [self.setup.place_length, self.setup.width] * np.random.rand(self.setup.particles, 2)
        self.Angle = (np.random.rand(1, self.setup.particles) - 0.5) * np.pi
        self.vel = np.vstack([np.cos(self.Angle), np.sin(self.Angle)]).T

        self.que_pos = [self.setup.length, self.setup.width] * np.random.rand(self.setup.que_num, 2) + \
                       [self.setup.begin_length, 0]
        self.que_init = np.random.rand(1, self.setup.que_num) - 0.5
        self.que_noise = self.setup.que_stren * 2 * np.pi * self.que_init
        self.tree_que = scipy.spatial.KDTree(self.que_pos, boxsize=[0, self.setup.width])

        self.fig, self.ax = plt.subplots(1, 1)
        self.neigh = None

    def image_init(self):
        plt.ion()
        self.ax.axis('square')
        self.ax.set_xlim((self.setup.left_length, self.setup.begin_length + self.setup.length + self.setup.end_length))
        self.ax.get_xaxis().set_visible(False);  self.ax.get_yaxis().set_visible(False)
        self.ax.set_ylim((0, self.setup.width))

    def vel_show(self):
        self.ax.set_xlim((self.setup.left_length, self.setup.begin_length + self.setup.length + self.setup.end_length))
        self.ax.set_ylim((0, self.setup.width))
        for i, v in enumerate(self.que_pos):
            self.ax.add_patch(matplotlib.patches.Circle(v, radius=self.setup.que_radius,
                                                        color=[1 - 2 * abs(self.que_init[0][i])] + [1, 1]))
        plt.show()
        self.ax.quiver(self.pos[:, 0], self.pos[:, 1], np.cos(self.Angle), np.sin(self.Angle), color='#0072BD', scale=30)
        plt.pause(self.setup.pause_time);  plt.cla()

    def calcu_angle(self):
        tree = scipy.spatial.KDTree(self.pos, boxsize=[0, self.setup.width])
        neigh = tree.query_ball_point(self.pos, r=self.setup.radius, workers=-1)
        angles_flat = self.Angle[0]
        sin_values = np.sin(angles_flat);  cos_values = np.cos(angles_flat)
        num_neighbors = [len(nlist) for nlist in neigh]

        all_neighbors = np.concatenate(neigh)
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

        self.pos = self.pos + self.setup.speed * self.vel
        self.pos[:, 1] = self.pos[:, 1] % self.setup.width  # 横向非周期性边界，所以仅在宽度方向上进行周期性边界处理

    def run(self):
        self.image_init()
        for t in tqdm(range(self.setup.times)):
            self.vel_show()
            self.update_position()


if __name__ == "__main__":
    ai = Main()
    ai.run()
