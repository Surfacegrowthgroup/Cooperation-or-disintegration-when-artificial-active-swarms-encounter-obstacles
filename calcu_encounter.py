# Author: ghb@Ashely
# 时间: 2023/3/17 11:28
import numpy as np
import scipy.spatial
import matplotlib.pyplot as plt
import multiprocessing
import networkx
from tqdm.rich import tqdm

from counter_settings import Setup


class Main:

    def __init__(self):
        self.setup = Setup()

        self.pos = [self.setup.place_length, self.setup.width] * np.random.rand(self.setup.particles, 2)
        self.Angle = (np.random.rand(1, self.setup.particles) - 0.5) * np.pi
        self.vel = np.vstack([np.cos(self.Angle), np.sin(self.Angle)]).T  # 每行是一个粒子的速度向量

        self.que_pos = [self.setup.length, self.setup.width] * np.random.rand(self.setup.que_num, 2) + \
                       [self.setup.begin_length, 0]
        self.que_noise = self.setup.que_stren * 2 * np.pi * (np.random.rand(1, self.setup.que_num) - 0.5)
        self.tree_que = scipy.spatial.KDTree(self.que_pos, boxsize=[0, self.setup.width])

        self.bar = None
        self.neigh = None  # 单一时刻全局粒子的邻居列表
        self.net = None
        self.indL = self.pos[:, 0] < self.setup.begin_length
        self.indR = self.pos[:, 0] > (self.setup.begin_length + self.setup.length)

        self.OrderL = np.zeros([1, self.setup.times])
        self.OrderR = np.zeros([1, self.setup.times])
        self.ClusterL = np.zeros([1, self.setup.times])
        self.ClusterR = np.zeros([1, self.setup.times])

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
        edges = [(i, neighbor) for i, neighbors in enumerate(self.neigh) for neighbor in neighbors if i != neighbor]
        self.net.add_edges_from(edges)

    def calcu_angle(self):
        tree = scipy.spatial.KDTree(self.pos, boxsize=[0, self.setup.width])
        self.neigh = tree.query_ball_point(self.pos, r=self.setup.radius, workers=-1)
        angles_flat = self.Angle[0]  # 准备工作，将角度数组展平为1D，构建邻居数量数组，并预计算所有sin和cos值
        sin_values = np.sin(angles_flat);  cos_values = np.cos(angles_flat)
        num_neighbors = [len(nlist) for nlist in self.neigh]

        all_neighbors = np.concatenate(self.neigh)  # 合并总邻居数组
        indptr = np.zeros(self.setup.particles + 1, dtype=int)  # 分隔N个位置的索引数组应当包含N+1个数字
        indptr[1:] = np.cumsum(num_neighbors)  # 构建分段索引

        sum_sin = np.add.reduceat(sin_values[all_neighbors], indptr[:-1])  # 计算每个粒子邻居的sin和cos总和
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

        self.indL = self.pos[:, 0] < self.setup.begin_length
        self.indR = self.pos[:, 0] > (self.setup.begin_length + self.setup.length)

    def reset_attribute(self):
        self.setup.particles = int(self.setup.width * self.setup.place_length * self.setup.par_density)
        self.setup.que_num = int(self.setup.width * self.setup.length * self.setup.que_density)

        self.pos = [self.setup.place_length, self.setup.width] * np.random.rand(self.setup.particles, 2)
        self.Angle = (np.random.rand(1, self.setup.particles) - 0.5) * np.pi
        self.vel = np.vstack([np.cos(self.Angle), np.sin(self.Angle)]).T  # 每行是一个粒子的速度向量

        self.que_pos = [self.setup.length, self.setup.width] * np.random.rand(self.setup.que_num, 2) + \
                       [self.setup.begin_length, 0]
        self.que_noise = self.setup.que_stren * 2 * np.pi * (np.random.rand(1, self.setup.que_num) - 0.5)
        self.tree_que = scipy.spatial.KDTree(self.que_pos, boxsize=[0, self.setup.width])

        self.indL = self.pos[:, 0] < self.setup.begin_length
        self.indR = self.pos[:, 0] > (self.setup.begin_length + self.setup.length)

    def run(self, coret):
        tmp_ratio = np.zeros(self.setup.var_times)
        tmp_OrderL = np.zeros(self.setup.var_times);  tmp_OrderR = np.zeros(self.setup.var_times)
        tmp_ClusterL = np.zeros(self.setup.var_times);  tmp_ClusterR = np.zeros(self.setup.var_times)
        res = dict({})
        self.bar = tqdm(total=self.setup.var_times * self.setup.loop_times, desc=f'current time:{coret}')

        for loop in range(self.setup.loop_times):
            for var in range(self.setup.var_times):
                self.reset_attribute()

                for t in range(self.setup.times):
                    self.update_position()  # 迭代下一步位置

                    self.OrderL[0][t] = self.cal_order(self.indL);  self.OrderR[0][t] = self.cal_order(self.indR)
                    self.ClusterL[0][t] = self.cal_cluster(self.indL);  self.ClusterR[0][t] = self.cal_cluster(self.indR)

                tmp_ratio[var] += self.cal_ratio()
                tmp_OrderL[var] += self.OrderL[0][self.setup.St:].mean()
                tmp_OrderR[var] += self.OrderR[0][self.setup.St:].mean()
                tmp_ClusterL[var] += self.ClusterL[0][self.setup.St:].mean()
                tmp_ClusterR[var] += self.ClusterR[0][self.setup.St:].mean()

                self.setup.width += self.setup.var_step  # 变量变化循环时要改变变量并之后更新信息
                self.bar.update(1)

            self.setup.width = self.setup.var  # 重复循环时要将变量大小重置

        self.bar.close()

        res['ratio'] = tmp_ratio / self.setup.loop_times
        res['OrderL'] = tmp_OrderL / self.setup.loop_times;  res['OrderR'] = tmp_OrderR / self.setup.loop_times
        res['ClusterL'] = tmp_ClusterL / self.setup.loop_times;  res['ClusterR'] = tmp_ClusterR / self.setup.loop_times

        return res


if __name__ == "__main__":
    ai = Main()
    # ai.run()

    core_num = 5  # 5
    loop_t = 4
    pool = multiprocessing.Pool(core_num)
    data = [pool.apply_async(func=ai.run, args=(i,)) for i in range(core_num * loop_t)]
    result = {
        'ratio': 0,
        'OrderL': 0,  'OrderR': 0,
        'ClusterL': 0,  'ClusterR': 0,
    }
    for d in data:
        result['ratio'] += d.get()['ratio']
        result['OrderL'] += d.get()['OrderL'];  result['OrderR'] += d.get()['OrderR']
        result['ClusterL'] += d.get()['ClusterL'];  result['ClusterR'] += d.get()['ClusterR']
    result['ratio'] /= (core_num * loop_t)
    result['OrderL'] /= (core_num * loop_t);  result['OrderR'] /= (core_num * loop_t)
    result['ClusterL'] /= (core_num * loop_t);  result['ClusterR'] /= (core_num * loop_t)
    plt.plot(result['ratio'], 'o', label='$R$')
    plt.plot(result['OrderL'], 'o', label=r'$\psi_o$');  plt.plot(result['OrderR'], 'o', label=r'$\psi_e$')
    plt.plot(result['ClusterL'], 'o', label='$C_o$');  plt.plot(result['ClusterR'], 'o', label='$C_e$')
    plt.legend(loc=0);  plt.show()
    print(result)
