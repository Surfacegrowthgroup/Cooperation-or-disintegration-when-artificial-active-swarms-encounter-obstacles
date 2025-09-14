# Author:ghb@Ashly
# ：时间：2022/12/17 12:15
import numpy


class Setup:

    def __init__(self):

        self.par_density = 0.1  # 0.1
        self.que_density = 0.01  # 0.01

        self.times = 10000  # 6000
        self.St = 3000
        self.loop_times = 10  # 10

        self.width = 50  # 50
        self.length = 400  # 400

        self.left_length = -100  # 显示距离原点线左侧区域的长度
        self.place_length = 50
        self.white_length = 5
        self.begin_length = self.place_length + self.white_length
        self.end_length = 100

        self.particles = int(self.width * self.place_length * self.par_density)
        self.radius = 1  # 1.0
        self.strength = 0.01  # 0.01
        self.speed = 0.3  # 0.3

        self.que_stren = 0  # 1.0
        self.que_num = int(self.width * self.length * self.que_density)
        self.que_radius = 0.5  # 0.5

        self.pause_time = 0.01

        self.var = self.que_stren  # 存储初始值
        self.var_step = 0.05
        self.var_goal = 1.0
        self.var_list = numpy.arange(self.var, self.var_goal + self.var_step, self.var_step)
        self.var_times = len(self.var_list)
        print(self.var_times*self.loop_times)
