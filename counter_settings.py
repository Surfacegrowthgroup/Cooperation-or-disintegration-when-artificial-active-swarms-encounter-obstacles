# Author:ghb@Ashly
# ：时间：2022/12/17 12:15
import numpy


class Setup:

    def __init__(self):

        self.par_density = 0.1
        self.que_density = 0.01

        self.times = 6000
        self.St = 3000
        self.loop_times = 10

        self.width = 100
        self.length = 0

        self.left_length = -100
        self.place_length = 50
        self.white_length = 5
        self.begin_length = self.place_length + self.white_length
        self.end_length = 100

        self.particles = int(self.width * self.place_length * self.par_density)
        self.radius = 1
        self.strength = 0.01
        self.speed = 0.3

        self.que_stren = 1.0
        self.que_num = int(self.width * self.length * self.que_density)
        self.que_radius = 0.5

        self.pause_time = 0.01

        self.var = self.length
        self.var_step = 50
        self.var_goal = 1200
        self.var_list = numpy.arange(self.var, self.var_goal + self.var_step, self.var_step)
        self.var_times = len(self.var_list)
