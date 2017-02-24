import numpy as np
from data.constants import *
from termcolor import colored

class Env:
    maze = []
    def __init__(self):
        f = open('env/Map_v01')
        self.maze = [list(l.strip('\n')) for l in f.readlines()]
    def show(self):
        for i in range(np.shape(self.maze)[0]):
            for j in range(np.shape(self.maze)[1]):
                k = self.get_block((i, j))[0]
                print(colored(self.maze[i][j], COLORS[k % len(COLORS)]), end = '')
            print()
    def update(self, cell, val):
        self.maze[cell[0]][cell[1]] = val
    def get_block(self, cell):
        i, j = cell
        left_bound = j // BLOCK_SIZE[1]
        right_bound = left_bound + BLOCK_SIZE[1]
        up_bound = i // BLOCK_SIZE[0]
        down_bound = up_bound + BLOCK_SIZE[0]
        np_maze = np.array(self.maze)
        block = np_maze[up_bound:down_bound, left_bound:right_bound]
        np_block = [[0 if j in ['0', 'A', 'S', 'F'] else 1 for j in i] for i in block]
        return left_bound + up_bound*BLOCK_SIZE[0], np.array(np_block)
    def step(self, cell, action):
        #Return values: new_cell, reward, is_done, is_block_changed
        i, j = np.array(cell) + np.array(action)
        if i < 0 or j < 0 or i >= MAZE_SIZE[0] or j >= MAZE_SIZE[1]:
            return cell, None, None, None
        if self.maze[i][j] == WALL:
            return cell, WALL_PUN, False, None
        if self.maze[i][j] == HOLE:
            return (i, j), HOLE_PUN, True, None
        if self.maze[i][j] == FIN:
            return (i, j), FIN_PUN, True, None
        self.update(cell, '0')
        self.update((i, j), 'A')
        self.show()
        is_block_changed = bool(self.get_block(cell)[0]-self.get_block((i, j))[0])
        if is_block_changed:
            print ("BLOCK CHANGED ", self.get_block((i, j))[0])
        return (i, j), 0, False, is_block_changed
class THSOM:
    sm = np.zeros((0, 0))
    tm = np.zeros((0, 0))
    neurons_num = 0
    def __init__(self, neurons_num, dim):
        self.neurons_num = neurons_num
        self.sm = np.random.rand(dim, neurons_num)
        self.tm = np.random.rand(neurons_num, neurons_num)
    def get_bmu(self, vec):
        mn = 1e9
        bmu = 0
        for i in range(self.neurons_num):
            dist = np.linalg.norm(self.sm[:,i] - vec)
            if (dist < mn):
               bmu = i
        return self.sm[:,bmu]
    def update_sm_weights(self, bmu, t, vec):
        rad = R0 * np.exp(-np.sqrt(t) / R1)
        for i in range(self.neurons_num):
            dist = np.linalg.norm(bmu - self.sm[:,i])
            if dist < rad:
                print ("SPATIAL VECTOR ", i)
                SLR = S0 * np.exp(-dist*dist / S1)
                TLR = T0 * np.exp(-t / T1)
                DIST = vec - self.sm[:,i]
                self.sm[:,i] += SLR * TLR * DIST
                print ("SLR = ", SLR, "TLR = ", TLR, "DIST = ", DIST)