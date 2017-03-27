import numpy as np
from data.constants import *
from termcolor import colored

class Env:
    maze = []
    def __init__(self):
        f = open(FILE_PATH)
        self.maze = [list(l.strip('\n')) for l in f.readlines()]
    #---------------------------------------------------------------------------------------------------#
    # Function to display process of agent's movement
    def show(self, position):
        left, right, up, down = self.get_block(position)
        for i in range(np.shape(self.maze)[0]):
            for j in range(np.shape(self.maze)[1]):
                if j >= left and j <= right and i >= up and i <= down:
                     print(colored(self.maze[i][j], 'red'), end='')
                else:
                     print(colored(self.maze[i][j], 'white'), end ='')
            print()
    #---------------------------------------------------------------------------------------------------#
    #Update cell with the value for displaying
    def update(self, cell, val):
        self.maze[cell[0]][cell[1]] = val
    #---------------------------------------------------------------------------------------------------#
    # Making a step function. It returns all neccessary data which is needed to define strategy
    # Return values: new_cell, reward, is_done
    def step(self, position, action):
        i, j = np.array(position) + np.array(action)
        if i < 0 or j < 0 or i >= MAZE_SIZE[0] or j >= MAZE_SIZE[1]:
            self.update(position, 'A')
            self.show(position)
            return position, None, None
        if self.maze[i][j] == WALL:
            return position, WALL_PUN, False
        if self.maze[i][j] == HOLE:
            return (i, j), HOLE_PUN, True
        if self.maze[i][j] == FIN:
            return (i, j), FIN_PUN, True
        self.update(position, '0')
        self.update((i, j), 'A')
        self.show((i, j))
        return (i, j), 0, False
    # ---------------------------------------------------------------------------------------------------#
    # Defining 4x4 Block were we are now
    def get_block(self, position):
        i, j = position
        left = j - BLOCK_SIZE[1]/2
        right = j + BLOCK_SIZE[1]/2 - 1
        up = i - BLOCK_SIZE[0]/2
        down = i + BLOCK_SIZE[0]/2 - 1
        #if we are out of left bound
        if left < 0:
            left = 0
            right = BLOCK_SIZE[1] - 1
        #if we are out of right bound:
        if right > MAZE_SIZE[1] - 1:
            right = MAZE_SIZE[1] - 1
            left = MAZE_SIZE[1] - 1 - BLOCK_SIZE[1]
        #if we are out of up bound
        if up < 0:
            up = 0
            down = BLOCK_SIZE[0] - 1
        #if we are out of down bound
        if down > MAZE_SIZE[0] - 1:
            down = MAZE_SIZE[0] - 1
            up = MAZE_SIZE[0] - 1 - BLOCK_SIZE[0]
        return (left, right, up, down)


class THSOM:
    sm = np.zeros((0, 0))
    tm = np.zeros((0, 0))
    neurons_num = 0
    dm = np.zeros((0, 0))
    sm2d = np.zeros((0, 0))
    def __init__(self, neurons_num, dim):
        self.neurons_num = neurons_num
        self.sm = np.random.rand(dim, neurons_num)
        self.tm = np.random.rand(neurons_num, neurons_num)
        self.dm = np.zeros((neurons_num, neurons_num))
    def get_bmu(self, vec):
        mn = 1e9
        bmu = 0
        for i in range(self.neurons_num):
            dist = np.linalg.norm(self.sm[:,i] - vec)
            if (dist < mn):
               bmu = i
               mn = dist
        return bmu
    def update_sm_weights(self, ibmu, t, vec):
        bmu = self.sm[:, ibmu]
        rad = R0 * np.exp(-t / R1)
        print ("RADIUS = ", rad)
        for i in range(self.neurons_num):
            dist = np.linalg.norm(bmu - self.sm[:,i])
            if dist < rad:
                print ("SPATIAL VECTOR ", i)
                SLR = S0 * np.exp(-dist*dist / S1)
                TLR = T0 * np.exp(-t / T1)
                DIFF = vec - self.sm[:,i]
                print ("Before ", self.sm[:,i])
                self.sm[:,i] += SLR * TLR * DIFF
                print ("SLR = ", SLR, "TLR = ", TLR, "SLR * TLR = ", SLR * TLR)
                print ("After ", self.sm[:,i])
    def dist_matrix(self):
        for i in range(self.neurons_num):
            for j in range(self.neurons_num):
                self.dm[i, j] = np.linalg.norm(self.sm[:,i] - self.sm[:,j])
        print(self.dm)
