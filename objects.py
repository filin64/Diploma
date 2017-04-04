import numpy as np
from data.constants import *
from termcolor import colored
from collections import deque

class Env:
    maze = []
    num_maze = []
    def __init__(self):
        f = open(FILE_PATH)
        self.maze = [list(l.strip('\n')) for l in f.readlines()]
        self.num_maze = np.zeros((MAZE_SIZE))
        for i in range (MAZE_SIZE[0]):
            for j in range (MAZE_SIZE[1]):
                if self.maze[i][j] == WALL:
                    self.num_maze[i, j] = 1
                else:
                    self.num_maze[i, j] = 0
        self.update((0, 0), 'A')
    #---------------------------------------------------------------------------------------------------#
    # To display process of agent's movement
    def show(self, position):
        left, right, up, down = self.get_block(position)
        for i in range(np.shape(self.maze)[0]):
            for j in range(np.shape(self.maze)[1]):
                if j >= left and j <= right and i >= up and i <= down:
                     print(colored(self.maze[i][j], 'red'), end='')
                else:
                     print(colored(self.maze[i][j], 'white'), end ='')
            print()
        print ('---------------------------------------------------------------------------------------------')
    #---------------------------------------------------------------------------------------------------#
    #Update cell with the value for displaying
    def update(self, cell, val):
        self.maze[cell[0]][cell[1]] = val
    #---------------------------------------------------------------------------------------------------#
    # It returns all neccessary data which is needed to define strategy
    # Return values: new_cell, reward, is_done
    def step(self, position, action):
        i, j = np.array(position) + np.array(action)
        # if we are out of bounds
        if i < 0 or j < 0 or i >= MAZE_SIZE[0] or j >= MAZE_SIZE[1]:
            self.update(position, 'A')
            return position, BOUND_PUN, True
        if self.maze[i][j] == WALL:
            self.update((i, j), '*')
            self.update(position, '0')
            return position, WALL_PUN, True
        if self.maze[i][j] == HOLE:
            return (i, j), HOLE_PUN, True
        if self.maze[i][j] == FIN:
            self.update((i, j), '*')
            return (i, j), FIN_PUN, True
        self.update(position, '0')
        self.update((i, j), 'A')
        return (i, j), REWARD, False
    # ---------------------------------------------------------------------------------------------------#
    # 4x4 Block were we are now
    def get_block(self, position):
        i, j = position
        left = j - int(BLOCK_SIZE[1]/2)
        right = j + int(BLOCK_SIZE[1]/2)
        if BLOCK_SIZE[1] % 2 == 0:
            right -= 1
        up = i - int(BLOCK_SIZE[0]/2)
        down = i + int(BLOCK_SIZE[0] / 2)
        if BLOCK_SIZE[0] % 2 == 0:
            down -= 1
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
    # ---------------------------------------------------------------------------------------------------#
    # Numerical value of block
    def return_block_as_vector(self, position):
        left, right, up, down = self.get_block(position)
        return self.num_maze[int(up):int(down)+1, int(left):int(right)+1].ravel()


class THSOM:
    sm = np.zeros((0, 0))
    tm = np.zeros((0, 0))
    neurons_num = 0
    dm = np.zeros((0, 0))
    def __init__(self, neurons_num, dim):
        #dim - length of vectors
        self.neurons_num = neurons_num
        self.sm = np.random.rand(dim, neurons_num) * 0.6
    def get_bmu(self, vec):
        mn = 1e9
        bmu = 0
        for i in range(self.neurons_num):
            dist = self.dist(x=vec, y=self.sm[:,i])
            if (dist < mn):
               bmu = i
               mn = dist
        return bmu
    def update_sm_weights(self, ibmu, t, vec):
        #ibmu - index of bmu, t - moment of time, vec - input vector
        bmu = self.sm[:, ibmu]
        # rad = max (10**(-10), R0 * np.exp(-t / R1))
        rad = 10**(-10)
        if LOG_ON: print ("RADIUS = ", rad)
        for i in range(self.neurons_num):
            dist = self.dist(x=self.sm[:,i], y=bmu)
            if dist < rad:
                if LOG_ON: print ("SPATIAL VECTOR ", i)
                SLR = S0 * np.exp(-dist*dist/S1)
                TLR = T0 * np.exp(-t / T1)
                DIFF = vec - self.sm[:,i]
                if LOG_ON: print ("Before ", self.sm[:,i])
                self.sm[:,i] += SLR * TLR * DIFF
                if LOG_ON: print ("Dist", dist, "SLR = ", SLR, "TLR = ", TLR, "DIFF", DIFF)
                if LOG_ON: print ("After ", self.sm[:,i])
    def dist(self, x, y):
        # x - input, y - neuron
        y = [np.uint64(1) if i > WALL_THOLD else np.uint64(0) for i in y]
        x = [np.uint64(1) if i > WALL_THOLD else np.uint64(0) for i in x]
        alpha = 0.5
        betta = 5
        y_ld = deque(y) #for left shift
        y_rd = deque(y) #for right shift
        d = dict()
        shift = 0
        d[shift] = sum(np.bitwise_xor(x, y))
        # shift left
        while y_ld[0] == 0 and abs(shift) < len(y_ld):
            y_ld.rotate(-1)
            y_list_ld = list(np.uint64(i) for i in y_ld)
            shift -= 1
            s = sum(np.bitwise_xor(x, y_list_ld))
            d[abs(shift)] = s
        shift = 0
        while y_rd[-1] == 0 and abs(shift) < len(y_rd):
            y_rd.rotate(1)
            y_list_rd = list(np.uint64(i) for i in y_rd)
            shift += 1
            s = sum(np.bitwise_xor(x, y_list_rd))
            d[abs(shift)] = s
        #ind - количество сдвигов
        ind = min(d, key=lambda i: d[i])
        ans = alpha * (1 - np.exp(-ind / betta)) + (1 - alpha) * (1 - np.exp(-d[ind] / betta))
        return ans
    def get_neuron_as_block(self, i):
        x = self.sm[:,i]
        for i in range(len(x)):
            if x[i] > WALL_THOLD:
                print('#', end='')
            else:
                print ('0', end='')
            if (i + 1) % BLOCK_SIZE[0] == 0:
                print ()
class QNet:
    weights = np.zeros((0, 0))
    n = BLOCK_SIZE[0]*BLOCK_SIZE[1]
    def __init__(self):
        self.weights = np.random.rand(self.n, 4)
    def predict(self, x):
        # x - input signal 16x1
        ans = np.dot(x.T, self.weights)
        if LOG_ON: print ('Distribution', ans)
        return np.argmax(ans)
    def back_prop(self, reward, x, ind):
        #x - input vector, y - output answer
        old_weights = np.copy(self.weights[:, ind])
        self.weights[:, ind] = self.weights[:, ind] + (x + 1) * reward + np.ones(self.n).T * DELTA
        if LOG_ON: print ('Weights Delta', self.weights[:, ind] - old_weights)
    def print_weights(self):
        print (self.weights)
