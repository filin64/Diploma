import numpy as np
from data.constants import *
from termcolor import colored
from collections import deque
import logging

logger = logging.getLogger()
logging.basicConfig(level = logging.CRITICAL)

def generate_patterns():
    n = BLOCK_SIZE[0]
    m = BLOCK_SIZE[1]
    matrix = np.zeros((n*m, 9))
    #generate horizontal walls:
    matrix[0:m, 0] = 1 #top
    matrix[n*m - m: n*m, 1] = 1 #bottom
    # generate vertical walls:
    matrix[0:m*n:m, 3] = 1 #left
    matrix[m-1:m*n:m, 4] = 1 #right
    #generate corners:
    matrix[0:m, 5] = 1
    matrix[0:m * n:m, 5] = 1
    matrix[0:m, 6] = 1
    matrix[m - 1:m * n:m, 6] = 1
    matrix[n * m - m: n * m, 7] = 1
    matrix[0:m * n:m, 7] = 1
    matrix[n * m - m: n * m, 2] = 1
    matrix[m - 1:m * n:m, 2] = 1
    return matrix


class Env:
    maze = []
    num_maze = [] #numerical representation of maze
    START = 0 #Start position coordinates
    FIN = 0 #Finish position coordinates
    MAZE_SIZE = 0
    def __init__(self, file_path):
        f = open(file_path)
        lines = f.readlines()
        self.maze = [list(l.strip('\n')) for l in lines] #read maze from file
        self.MAZE_SIZE = np.shape(self.maze)
        self.num_maze = np.zeros((self.MAZE_SIZE)) 
        for i in range (self.MAZE_SIZE[0]):
            for j in range (self.MAZE_SIZE[1]):
                if self.maze[i][j] == WALL:
                    self.num_maze[i, j] = 1
                else:
                    self.num_maze[i, j] = 0
        s = ''.join(lines)
        s = s.replace('\n', '')
        start = s.index('S')
        fin = s.index('F')
        self.START = (int(start / self.MAZE_SIZE[1]), start % self.MAZE_SIZE[1]) #defining where the points are
        self.FIN = (int(fin / self.MAZE_SIZE[1]), fin % self.MAZE_SIZE[1])
        self.update(self.START, 'A')
    #---------------------------------------------------------------------------------------------------#
    # To display process of agent's movement
    def show(self, position):
        left, right, up, down = self.get_block(position)
        for i in range(np.shape(self.maze)[0]):
            for j in range(np.shape(self.maze)[1]):
                if j >= left and j <= right and i >= up and i <= down:
                     print(colored(self.maze[i][j], 'red'), end='')
                elif self.maze[i][j] == WALL:
                    print(colored(self.maze[i][j], 'green'), end='')
                elif self.maze[i][j] == FIN:
                    print(colored(self.maze[i][j], 'blue'), end='')
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
        if self.maze[i][j] == WALL:
            # self.update((i, j), '*')
            # self.update(position, '0')
            return position, WALL_PUN, False
        if self.maze[i][j] == FIN:
            self.update((i, j), '*')
            return (i, j), FIN_PUN, True
        self.update(position, '0')
        self.update((i, j), 'A')
        prev_dist = np.linalg.norm(np.array(self.FIN) - np.array(position)) #distance to finish point
        cur_dist = np.linalg.norm(np.array(self.FIN) - np.array((i, j)))
        reward = DIST_REWARD+REWARD if cur_dist < prev_dist else REWARD #motivation to move straight to the finish
        return (i, j), reward, False
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
        if right > self.MAZE_SIZE[1] - 1:
            right = self.MAZE_SIZE[1] - 1
            left = self.MAZE_SIZE[1] - BLOCK_SIZE[1]
        #if we are out of up bound
        if up < 0:
            up = 0
            down = BLOCK_SIZE[0] - 1
        #if we are out of down bound
        if down > self.MAZE_SIZE[0] - 1:
            down = self.MAZE_SIZE[0] - 1
            up = self.MAZE_SIZE[0] - BLOCK_SIZE[0]
        return (left, right, up, down)
    # ---------------------------------------------------------------------------------------------------#
    # Numerical value of block
    def return_block_as_vector(self, position):
        left, right, up, down = self.get_block(position)
        return self.num_maze[int(up):int(down)+1, int(left):int(right)+1].ravel()


class THSOM:
    sm = np.zeros((0, 0))
    tm = []
    neurons_num = 0
    dm = np.zeros((0, 0))
    # ---------------------------------------------------------------------------------------------------#
    # Constructor
    def __init__(self, neurons_num, dim):
        #dim - length of vectors
        self.neurons_num = neurons_num
        if RAND_SM == True:
            self.sm = np.random.rand(dim, neurons_num) * 0.8
        else:
            self.sm = generate_patterns()
        self.tm = [[[0, 0, 0, 0] for i in range(neurons_num)] for j in range(neurons_num)]
        #######first - actions, last - weight

    # ---------------------------------------------------------------------------------------------------#
    def get_bmu(self, vec):
        mn = 1e9
        bmu = 0
        for i in range(self.neurons_num):
            dist = self.dist(x=vec, y=self.sm[:,i])
            logger.info("Dist between " + str(i) + " = " + str(dist))
            if (dist < mn):
               bmu = i
               mn = dist
        return bmu

    # ---------------------------------------------------------------------------------------------------#
    def print_bmu(self, ibmu):
        bmu = self.sm[:, ibmu]
        bmu.shape = (BLOCK_SIZE[0], BLOCK_SIZE[1])
        for i in range(bmu.shape[0]):
            for j in range(bmu.shape[1]):
                print(bmu[i, j], end='')
            print()

    # ---------------------------------------------------------------------------------------------------#
    def update_sm_weights(self, ibmu, t, vec):
        #ibmu - index of bmu, t - moment of time, vec - input vector
        bmu = self.sm[:, ibmu]
        # rad = max (10**(-10), R0 * np.exp(-t / R1))
        rad = 10**(-10)
        logger.info("RADIUS = " + str(rad))
        for i in range(self.neurons_num):
            dist = self.dist(x=self.sm[:, i], y=bmu)
            if dist < rad:
                logger.info("SPATIAL VECTOR " + str(i))
                SLR = S0 * np.exp(-dist*dist/S1)
                TLR = T0 * np.exp(-t / T1)
                DIFF = vec - self.sm[:,i]
                logger.info("Before " + str(self.sm[:, i]))
                self.sm[:,i] += SLR * TLR * DIFF
                logger.info("Dist" + str(dist) + "SLR = " + str(SLR) + "TLR = " + str(TLR) + "DIFF" + str(DIFF))
                logger.info("After " + str(self.sm[:, i]))

    # ---------------------------------------------------------------------------------------------------#
    def update_tm_weights(self, prev, cur, action, reward):
        #prev - previous state , cur - current state
        self.tm[prev][cur][action] = min(max(self.tm[prev][cur][action] + reward, MIN_TM), MAX_TM)
    def get_action(self, cur):
        #cur - current neuron
        max_w = -1
        action = 0
        for i in self.tm[cur]:
            if np.max(i) > max_w:
                max_w = np.max(i)
                action = np.argmax(i)
        if max_w == 0:
            return np.random.randint(4)
        return action

    # ---------------------------------------------------------------------------------------------------#
    # We define special metrics
    def dist(self, x, y):
        # x - input, y - neuron
        y = [np.uint64(1) if i > WALL_THOLD else np.uint64(0) for i in y]
        x = [np.uint64(1) if i > WALL_THOLD else np.uint64(0) for i in x]
        alpha = 0.3
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
            d[str(shift)] = s
        shift = 0
        while y_rd[-1] == 0 and abs(shift) < len(y_rd):
            y_rd.rotate(1)
            y_list_rd = list(np.uint64(i) for i in y_rd)
            shift += 1
            s = sum(np.bitwise_xor(x, y_list_rd))
            d[str(shift)] = s
        #ind - количество сдвигов
        ind = min(d, key=lambda i: d[i])
        ans = alpha * (1 - np.exp(-abs(int(ind)) / betta)) + (1 - alpha) * (1 - np.exp(-d[ind] / betta))
        return ans

    # ---------------------------------------------------------------------------------------------------#
    # Graphical neuron representation
    def get_neuron_as_block(self, i):
        x = self.sm[:,i]
        for i in range(len(x)):
            if x[i] > WALL_THOLD:
                print('#', end='')
            else:
                print ('0', end='')
            if (i + 1) % BLOCK_SIZE[0] == 0:
                print ()
    def print_tm(self):
        for i in self.tm:
            print(i)
    def print_sm(self):
        for i in range(self.neurons_num):
            print('Neuron', i)
            self.get_neuron_as_block(i)
