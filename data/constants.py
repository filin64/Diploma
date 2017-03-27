import numpy as np
FILE_PATH = 'env/Map_v03'
with open(FILE_PATH) as f:
        ls = [list(l.strip('\n')) for l in f.readlines()]
MAZE_SIZE = np.shape(ls)
BLOCK_SIZE = (4, 4)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
WALL = '#'
WALL_PUN = -0.5
HOLE = 'H'
HOLE_PUN = -1
FIN = 'F'
FIN_PUN = 1
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_WORDS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
COLORS = ['grey', 'white', 'yellow', 'blue', 'magenta', 'cyan', 'white']
L0 = 1
L1 = 1
R0 = 3
R1 = 25
T0 = 1
T1 = 200
S0 = 1
S1 = 4