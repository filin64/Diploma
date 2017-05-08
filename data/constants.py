# FILE_PATH = ['env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08',
#              'env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08', 'env/Map_v08']
FILE_PATH = ['env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09',
             'env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09', 'env/Map_v09']
BLOCK_SIZE = (3, 3)
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
WALL = '#'
WALL_PUN = -0.95
FIN = 'F'
FIN_PUN = 0.03
ACTIONS = [UP, DOWN, LEFT, RIGHT]
ACTIONS_WORDS = ['UP', 'DOWN', 'LEFT', 'RIGHT']
COLORS = ['grey', 'white', 'yellow', 'blue', 'magenta', 'cyan', 'white']
L0 = 1
L1 = 1
R0 = 0.7
R1 = 20
T0 = 1
T1 = 100
S0 = 0.95
S1 = 0.8
NN = 20
#Neurons num
GAMMA = 0.95
REWARD = 0.0001
DELTA = 0.000001
LOG_ON = False
WALL_THOLD = 0.5
DIST_REWARD = 0.008
MIN_TM = 0
MAX_TM = 1
M0 = 0
M1 = 50
RAND_SM = True
VISUALIZE = True