import objects as obj
import numpy as np
from data.constants import *

ff = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':ff})
while (True):
    env = obj.Env()
    cell = (0, 0)
    for T in range(1000):
        print ("TIME = ", T)
        action_ind = np.random.randint(4)
        action = ACTIONS[action_ind]
        print ("ACTION = ", ACTIONS_WORDS[action_ind])
        cell, reward, is_done = env.step(cell, action)
        if is_done:
            break
        input("Press enter.")
