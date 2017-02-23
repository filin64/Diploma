import objects as obj
import numpy as np
from data.constants import *


while (True):
    env = obj.Env()
    cell = (0, 0)
    thsom = obj.THSOM()
    while (True):
        action = ACTIONS[np.random.randint(4)]
        print (action)
        cell, reward, is_done, is_block_changed = env.step(cell, action)
        input("Press enter")
        env.get_block(cell)
        if is_done:
            break
