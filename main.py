import objects as obj
import numpy as np
from data.constants import *


while (True):
    env = obj.Env()
    cell = (0, 0)
    thsom = obj.THSOM(20, 16)
    for T in range(1000):
        print ("TIME = ", T)
        action = ACTIONS[np.random.randint(4)]
        print ("ACTION = ", action)
        cell, reward, is_done, is_block_changed = env.step(cell, action)
        if is_block_changed:
            block = env.get_block(cell)[1]
            in_vec = block.ravel().T
            print ("Input Vector ", in_vec)
            bmu = thsom.get_bmu(in_vec)
            print ("BMU ", bmu)
            thsom.update_sm_weights(bmu, T, in_vec)
            # input("Press enter.")
        if is_done:
            break
    print (thsom.sm)
    input("Press enter.")
