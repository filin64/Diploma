import objects as obj
import numpy as np
from data.constants import *
import time

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

ff = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':ff})
thsom = obj.THSOM(NN, BLOCK_SIZE[0]*BLOCK_SIZE[1])
step = 0
prev_state = 0
while (True):
    env = obj.Env()
    position = START
    is_done = False
    env.show(position)
    for T in range(1000):
        if LOG_ON: print ("TIME = ", T)
        block = env.return_block_as_vector(position) #where we are?
        ibmu = thsom.get_bmu(block) #Index of BMU
        if LOG_ON: print ("BMU", ibmu)
        thsom.update_sm_weights(ibmu, T, block) #update spatial weights
        if T > 0: thsom.update_tm_weights(prev_state, ibmu, best_action, reward) #update temp weights from prev to cur
        # if is_done: break #if it was done on prev step then start again
        prev_state = ibmu #remember prev state
        if np.random.rand() <= max (np.exp(-step / 50), 0.3):  #Greedy Policy
            best_action = np.random.randint(len(ACTIONS))
            if LOG_ON: print('Greedy', np.exp(-step / 50))
        else:
            best_action = thsom.get_action(ibmu) #define the best action in current state
        if LOG_ON: print ('Best Action', ACTIONS_WORDS[best_action])
        action = ACTIONS[best_action]
        position, reward, is_done = env.step(position, action) #make an action
        if LOG_ON: print ('Reward', reward)
        if LOG_ON: thsom.print_sm()
        if LOG_ON: thsom.print_tm()
        env.show(position)
        step += 1 #increase general step counter
        # input("Press Enter")
        time.sleep(1)
        if is_done:
            print ('Done!', step)
            break
    break
        # os.system('clear')