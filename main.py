import objects as obj
import numpy as np
from data.constants import *
import time
import logging

logger = logging.getLogger()
logging.basicConfig(level = logging.INFO)
ff = lambda x: "%.5f" % x #setting float precision
np.set_printoptions(formatter={'float_kind':ff})
thsom = obj.THSOM(NN, BLOCK_SIZE[0]*BLOCK_SIZE[1]) #creating THSOM instance
M0 = 0 #parameters for greedy policy
M1 = 10
prev_state = 0 #previous env state
for file_num, file_path in enumerate(FILE_PATH):
    env = obj.Env(file_path)
    position = env.START
    is_done = False
    env.show(position)
    thsom.print_sm()
    if file_num > 0: M0 = 1000 #if we're not in training session turn off greedy-policy
    for T in range(10000):
        logger.critical("TIME = " + str(T))
        block = env.return_block_as_vector(position) #where we are?
        ibmu = thsom.get_bmu(block) #Index of BMU
        logger.warning("BMU " + str(ibmu))
        thsom.update_sm_weights(ibmu, T, block) #update spatial weights
        if T > 0: thsom.update_tm_weights(prev_state, ibmu, best_action, reward) #update temp weights from prev to cur
        prev_state = ibmu #remember prev state
        if np.random.rand() <= max (np.exp(-M0 / M1), 0.3):  #Greedy Policy
            best_action = np.random.randint(len(ACTIONS))
            logger.info('Greedy ' + str(np.exp(-M0 / M1)))
        else: best_action = thsom.get_action(ibmu) #define the best action in current state
        logger.info('Best Action' + ACTIONS_WORDS[best_action])
        action = ACTIONS[best_action]
        position, reward, is_done = env.step(position, action) #make an action
        logger.warning('Reward ' + str(reward))
        if LOG_ON: thsom.print_sm()
        if LOG_ON: thsom.print_tm()
        env.show(position)
        T += 1 #increase general step counter
        M0 += 1
        time.sleep(0.3)
        # input("Press Enter")
        if is_done:
            print ('Done!', T)
            break
    input("Press Enter")