import objects as obj
import numpy as np
from data.constants import *
from matplotlib import pyplot as plt
import logging
import time
ff = lambda x: "%.5f" % x #setting float precision
np.set_printoptions(formatter={'float_kind':ff})
thsom = obj.THSOM(NN, BLOCK_SIZE[0]*BLOCK_SIZE[1]) #creating THSOM instance
M0 = 0 #parameters for greedy policy
M1 = 100
prev_state = 0 #previous env state
STEPS = []
for file_num, file_path in enumerate(FILE_PATH):
    env = obj.Env(file_path)
    position = env.START
    is_done = False
    thsom.print_sm()
    if file_num > 0: M0 = 1000 #if we're not in training session turn off greedy-policy
    for T in range(1000):
        info_file = open('data/info.log', 'w')
        info_file.truncate()
        logging.critical("TIME = " + str(T))
        block = env.return_block_as_vector(position) #numerical (vector) presentation of current block  we are?
        ibmu = thsom.get_bmu(block) #Index of BMU
        thsom.get_neuron_as_block(ibmu, info_file) #present current state as symbol block
        logging.warning("BMU " + str(ibmu))
        thsom.update_sm_weights(ibmu, T, block) #update spatial weights
        if T > 0: dist_to_finish = env.dist_to_fin(prev_position) #check distance to finish point at the previous step
        if T > 0: thsom.update_tm_weights(prev_state, ibmu, best_action, reward, dist_to_finish) #update temp weights from prev to cur
        logging.critical('----------------------------------------')
        # input("Press Enter")
        time.sleep(0.5)
        prev_state = ibmu #remember prev state
        prev_position = position #remember prev position
        if np.random.rand() <= max (np.exp(-M0 / M1), 0.1):  #Greedy Policy
            best_action = np.random.randint(len(ACTIONS))
            logging.info('Greedy ' + str(np.exp(-M0 / M1)))
        else: best_action = thsom.get_action(ibmu) #define the best action in current state
        logging.info('Best Action' + ACTIONS_WORDS[best_action])
        info_file.write('Best Action' + ACTIONS_WORDS[best_action] + '\n')
        action = ACTIONS[best_action]
        position, reward, is_done = env.step(position, action) #make an action
        logging.warning('Reward ' + str(reward))
        info_file.write('Reward ' + str(reward) + '\n')
        # if LOG_ON: thsom.print_sm()
        if LOG_ON: thsom.print_tm()
        T += 1 #increase general step counter
        M0 += 1
        info_file.close()
        if VISUALIZE: env.show(position)
        if is_done:
            print ('Done!', T)
            STEPS.append(T)
            break
    # input("Press Enter")
plt.plot(STEPS)
plt.show()