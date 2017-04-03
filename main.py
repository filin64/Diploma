import objects as obj
import numpy as np
from data.constants import *
import time

ff = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':ff})
thsom = obj.THSOM(NN, 16)
q_net = obj.QNet()

while (True):
    env = obj.Env()
    position = (0, 0)
    env.show(position)
    for T in range(1000):
        if LOG_ON: print ("TIME = ", T)
        block = env.return_block_as_vector(position) #where we are?
        ibmu = thsom.get_bmu(block) #Index of BMU
        if LOG_ON: print ("BMU", ibmu)
        thsom.update_sm_weights(ibmu, T, block)
        if np.random.rand() <= 0.35:  #Greedy Policy
            best_action = np.random.randint(len(ACTIONS))
        else:
            best_action = q_net.predict(thsom.sm[:,ibmu])
        if LOG_ON: print ('Best Action', ACTIONS_WORDS[best_action])
        action = ACTIONS[best_action]
        position, reward, is_done = env.step(position, action)
        if LOG_ON: print ('Reward', reward)
        # for i in range(thsom.neurons_num):
        #     print('Neuron', i)
        #     thsom.get_neuron_as_block(i)
        q_net.back_prop(reward, thsom.sm[:,ibmu], best_action)
        if LOG_ON: q_net.print_weights()
        # input("Press Enter")
        time.sleep(1)
        # os.system('clear')
        if is_done:
            break