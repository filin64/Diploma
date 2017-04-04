import objects as obj
import numpy as np
from data.constants import *
import time

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

ff = lambda x: "%.5f" % x
np.set_printoptions(formatter={'float_kind':ff})
thsom = obj.THSOM(NN, BLOCK_SIZE[0]*BLOCK_SIZE[1])
# q_net = obj.QNet()
Q = np.random.rand(NN, len(ACTIONS))*0.1

while (True):
    env = obj.Env()
    position = (0, 0)
    is_done = False
    env.show(position)
    for T in range(1000):
        if LOG_ON: print ("TIME = ", T)
        block = env.return_block_as_vector(position) #where we are?
        ibmu = thsom.get_bmu(block) #Index of BMU
        if LOG_ON: print ("BMU", ibmu)
        thsom.update_sm_weights(ibmu, T, block)
        if np.random.rand() <= 0.25:  #Greedy Policy
            best_action = np.random.randint(len(ACTIONS))
            print ('Greedy')
        else:
            # best_action = q_net.predict(thsom.sm[:,ibmu])
            best_action = np.argmax(Q[ibmu, :])
        if LOG_ON: print ('Best Action', ACTIONS_WORDS[best_action])
        action = ACTIONS[best_action]
        position, reward, is_done = env.step(position, action)
        if T != 0:
            Q[old_state, old_action] = Q[old_state, old_action] + GAMMA*(reward + sigmoid(Q[ibmu, best_action]))
        old_action = best_action
        old_state = ibmu
        if LOG_ON: print ('Reward', reward)
        if LOG_ON:
            for i in range(thsom.neurons_num):
                print('Neuron', i)
                thsom.get_neuron_as_block(i)
        # q_net.back_prop(reward, thsom.sm[:,ibmu], best_action)
        # if LOG_ON: q_net.print_weights()
        if LOG_ON: print (Q)
        env.show(position)
        input("Press Enter")
        if is_done:
            break
        # time.sleep(1)
        # os.system('clear')