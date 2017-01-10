import numpy as np
import gym

def get_direction(action, n):
    if action == 0: #if action is left
        return -1
    if action == 1: #if action is right
        return 1
    if action == 2: #if action is down
        return n
    if action == 3:
        return -n #if action is up

def get_action(observation, new_cell):
    if new_cell < observation:
        if new_cell == observation % n:
            return 3 #move up
        return 0 #move left
    if new_cell > observation:
        if new_cell == observation % n:
            return  # move down
        return 1  # move right
    return np.random.rand(4)

env = gym.make("FrozenLake-v0")

n = env.desc.shape[0]
m = env.desc.shape[1]
N = n*m #neuron number aka field size
TRIALS = 100
STEPS = 50
alpha = 0.1
gamma = 0.9

TW = np.zeros((N, N)) #Transition matrix aka temporal weights
SW = np.zeros(N) #Spatial matrix
V = np.random.rand(N)
#predefine matrix. We can move only to neighbor cells
for i in range(N):
    if i - 1 >= 0:
        TW[i][i - 1] = 1
    if i + 1 < N:
        TW[i][i + 1] = 1
    if i - n >= 0:
        TW[i][i - n] = 1
    if i + n < N:
        TW[i][i + n] = 1

spatial_vectors = []
for i in range(n):
    for j in range(m):
        spatial_vectors.append(np.array([i, j]))

for j in range (TRIALS):
    observation = env.reset() #start iteration

    for i in range(STEPS):
        action = get_action(observation, np.argmax(TW[observation])) # define the next action
        #spatial weights update
        for i, vec in enumerate(spatial_vectors):
            if np.linalg.norm(vec - spatial_vectors[observation]) == 1:
                SW[i] += alpha
        SW[observation] += 1
        #temporal weight update
        k = np.argmax(V)
        new_cell = observation + get_direction(action, n)  # define cell in the next moment of time
        TW_old = TW[observation][new_cell] #old value
        observation, reward, done, info = env.step(action)  # make an action
        if done:
            if new_cell == k:
                TW[observation][new_cell] = min(max(TW_old + alpha * (1 - TW_old + reward), 0), 1)
            else:
                TW[observation][new_cell] = min(max(TW_old + alpha * (- TW_old + reward), 0), 1)
        else:
            if new_cell == k:
                TW[observation][new_cell] = min(max(TW_old + alpha * (1 - TW_old), 0), 1)
            else:
                TW[observation][new_cell] = min(max(TW_old - alpha * TW_old, 0), 1)
        V[new_cell] = (1 - gamma) * SW[observation] + gamma * np.dot(V, TW[observation])
        V /= np.max(V)
        if done:
            print("Episode finished after {} timesteps with r={}. Observation={}".format(i, reward, observation))
print (TW)