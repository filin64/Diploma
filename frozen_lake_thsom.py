import numpy as np
import gym

def get_direction(action, n):
    if action == 0: #if action is left
        return -1
    if action == 1: #if action is right
        return 1
    if action == 2: #if action is up
        return -n
    if action == 3:
        return n #if action is down

env = gym.make("FrozenLake-v0")

n = env.desc.shape[0]
m = env.desc.shape[1]
N = n*m #neuron number aka field size
TRIALS = 1000
alpha = 0.1
gamma = 0.9

TW = np.zeros((N, N)) #Transition matrix aka temporal weights
SW = np.zeros(N) #Spatial matrix
V = np.random.rand(N)
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

observation = env.reset() #this is an analogue of input vector
action = np.argmax(TW[observation])

for i in range(TRIALS):
    TW_old = TW[observation][action] #temp weight at prev time moment
    observation, reward, done, info = env.step(action) #make an action
    action = np.argmax(TW[observation]) #define the next action
    #spatial weights update
    for i, vec in enumerate(spatial_vectors):
        if np.linalg.norm(vec - spatial_vectors[observation]) == 1:
            SW[i] += alpha
    SW[observation] += 1
    #temporal weight update
    k = np.argmax(V)
    coordinate = observation + get_direction(action, n)
    if done:
        if coordinate == k:
            TW[observation][coordinate] += min(max(TW_old + alpha*(1 - TW_old + reward), 0), 1)
        else:
            TW[observation][coordinate] += min(max(TW_old + alpha*(TW_old + reward), 0), 1)
    else:
        if coordinate == k:
            TW[observation][coordinate] += min(max(TW_old + alpha*(1 - TW_old), 0), 1)
        else:
            TW[observation][coordinate] += min(max(TW_old + alpha*TW_old, 0), 1)
    V[coordinate] = (1 - gamma)*SW[observation] + gamma*np.dot(V, TW[observation])
    V /= np.max(V)
    if done:
        print("Episode finished after {} timesteps with r={}.".format(i, reward))