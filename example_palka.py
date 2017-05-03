import gym
import math
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D




def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def episode(alpha,beta):
    cum_reward = 0
    env =gym.make('CartPole-v0')
    
    observation = env.reset()
    while True:
        #env.render()
        action = np.random.binomial(1, sigmoid(alpha*observation[3]+beta*observation[1]))
        observation, reward, done, info = env.step(action)
        cum_reward += reward
        if done:
            break
    print(cum_reward,alpha  )
    return cum_reward

start = 0.0 
stop = 100.0
N=2
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
'''
X = [i for i in np.linspace(start,stop,N) for _ in np.linspace(start,stop,N) ]
Y = np.linspace(start,stop,N)
'''
x = y = np.arange(start, stop, 1)
X, Y = np.meshgrid(x, y)
zs = np.array([episode(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z)  
plt.show()
