import matplotlib.animation as animation
from scipy.integrate import odeint
from numpy import arange
from pylab import *

alpha = 2
beta = 1
sigma = 1.5
gamma = 1


def BoatFishSystem(state, t):
    fish, boat = state
    d_fish = fish * (alpha - beta * boat - fish)
    d_boat = -boat * (gamma - sigma * fish)
    return [d_fish, d_boat]

t = arange(0, 1000, 0.05)
init_state = [1, 1]
state = odeint(BoatFishSystem, init_state, t)


fig = figure()
xlabel('number of fish')
ylabel('number of boats')
plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)


def animate(i):
    plot(state[0:i, 0], state[0:i, 1], 'b-')

ani = animation.FuncAnimation(fig, animate, interval=2)

show()
