import matplotlib.animation as animation
import numpy as np
from scipy.integrate import odeint
from pylab import *
import matplotlib.pyplot as plt

price = 735
effort_scale = 2e-6
marginal_cost = 556380
carrying_capacity = 3.2e6
intrinsic_growth = 0.08
catchability = marginal_cost / (price * 0.25e6)


def BoatFishSystem(state, t, time_scale=0.1):
    stock, effort = state
    net_growth = intrinsic_growth * stock * (1 - (stock/carrying_capacity))
    harvest = catchability * stock * effort
    d_stock = net_growth - harvest
    d_effort = effort_scale * (price * catchability * stock - marginal_cost)
    return [d_stock * time_scale, d_effort * time_scale]

xx = -(intrinsic_growth * marginal_cost) / (price * catchability * carrying_capacity)

J_equil = np.array([
    [xx, -(marginal_cost/price)],
    [(price*catchability), 0]
])

from numpy import linalg as LA

values, vectors = LA.eig(J_equil)
print values


# def BoatFishSystem(state, t):
#     fish, boat = state
#     d_fish = fish * (alpha - beta * boat - fish)
#     d_boat = -boat * (gamma - sigma * fish)
#     return [d_fish, d_boat]

t = np.arange(0, 9999, 10)
init_state = [0.75e6, 15]
state = odeint(BoatFishSystem, init_state, t)

step = 1
for i in np.arange(0, 9999, 10):
    fig = figure()
    xlabel('number of fish')
    ylabel('number of boats')
    plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
    plot(state[0:step, 0], state[0:step, 1], 'b-')

    g_nullcline = marginal_cost / (price * catchability)
    axvline(x=g_nullcline, linestyle='--', color='r', alpha=0.2)

    def f(x):
        return (intrinsic_growth / catchability) * (1 - (x/carrying_capacity))

    xs = np.array(linspace(0, 800000))
    ys = [f(x) for x in xs]

    s = '%03d' % step
    print s
    title('time step (of 1000): %s' % s)
    plot(xs, ys, 'r--', alpha=0.2)

    savefig('imgs/iter%s.png' % s)
    fig.clf()
    step += 1



# axvline(x=0.66666666, linestyle='--', color='r', alpha=0.2)
# show()
# savefig('diff-direct.png', bbox_inches='tight')


# def animate(i):
#     title('\ntime step (of 200): %s\n' % i)
#     plot(state[0:i, 0], state[0:i, 1], 'b-')

# ani = animation.FuncAnimation(fig, animate, interval=2)

# show()

# step = 1
# for i in np.arange(0, 20, 0.1):

#     plt.figure(1)

#     plt.subplot(211)
#     xlabel('number of fish\n')
#     ylabel('number of boats')
#     plt.axis([0.5, 1, 1, 1.7])
#     plot(state[:, 0], state[:, 1], 'b-', alpha=0.2)
#     plot(xs, ys, 'r--', alpha=0.2)
#     axvline(x=0.66666666, linestyle='--', color='r', alpha=0.2)
#     plot(state[0:step, 0], state[0:step, 1], 'b-')
#     s = '%03d' % step
#     print s
#     title('time step (of 200): %s' % s)

#     plt.subplot(212)
#     plt.axis([0, 20, 0.5, 1.7])
#     a, = plot(t, state[:, 0], 'k-', alpha=0.2)
#     b, = plot(t, state[:, 1], 'g-', alpha=0.2)

#     plot(t[0:step], state[0:step, 0], 'k-')
#     plot(t[0:step], state[0:step, 1], 'g-')

#     plt.legend()

#     xlabel('time')
#     ylabel('number of boats/fish')

#     plt.tight_layout()
#     savefig('imgs/iter%s.png' % s)

#     step += 1
#     plt.close()

# animated gif
# convert -delay 3 -loop 0 imgs/*.png differential-animated-dual.gif
