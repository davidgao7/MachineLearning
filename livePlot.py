"""
A live update plot
source: https://www.youtube.com/watch?v=Ercd-Ip5PfQ
"""
import math

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

plt.style.use('fivethirtyeight')  # for nicer plot

xs = []
ys = []
yss = []
count = 0


def func1(x):
    return x ** 2


def func2(x):
    return x ** 2.5


def animate(index):
    xs.append(index)
    ys.append(func1(index))
    yss.append(func2(index))

    plt.cla()  # update plot with same color line

    plt.plot(xs, ys, label='x^2')
    plt.plot(xs, yss, label='x^2.5')

    plt.legend(loc='upper left')


data = np.arange(100).tolist()
ani = animation.FuncAnimation(plt.gcf(), animate, data, interval=1000)  # 1000ms = 1s
# plt.tight_layout()  # smooth padding, smooth the graph, not necessary in this case
plt.show()
