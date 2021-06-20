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
    return math.exp(x)


def animate(index):
    xs.append(index)
    ys.append(func1(index))
    yss.append(func2(index))
    index += 1

    plt.cla()  # update plot with same color line

    plt.plot(xs, ys, label='x^2')
    plt.plot(xs, yss, label='log_2(x)')

    plt.legend(loc='upper left')


ani = animation.FuncAnimation(plt.gcf(), animate, interval=1000)  # 1000ms = 1s
plt.tight_layout()  # smooth padding
plt.show()
