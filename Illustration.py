import numpy as np
import time
import threading
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# visualise how r_vec and t_vec evolve
def __init():
    line1.set_data([], [])
    line2.set_data([], [])
    return line1, line2

def __update(frame):
    x = frame / 10
    x_data.append(x)
    y1_data.append(np.sin(x))
    y2_data.append(np.cos(x))

    line1.set_data(x_data, y1_data)
    line2.set_data(x_data, y2_data)
    return line1, line2

def DynamicDrawThreeLines(data_list, legend_list):
    n_data = len(data_list)
    x_data, y1_data, y2_data, y3_data = [], [], [], []
    plt.ion()
    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label=legend_list[0])
    line2, = ax.plot([], [], 'g-', label=legend_list[1])
    line3, = ax.plot([], [], 'b-', label=legend_list[2])
    for i in range(n_data):
        x_data.append(i)
        y1_data.append(data_list[i][0])
        y2_data.append(data_list[i][1])
        y3_data.append(data_list[i][2])
        line1.set_data(x_data, y1_data)
        line2.set_data(x_data, y2_data)
        line3.set_data(x_data, y3_data)

        ax.relim()        # recompute limits if needed
        ax.autoscale_view()
        
        plt.draw()
        plt.pause(0.1)  # short pause for update

    plt.ioff()
    plt.show()

if __name__ == "__main__":    
# Data containers
    x_data = []
    y1_data = []
    y2_data = []

    fig, ax = plt.subplots()
    line1, = ax.plot([], [], 'r-', label="sin(x)")
    line2, = ax.plot([], [], 'b-', label="cos(x)")

    ax.set_xlim(0, 10)
    ax.set_ylim(-2, 2)
    ax.legend()

    ani = FuncAnimation(fig, __update, frames=range(200),
                        init_func=__init, blit=True, interval=50, repeat=False)

    plt.show()