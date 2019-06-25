try:
    from .core import Observer
except ImportError:
    from qcifc.core import Observer

import matplotlib.pyplot as plt


class Plotter(Observer):

    def __init__(self, n):
        self.fig, self.ax = plt.subplots()
        self.x = []
        self.y = [[] for _ in range(n)]
        args = []
        for y in self.y:
            args.append(self.x)
            args.append(y)
        self.ax.semilogy(*args)
        self.iy = 0

    def update(self, items, **kwargs):
        if items and isinstance(items[0], float):
            self.y[self.iy].append(items[1])
            self.iy += 1

    def reset(self):
        x = range(len(self.y[0]))
        args = []
        for y in self.y:
            args.append(x)
            args.append(y)
        self.ax.clear()
        self.ax.semilogy(*args)
        self.fig.canvas.draw()
        self.fig.show()
        self.iy = 0
