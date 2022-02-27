import matplotlib.pyplot as plt
from IPython.display import display
import numbers


class RealTimeGraph:

    def __init__(self, title,  x_label=None, y_label=None, legend=None, figsize=(7,5),
                 ):
        f, ax = plt.subplots(figsize=figsize)
        self.ax = ax
        self.f = f
        self.x_label = x_label
        self.y_label = y_label
        self.legend = legend
        self.title = title
        self.ax.set_title(title)
        plt.ion()
        self.x_list = []
        self.y_list = []

    def add(self, x_list, y_list, legend=None):
        if legend is not None:
            self.legend = legend
        if isinstance(y_list[0], numbers.Number):
            self.y_list = [y_list]
        else:
            self.y_list = y_list
        self.x_list = x_list

        self.ax.clear()
        for y in y_list:
            self.ax.plot(x_list, y)

        self.ax.set_xlabel(self.x_label)
        self.ax.set_ylabel(self.y_label)

        if self.legend is not None:
            self.ax.legend(self.legend)
        display(self.f)
        plt.pause(0.1)

    def __enter__(self):
        return self.add

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.save_fig()

    def save_fig(self, save_path=None):
        if len(self.x_list) == 0 or len(self.y_list) == 0 or save_path is None:
            return
        if save_path is not None:
            self.save_path = save_path

        plt.figure()
        plt.title(self.title)
        for y in self.y_list:
            plt.plot(self.x_list, y)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        if self.legend is not None:
            plt.legend(self.legend)
        if self.save_path is not None:
            plt.savefig(self.save_path)
            plt.clf()
            plt.close()

if __name__ == "__main__":
    import math
    import time
    x = []
    y = []
    y_1 = []

    with RealTimeGraph("test", save_path="tset.png") as rg:
        for i in range(100):
            x.append(i)
            y.append(math.cos(i))
            y_1.append(math.log(i+1))
            rg(x, [y,y_1])

            time.sleep(1)
