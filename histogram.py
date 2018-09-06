import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 15})

def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2 - 0.08, 1.03*height, '%s' % float(height))

if __name__ == '__main__':

    n_groups = 2

    # F1, precision, recall
    means_uniform = (69.7, 53.8)

    means_gaussian = (76.4, 62.5)



    fig, ax = plt.subplots()

    index = np.arange(n_groups)
    bar_width = 0.3

    opacity = 0.6
    error_config = {'ecolor': '0.3'}

    aaa = ax.bar(index, means_uniform, bar_width,
                    alpha=opacity, color='b',
                    error_kw=error_config,
                    label='SER without attention')

    bbb = ax.bar(index + bar_width, means_gaussian, bar_width,
                    alpha=opacity, color='r',
                    error_kw=error_config,
                    label='SER with attention')

    autolabel(aaa)
    autolabel(bbb)

    ax.set_ylabel('%')
    ax.set_xticks(index + bar_width / 2)
    ax.set_xticklabels(('UAR', 'F1'))
    ax.legend()

    # fig.tight_layout()
    plt.ylim(0, 100)
    plt.show()