import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import numpy as np

def plot_histograms(related, unrelated, filename):
    fig = plt.figure()
    bins = np.linspace(0, 1, 100)
    plt.hist(related, bins, label='related')
    plt.hist(unrelated, bins, label='unrelated')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 1, 0.1))
    fig.savefig(filename, format='svg', dpi=1200)


def plot_histograms_word_mover_distance(related, unrelated, filename):
    fig = plt.figure()
    bins = np.linspace(0, 2, 100)
    plt.hist(related, bins, label='related')
    plt.hist(unrelated, bins, label='unrelated')
    plt.legend(loc='upper right')
    plt.xticks(np.arange(0, 2, 0.2))
    fig.savefig(filename, format='svg', dpi=1200)